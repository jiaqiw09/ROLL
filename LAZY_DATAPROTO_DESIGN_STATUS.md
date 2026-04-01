# LazyDataProto 设计与当前状态

## 1. 设计目标

这套改动的核心目标只有一条：

**主 batch 在 controller / pipeline 侧应当以 `LazyDataProto` 为主态流转。**

展开来说，就是下面几件事：

1. rollout / generate 刚产出的结果可以先是 `DataProto`
2. 一旦进入 replay / TQ，主对象应当切到 `LazyDataProto`
3. scheduler 返回给 `rlvr_pipeline` 的主 batch，应当还是 `LazyDataProto`
4. pipeline 内部允许本地读字段、写字段、做 reward / advantage / reorder / group_by
5. 这些局部操作不应该轻易把主 batch 打回 eager
6. dispatch 到 worker 时，不直接传大 tensor，而是传 TQ handle
7. worker 返回结果时，也尽量继续走 handle / lazy 路线，避免 controller 侧重新拉回大 tensor

一句话就是：

- 主对象主态：`LazyDataProto`
- worker 边界句柄：`BatchMeta`
- worker 内部计算：`DataProto`

## 2. 三种对象各自代表什么

这里最容易混，所以单独说清楚。

### 2.1 `LazyDataProto`

`LazyDataProto` 是主对象。

它表示：

1. 底层仍然挂着 TQ / KV 数据
2. 本地可以只有部分字段缓存
3. 本地也可以带新增或更新过的字段
4. 后续还能继续 `group_by`、`reorder`、`chunk`、`dispatch`

所以看到 `LazyDataProto` 时，应该理解为：

**“这还是主 batch，本体还在 lazy 语义里。”**

### 2.2 `BatchMeta`

`BatchMeta` 是 worker 边界的句柄。

它表示：

1. controller 不直接把大 tensor 通过 Ray 发过去
2. 而是把 TQ 中这批数据的句柄发给 worker
3. worker 再在本地把它转成 `DataProto` 来算

所以看到 `BatchMeta` 时，应该理解为：

**“这是跨 worker 边界的传输形态，不是主对象主态。”**

### 2.3 `DataProto`

`DataProto` 是 eager 计算视图。

它表示：

1. 当前这一步需要本地真实 tensor 来算
2. 或者这是某个局部中间结果
3. 它不一定是主 batch

所以看到 `DataProto` 时，不要立刻理解成“lazy 失败了”。  
要先问一句：

**“这个 `DataProto` 是主 batch 吗，还是局部计算窗口？”**

## 3. 我们希望的标准数据流

当前理想的数据流是：

1. generate / rollout 结果产生：`DataProto`
2. 写入 replay / TQ：`LazyDataProto`
3. scheduler 返回给 pipeline：`LazyDataProto`
4. pipeline 主 batch 持续流转：`LazyDataProto`
5. dispatch 前：`LazyDataProto`
6. dispatch 边界：`BatchMeta`
7. worker 内部计算：`DataProto`
8. worker 返回 collect 后：重新收成 `LazyDataProto`
9. 后续继续并回主 batch：仍然是 `LazyDataProto`

真正重要的是第 3、4、8、9 步。

因为这几步决定：

**主 batch 有没有在 controller 侧持续保持 lazy。**

## 4. 当前实现语义

按照现在这版代码，主语义是：

### 4.1 `LazyDataProto` 支持本地 cache

现在 `materialize()` 更接近：

- 把某些字段拉到本地缓存

而不是：

- 把整个对象永久变成 eager

所以：

- `lazy_backed=True` 表示它底层还挂着 TQ / KV
- `materialized=True` 只表示当前本地字段已经较完整
- 这不等于它已经变成 `DataProto`

### 4.2 `LazyDataProto` 支持本地 overlay

现在主 batch 上直接写：

```python
batch.batch["field"] = value
```

是允许的。

这类写入会被当成：

1. 本地新增字段
2. 或本地 authoritative 字段

后续这些字段会随着：

- `select_idxs`
- `group_by`
- `reorder`
- `chunk`

继续流转，而不是直接丢掉。

### 4.3 flush 的主要时机在 dispatch 前

当前主要规则是：

1. 本地 dirty / authoritative 字段先留在 `LazyDataProto` 本地
2. 到 `prepare_for_remote()` 时，再 flush 到 TQ
3. 然后转成 `BatchMeta` 发给 worker

所以：

- 平时主对象还是 `LazyDataProto`
- 真正跨 worker 时才转句柄

## 5. 当前实际跑通后的状态

基于最新跑通的 `b.log`，当前已经能明确确认：

1. `scheduler -> pipeline` 主入口是 `LazyDataProto`
2. `generate_output` 合并后仍然是 `LazyDataProto`
3. dispatch 前主 batch 仍然是 `LazyDataProto`
4. dispatch 边界是 `BatchMeta`
5. worker 内部计算是 `DataProto`
6. worker 返回结果 collect 后，已经可以重新回到 `LazyDataProto`
7. reward 后主 batch 仍然是 `LazyDataProto`

所以当前主链已经基本符合设计：

- 主对象主态：`LazyDataProto`
- worker 边界：`BatchMeta`
- worker 内部：`DataProto`

## 6. 如何看日志里的类型

后面看日志时，建议只按下面这个规则判断。

### 6.1 这些点最重要

如果下面这些日志点是 `LazyDataProto`，说明主链基本是对的：

1. `after_scheduler_get_batch`
2. `after_generate_output_concat`
3. `before_ref_compute_log_probs_reference`
4. `before_group_by_domain_reward`
5. `after_reward_concat`

因为这些点代表的都是主 batch。

### 6.2 这些点出现 `DataProto` 不一定有问题

下面这些地方出现 `DataProto`，不一定代表设计退化：

1. worker 内部计算
2. scheduler 内部局部拼装
3. 某个 helper 的局部 eager 计算窗口

因为这里的 `DataProto` 可能只是短暂中间态。

### 6.3 真正的判断标准

不要只看“有没有 `DataProto`”，而要看：

1. 主 batch 在离开这个阶段后，是否仍然是 `LazyDataProto`
2. 后续继续传递的对象，是否仍然是 `LazyDataProto`

也就是说：

- 中间出现局部 `DataProto` 可以接受
- 主对象掉回 `DataProto` 才是大问题

## 7. 当前还剩什么问题

当前主链已经基本通了，剩余问题主要是两类：

1. 某些字段在 TQ KV backend 下仍有 1D / 2D warning
   - `scores`
   - `response_level_rewards`
   - `prompt_id`

2. 后续还可以继续优化：
   - 哪些中间结果值得继续保持 lazy
   - 哪些局部结果保持 eager 也完全可以接受

## 8. 当前结论

现在可以把这套实现简单理解成下面这句话：

**`LazyDataProto` 是 controller / pipeline 侧主 batch 的主态，`BatchMeta` 是 worker 边界句柄，`DataProto` 是局部 eager 计算视图。**

这也是当前读代码、看日志、判断问题时最应该坚持的视角。
