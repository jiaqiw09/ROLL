# LazyDataProto 设计目标与当前实现状态

## 1. 目标设计

当前想要实现的整体设计是：

1. rollout / generate 结果最开始可以是 `DataProto`，但一旦进入 replay/TQ，就应该以 `LazyDataProto` 为主。
2. 从 scheduler 返回到 `rlvr_pipeline` 的主对象，应该保持为 `LazyDataProto`。
3. 在 pipeline 内部，允许本地读取 tensor，也允许局部物化，但“本地读/本地缓存”不应该自动破坏 lazy 身份。
4. 现有 `DataProto` 风格写法必须继续兼容，例如：
   - `batch.batch["field"] = value`
5. 本地字段状态要区分：
   - `clean cache`：只是从 TQ 拉到本地，但没有修改
   - `authoritative local fields`：本地新写入或本地更新过的字段
6. flush 到 TQ 的主要时机，应该放在 remote dispatch 边界：
   - controller 侧主态仍是 `LazyDataProto`
   - dispatch 边界转为 `BatchMeta`
   - worker 内部使用 `DataProto`
7. worker 返回结果时，也尽量避免大 tensor 走 Ray 直传，优先保留 TQ/handle 路线。

## 2. 当前 controller / worker 边界

当前已经实现的边界是：

1. controller / pipeline 主态：`LazyDataProto`
2. controller 内部 lazy 存储与拼接底座：`KVBatchMeta`
3. dispatch 边界：`LazyDataProto -> KVBatchMeta -> BatchMeta`
4. worker 输入：`BatchMeta`
5. worker 内部真正计算时：`DataProto`
6. worker 返回结果 collect：
   - reference 分支已经能回到 controller 侧形成 `LazyDataProto`
   - 部分 actor 相关 collect 目前仍然会落到 `DataProto`

也就是说，controller / worker 分层基本符合预期，但 worker 结果收集还没有完全统一成 lazy。

## 3. 已经实现正确的部分

根据最新 `b.log`，下面这些已经是对的：

1. `scheduler -> pipeline` 入口已经是 `LazyDataProto`
   - `rlvr.after_scheduler_get_batch` 是 `LazyDataProto`
   - `rlvr.after_generate_output_concat` 是 `LazyDataProto`
2. `generate_output` 在 pipeline 主对象里继续保持 `LazyDataProto`
3. dispatch 前输入 worker 的对象主态仍然是 `LazyDataProto`
4. dispatch 边界按预期转成 `BatchMeta`
5. `batch.batch["field"] = value` 这类 `DataProto` 风格写法已经开始被 lazy 层追踪
6. `group_by("domain")` 现在已经能返回 `LazyDataProto` 子 batch
7. 像 `old_log_probs`、`prompt_id`、`ref_log_probs`、`token_level_rewards` 这类本地字段，比之前更能在 pipeline 中被保留下来

## 4. 当前仍然偏离目标的地方

目前和目标设计相比，主要还有这些偏离：

1. actor compute-log-probs 返回结果 collect 之后，仍然是：
   - `rlvr.after_actor_compute_log_probs_concat -> DataProto`
   这说明 actor 结果收集路径还没有完全 lazy / TQ 化。

2. reward window 后的合并仍然会 fallback 到 eager：
   - `protocol.lazy.concat.fallback_eager`
   - `rlvr.after_reward_concat -> DataProto`
   这是当前最主要的偏离点。

3. 一旦 reward concat fallback 之后，后续 dispatch 也会跟着使用 `DataProto`：
   - `decorator.prepare_for_remote.input -> DataProto`

所以当前状态可以概括为：

- reward 之前的主链基本已经符合 lazy 设计
- reward 之后的 concat 还会掉回 eager

## 5. 当前 LazyDataProto 的语义

当前实现中，希望 `LazyDataProto` 具有下面这些语义：

1. `materialize()` 更接近“本地缓存字段”，不再等价于“彻底变成 eager”。
2. `LazyDataProto` 同时持有：
   - TQ/KV 底座（`_kv_meta`）
   - 本地 cache
   - 本地 authoritative fields
3. 直接通过 `batch.batch[...]` 的读写会被追踪
4. 通过 `batch.batch[...]` 读取缺失字段时，可以按需从 TQ 自动补拉

当前几个关键状态的理解方式：

- `lazy_backed=True`：仍然有 TQ / KV 底座
- `materialized=True`：当前已知 tensor 字段已经在本地齐备
- 但这不等于对象已经变成 `DataProto`

## 6. 当前最主要的剩余问题

现在最需要继续解决的是：

1. reward 阶段结束后的 concat，尽量继续保持 lazy，而不是 fallback 到 `DataProto`

更具体地说：

1. domain batch 应继续保持 `LazyDataProto`
2. reward 阶段产生的本地 overlay 字段，应继续挂在 lazy 对象上流转
3. `BatchData(batch_list).concat()` 在 reward 后应该尽量避免 eager fallback

## 7. 简短总结

当前整体状态是：

1. 你的 lazy 设计已经实现了一大半。
2. 最关键的前半段要求已经成立：
   - scheduler 输出进入 `rlvr_pipeline` 时已经是 `LazyDataProto`
3. worker 边界也已经基本按设计工作：
   - controller 用 `LazyDataProto`
   - worker 边界用 `BatchMeta`
4. 当前最主要还没收完的地方是：
   - actor collect 仍然有 eager
   - reward 后 concat 明确 fallback 到 `DataProto`

这份文件当前可以作为我们后续继续推进时的对齐基准：

- 目标架构是什么
- 现在已经做到哪一步
- 还差哪一步
