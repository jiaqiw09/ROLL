# Ray 与 TQ 路径性能对比结论

## 1. 说明

这份结论基于下面两份日志：

1. [`ray.log`](/Users/humphrey/Documents/github/newtq/ROLL/ray.log)
2. [`tq.log`](/Users/humphrey/Documents/github/newtq/ROLL/tq.log)

需要先说明一点：

**这两份日志不是完全同一轮 run。**

因此，这里的结论更适合用于判断：

1. Ray 路径和 TQ 路径的时间主要花在哪
2. 哪一段更容易在后期机器数、数据量增大后被放大

而不适合把所有指标都当成严格 A/B 实验结果。

## 2. 核心结论

当前日志已经能说明：

1. `TQ` 路径会增加一点 dispatch 前置成本
2. 但 `TQ` 路径明显降低了 collect / `ray_get` 这段等待时间
3. 真正的大头不是 controller 本地 `concat`
4. 真正值得关注的是 worker 返回前后的整体等待

一句话概括就是：

**TQ 的收益不在“发出去更快”，而在“收回来更轻”。**

## 3. 关键对比表

| 项目 | 纯 Ray | TQ | 结论 |
|---|---:|---:|---|
| reference dispatch | 0.041s | 0.313s | TQ 前置更重 |
| reference `ray_get` | 17.80s | 15.73s | TQ 少等约 2.07s |
| reference total | 17.86s | 16.04s | TQ 快约 10% |
| actor compute submit/dispatch | 0.046s | 0.243s | TQ 前置更重 |
| actor compute collect `ray_get` | 9.45s | 6.95s | TQ 少等约 2.50s |
| actor compute collect total | 9.46s | 6.95s | TQ 快约 27% |
| actor train submit/dispatch | 0.056s | 0.261s | TQ 前置更重 |
| actor train collect `ray_get` | 44.10s | 38.85s | TQ 少等约 5.25s |
| actor train collect total | 44.10s | 38.85s | TQ 快约 12% |
| `step_train` | 44.24s | 39.38s | TQ 快约 11% |

## 4. 如何理解这些数字

### 4.1 dispatch 为什么 TQ 反而更慢

这是因为 TQ 路径在 dispatch 前多做了这些事情：

1. `prepare_for_remote`
2. flush 本地 authoritative 字段
3. `LazyDataProto -> KVBatchMeta -> BatchMeta`

所以前置时间会比纯 Ray 略高，这是正常现象。

### 4.2 collect 为什么 TQ 更快

这是这次对比里最重要的结论。

当前日志显示：

1. 纯 Ray 路径下，worker 返回阶段等待更长
2. TQ 路径下，这部分等待明显下降
3. 下降最明显的是：
   - actor `compute_log_probs`
   - actor `train_step`

这说明 TQ 已经开始把“结果回传 / collect”这一段的成本压下来了。

## 5. 哪些不是瓶颈

从当前日志看，这几类时间都不重：

1. controller 本地 `concat`
2. controller 本地 collect 合并
3. reward 本地后处理

尤其是 `collect/concat`，目前几乎都是毫秒级，可以基本忽略。

所以当前不是：

- controller 拼接太慢
- 或者本地 union / concat 太慢

真正的大头还是远端执行完成并返回可 collect 结果之前的等待。

## 6. 哪些是后期扩容最值得关注的

如果后面机器数和数据量增大，最值得关注的是：

1. `time/controller/*/collect/ray_get`
2. `time/step_train`
3. `time/ref_log_probs_values`

原因是：

1. 这些已经是当前时间大头
2. 它们比 dispatch 更容易随 worker 数、batch 大小、token 长度而放大

所以从扩容风险角度看：

- `dispatch` 不是当前主风险
- `collect/ray_get` 才是更可能被放大的部分

## 7. 当前判断

结合这两份日志，当前可以下一个比较明确的判断：

1. `TQ` 路线是有价值的
2. 它虽然增加了一点 dispatch 前置开销
3. 但已经明显降低了 collect / 返回等待成本
4. 而后期扩容时，真正危险的恰恰是返回等待这一段

因此，从后期扩容视角看：

**继续沿 `LazyDataProto + TQ + BatchMeta` 这条路线优化，是合理的。**

## 8. 需要注意的边界

这次对比里有一个指标不能直接下严格结论：

- `step_generate`

因为两份日志里：

- Ray: `97.79s`
- TQ: `0.32s`

这个差异过大，更像是两轮运行的生成过程状态不同，而不是单纯由 Ray/TQ 传输路径导致。

所以后面如果要做更严格的性能结论，建议：

1. 用相同配置
2. 相同数据
3. 相同 step
4. 分别跑一版 Ray 路径和 TQ 路径

再重点对比下面这些指标：

1. `time/controller/*/dispatch`
2. `time/controller/*/collect/ray_get`
3. `time/controller/*/collect/concat`
4. `time/step_train`
5. `time/ref_log_probs_values`
