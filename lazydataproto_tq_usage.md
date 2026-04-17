# LazyDataProto 与 TQ (TransferQueue) 使用机制说明

## 0. 设计思路与继承关系

`LazyDataProto` 直接继承自 `DataProto`。

## 1. LazyDataProto 核心字段解析

| 字段名                           | 类型                      | 作用说明                                                                                             |
| :---------------------------- | :---------------------- | :----------------------------------------------------------------------------------------------- |
| `_kv_meta`                    | `KVBatchMeta`           | 核心句柄。记录了这批数据在 TQ 远端存储中的身份信息，包含 `keys` 样本 ID 列表、`partition_id` 路由 ID、以及远端可用的 `fields` Tensor 字段名。 |
| `batch`                       | `TensorDict`            | 本地 Tensor 缓存。通过代理拦截读取操作，当试图读取本地没有的 Tensor 时，自动触发去 TQ 拉取。                                         |
| `non_tensor_batch`            | `Dict[str, np.ndarray]` | 非 Tensor 数据。如 prompt 文本串、uuid 等。这部分数据很轻，始终保存在本地，不走 TQ。                                           |
| `meta_info`                   | `Dict[str, Any]`        | 随 `LazyDataProto` 一起流转的全局元数据，如 `global_step`、`metrics` 等。                                        |
| `_tensor_field_names`         | `List[str]`             | 逻辑 Tensor 字段名。记录这批数据“逻辑上”拥有哪些 Tensor，包括 TQ 里的和本地新增的。                                             |
| `_local_authoritative_fields` | `Set[str]`              | 本地权威字段。记录在本地新增或修改过的 Tensor 字段名，例如刚计算完的 `advantages`。这些字段在跨节点传输前必须 flush 到 TQ 中。                  |
| `_local_cached_fields`        | `Set[str]`              | 本地已缓存字段。记录从 TQ 拉取下来并缓存在本地 `batch` 中的 Tensor 字段名，避免重复拉取。                                          |
| `_materialized`               | `bool`                  | Materialize 标记。当逻辑上所有的 Tensor 字段都已经存在本地 `batch` 中时为 `True`，否则为 `False`。                          |


## 2. 批处理操作的 Lazy 实现

`LazyDataProto` 之所以能在 Pipeline 中无缝替换 `DataProto`，是因为它**复刻了** **`DataProto`** **常用的批处理 API**（如 `chunk`, `concat`, `select`, `reorder`, `group_by`），但实现方式完全不同：

- **Eager 模式下**：`concat` 或 `chunk` 会直接调用 PyTorch 的 API 去拼接或切分真实的 Tensor，伴随着大量的显存申请和数据搬运。
- **Lazy 模式下**：在 `LazyDataProto` 中调用这些操作时，**底层实际上只对** **`_kv_meta.keys`（样本 ID 列表）进行切分或拼接**。
  - 例如，`concat` 两个 `LazyDataProto` 只是把它们的 keys 列表拼在一起。
  - `chunk` 只是把 keys 列表切成几份，生成几个新的 `LazyDataProto`。

### 已实现支持度速览


| 功能 / 操作                | 当前实现状态                   | 机制说明                                                                                                                  |
| :--------------------- | :----------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| **`concat`**           | 🟢 **纯 Lazy + 自动 Flush** | 如果参与拼接的 `LazyDataProto` 身上有本地新增的 Tensor，**会先触发自动 flush 将它们推入 TQ**。大家全变干净后，再直接合并底层的 `kv_meta.keys` 列表，完成无 Tensor 拷贝拼接。 |
| **`chunk`**            | 🟢 **纯 Lazy + 自动 Flush** | 同样会在切分前触发 flush 洗白数据，然后对 `kv_meta.keys` 列表进行等分切片，返回一组全新的 `LazyDataProto` 句柄。                                          |
| **`select`**           | 🟢 **纯 Lazy 支持**         | 根据给定的 index，从 `kv_meta.keys` 列表中挑选出对应的 ID 组合成新句柄。                                                                     |
| **`reorder`**          | 🟢 **纯 Lazy 支持**         | 根据新的 index 顺序，打乱重排 `kv_meta.keys`。                                                                                    |
| **`group_by`**         | 🟢 **纯 Lazy 支持**         | 将批次按指定维度分组，返回多个按组切分的 `LazyDataProto` 子句柄。                                                                             |
| **`batch[key] = val`** | 🟢 **支持（Local Write）**   | 允许像字典一样直接塞新 Tensor 进去（例如算完的 reward），触发后续的自动 flush 机制。                                                                 |


## 3. 核心机制：Local Write 与 Flush 配合

在 Pipeline 的执行过程中，Controller 端通常需要将来自 Actor 和 Critic 的数据合并，并在本地计算产生新的 Tensor（如 `advantages` 和 `returns`）。由于原始数据存储在 TQ 中，新产生的 Tensor 需要一套机制同步到 TQ。

`LazyDataProto` 通过内部的两个状态集合（`_local_cached_fields` 与 `_local_authoritative_fields`）配合自动 Flush 机制，共同管理 Tensor 的生命周期：

1. **按需拉取与缓存标记**：当计算逻辑需要访问 TQ 中已有的数据（如读取 `old_log_probs`）时，代理层从 TQ 拉取真实的 Tensor 数据存入本地的 `batch` (TensorDict) 中，并将该字段的**名称**（字符串）加入 `_local_cached_fields` 集合。这表明 `batch` 里的该数据是远端的镜像，随时可以丢弃，不需要同步回远端。
2. **直接本地写**：通过 `batch.batch["advantages"] = adv_tensor` 进行赋值操作。
3. **标记 authoritative**：代理层捕获到新的 Tensor 后，会将对应的字段名（如 `"advantages"`）加入 `_local_authoritative_fields` 集合。这表示该字段为本地新增数据，本地是该数据的 authoritative 数据源（此时 TQ 里还没有）。
4. **携带标记流转**：带有 authoritative 标记的 `LazyDataProto` 可以继续执行 `group_by`、`reorder` 等元数据操作。
5. **触发 Flush**：当该对象执行 `concat`，或准备通过 Ray 派发给 Worker（如触发 `prepare_for_remote` / `chunk`）时，框架会检查并触发 `_flush_authoritative_to_tq()`。
6. **推入 TQ 并状态转移**：该方法将本地的 `advantages` Tensor 推入 TQ 显存池中。同步完成后，框架会将 `"advantages"` 从 `_local_authoritative_fields` 集合中**移除（清空 authoritative 标记）**，并将其加入到 `_local_cached_fields` 缓存集合中。此时，对象恢复为纯元数据句柄状态，且本地 `batch` 中保留了该 Tensor 作为缓存副本。

该方案后续还需要优化此处缓存，脏数据，以及nontensor的维持情况，当前操作比较冗余。


## 4. 在 `rlvr_pipeline.py` 中的数据流转过程

在 `RLVRPipeline.run()` 中，数据经历了“生成 -> 评价 -> 训练”的完整生命周期：

### 阶段一：采样生成 (Generate)

1. **获取数据**：`generate_schedulers` 从 `actor_infer` 获取采样结果。
2. **返回 LazyDataProto**：此时返回给 Pipeline 的 `domain_batch` 是一个 `LazyDataProto`。
   - **状态**：大块的生成 Tensor（如 `responses`, `logits`）都在 Infer 节点的 TQ 里。Pipeline 本地只拿到了 `_kv_meta` 和轻量的文本 `non_tensor_batch`。
3. **Concat**：Pipeline 执行 `BatchData([...]).concat()` 时，因为大家都是 `LazyDataProto`，底层只是简单把 `_kv_meta.keys` 合并在一起，**完全没有发生真实的 Tensor 拼接和显存拷贝**。

### 阶段二：LogProbs 与 Reward 计算 (Read & Write)

1. **读取**：
   - Pipeline 将 `LazyDataProto` 传给 Reference 或 Actor\_train 计算 `old_log_probs`。
   - 当这些模块需要用到 `responses` 等 Tensor 时，触发 `_TrackedTensorDictProxy`，通过 `tq.kv_batch_get` 自动把需要的 Tensor 拉取到模块本地。拉取后，字段名加入 `_local_cached_fields`。
2. **写入**：
   - Pipeline 本地计算 Advantage 和 Reward（如 `compute_advantage`）。
   - 计算结果（如 `advantages`, `returns`）直接赋值给本地 `batch`（例如 `batch.batch["advantages"] = ...`）。
   - 此时，这些新字段会被加入 `_local_authoritative_fields`（标记为脏数据），说明 TQ 里还没有这些新数据。

### 阶段三：训练分发 (Train)

1. **触发 Flush**：
   - 在将 batch 派发给 `actor_train.train_step` 之前（比如在内部的 `concat` 或 `prepare_for_remote` 阶段），会触发 `_flush_authoritative_to_tq()`。
   - Pipeline 会把 `_local_authoritative_fields` 里的本地 Tensor（如 `advantages`）推送到 TQ 远端存储中，并清空 authoritative 标记。
2. **下发句柄**：
   - Pipeline 把更新过 `fields` 的 `_kv_meta` 发给训练节点的 Worker。
   - 训练 Worker 拿到句柄后，从本地（或跨机）TQ 高速拉取完整的训练 Tensor，执行反向传播。

## 4. 全流程与 `tqbridge` 及组件交互

在 Controller 与异构 Worker 集群（如 Actor、Reward）通过 Ray RPC 进行通信的底层，`LazyDataProto` 与 `tqbridge` (TransferQueue 桥接器) 合作。

### 1) 数据分发 (Dispatch)

- **Controller 端**：当 Pipeline 调用 `actor.compute_log_probs(batch)` 时，`batch`（通常是一个 `LazyDataProto`）会进入 RPC 分发流程。
- **拆分与预处理**：`LazyDataProto.chunk()` 被调用，将大 batch 切分为多块。此时如果 `_local_authoritative_fields` 有本地新增的 Tensor，会触发 `prepare_for_remote()` 将这些 Tensor 强制 flush 到 TQ 中。
- **剥离并转为 BatchMeta**：在通过 Ray 将参数传给 Worker 之前，`LazyDataProto` 会剥离掉所有 Tensor，提取出底层的元数据，并**转换为** **`BatchMeta`**（TQ 原生的传输句柄）。因此，Ray RPC 链路上完全不包含任何显存/内存 Tensor，只有轻量的字符串和指针。
- **Worker 接收**：Worker 侧的 `@tqbridge`（或其他相关 decorator）接收到 `BatchMeta` 句柄后，会根据需要将其转回 `DataProto`（通过去 TQ 拉取数据）传递给真实的执行函数。

### 2) 数据收集 (Collect)

- **Worker 计算与拦截**：Worker 计算出新结果（如 `log_probs` Tensor）后，`@tqbridge` 装饰器会拦截函数的返回值。
- **存入 TQ**：装饰器底层（如 `_update_meta_with_output`）会将新增的 Tensor 写入 Worker 本地的 TQ 节点。
- **返回 BatchMeta 句柄**：写入完成后，装饰器会**返回一个更新了字段（fields）的** **`BatchMeta`** **句柄**，通过 Ray RPC 传回给 Controller。
- **Controller 接收与重组**：Controller 收到这批 Worker 返回的 `BatchMeta` 句柄后，在 `concat()` 阶段将它们重新合并，并**转换回** **`LazyDataProto`**（内部包含转回的 `KVBatchMeta`）。至此，全流程的 Collect 完成，Tensor 依旧安安静静躺在分布式的 TQ 显存池里。
