# ROLL TQ 元信息流转说明（KVBatchMeta / BatchMeta / DataProto）

## 背景

在开启 `transfer_queue`（TQ）后，RLVR 流程会在 `DataProto` 与 `KVBatchMeta/BatchMeta` 之间来回转换。  
本说明用于澄清：

1. `meta_info` 是否会丢失；
2. 哪些字段应放在 `batch`，哪些应放在 `meta_info`；
3. 当前实现中已知的坑和修复方式。

---

## 一句话结论

- 输入侧 `DataProto.meta_info` 不会丢，会通过 `KVBatchMeta.extra_info` 传递并恢复。
- 逐样本数据（如 `prompt_id`、`sample_uuid`）必须放在 `batch/non_tensor_batch`，保持 batch 对齐。
- Worker 输出阶段新增的 `output.meta_info`（尤其 metrics）不会自动写回 TQ 元信息，需显式处理。
- 不要把标量 `meta_info` 写进 `TensorDict(batch)`，否则可能出现 batch 维度错误。

---

## 当前流转（TQ 模式）

1. `DataProto -> KVBatchMeta`
   - `data.batch` 写入 TQ 存储；
   - `data.meta_info` 放到 `KVBatchMeta.extra_info`。

2. `KVBatchMeta -> BatchMeta`
   - `extra_info` 继续保留到 `BatchMeta.extra_info`。

3. `BatchMeta -> DataProto`（worker 入参桥接）
   - 从 TQ 拉真实 batch 数据；
   - `extra_info` 恢复到 `DataProto.meta_info`。

4. Worker 计算输出
   - 输出的 tensor 字段会写回 TQ；
   - 但输出 `meta_info` 默认不会自动回写到 KV 元信息。

---

## 字段放置规范

### 放在 `batch` / `non_tensor_batch`（逐样本）

- 每条样本独立值；
- 第一维必须等于 batch size；
- 例：`input_ids`、`response_mask`、`prompt_id`、`sample_uuid`。

### 放在 `meta_info`（整批控制）

- 批级控制参数；
- 可为标量、list、dict；
- 例：`global_step`、`disable_adapter`、`is_offload_states`、`loss_mask_keys`。

---

## 已知问题与根因

### 现象

报错类似：

`RuntimeError: batch dimension mismatch, got self.batch_size=torch.Size([B]) and value.shape=torch.Size([])`

### 根因

在 `BatchMeta -> DataProto` 过程中，把 `meta.extra_info` 逐项写入了 `TensorDict`，其中包含标量（shape `[]`），与 batch size `[B]` 不匹配。

### 正确做法

- `extra_info` 只进入 `DataProto.meta_info`；
- 不应注入到 `TensorDict(batch)`。

---

## 建议修复点

文件：`roll/utils/transferqueue_utils.py`

函数：`_async_meta_to_realdata`

- 保留：从 TQ 拉 `tensordict`；
- 删除：把 `meta_info` 遍历写入 `tensordict` 的逻辑。

---

## FAQ

### Q1：放进 TQ 之后会丢 `DataProto.meta_info` 吗？

不会。输入侧 `meta_info` 会经由 `KVBatchMeta.extra_info` 继续传递并恢复。

### Q2：为什么我感觉有些信息“没回来”？

通常是 Worker 输出阶段新增的 `meta_info`（如 metrics）没有自动回写到 KV 元信息。  
这部分需要在 pipeline 中显式收集和回填。

---

## 排查清单

1. 确认 `transfer_queue` 包版本与代码一致（是否包含 `KVBatchMeta`）。
2. 确认 `batch` 字段都满足 batch 维度一致。
3. 检查是否把标量控制字段错误写入了 `TensorDict(batch)`。
4. 区分“输入 meta 传递”与“输出 meta 回流”两条链路。
