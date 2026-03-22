"""
Pipeline-level TQ utilities for converting DataProto <-> KVBatchMeta.

These functions are used by the main pipeline controller (e.g. RLVRPipeline)
to write DataProto data into TransferQueue and convert to KVBatchMeta for
subsequent compute steps that operate through the @tqbridge decorator.

Usage (Step 4 stub — data still flows through the controller once):

    # After Scheduler produces DataProto
    batch: DataProto = ray.get(scheduler.get_batch.remote(...))

    # Write batch to TQ, get KVBatchMeta reference
    kv_meta = dataproto_to_kv_batch_meta(batch, partition_id="train")

    # Subsequent steps receive KVBatchMeta — @tqbridge on workers
    # fetches data directly from TQ, not via Ray RPC
    ref_result = self.reference.compute_log_probs(kv_meta)
    # ref_result is also KVBatchMeta (written back by @tqbridge)

    # To read specific fields back as DataProto for local computation
    advantage_data = kv_batch_meta_to_dataproto(kv_meta, fields=[...])
"""

import os
import uuid
import logging
from typing import List, Optional

import transfer_queue as tq
from transfer_queue import KVBatchMeta

from roll.distributed.scheduler.protocol import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ROLL_LOGGING_LEVEL", "WARN"))

_TQ_INITIALIZED = False


def fmt_kv_meta(meta) -> str:
    """Format KVBatchMeta internals for debug printing."""
    if meta is None:
        return "KVBatchMeta(None)"
    n = len(meta.keys) if meta.keys else 0
    keys_head = meta.keys[:3] if n > 3 else meta.keys
    tags_head = (meta.tags or [])[:2]
    extra = meta.extra_info or {}
    extra_summary = {}
    for k, v in extra.items():
        if isinstance(v, dict):
            extra_summary[k] = f"dict({len(v)} keys: {list(v.keys())[:5]})"
        elif isinstance(v, list):
            extra_summary[k] = f"list(len={len(v)})"
        else:
            extra_summary[k] = repr(v)[:80]
    return (
        f"KVBatchMeta(\n"
        f"  num_keys={n}, keys_sample={keys_head}{'...' if n > 3 else ''},\n"
        f"  partition_id='{meta.partition_id}',\n"
        f"  fields={meta.fields},\n"
        f"  tags_sample={tags_head}{'...' if n > 2 else ''},\n"
        f"  extra_info={extra_summary}\n"
        f")"
    )


def fmt_dataproto(dp) -> str:
    """Format DataProto internals for debug printing."""
    if dp is None:
        return "DataProto(None)"
    batch_keys = list(dp.batch.keys()) if dp.batch is not None else []
    batch_shape = list(dp.batch.batch_size) if dp.batch is not None else None
    batch_dtypes = {}
    if dp.batch is not None:
        for k in batch_keys[:10]:
            t = dp.batch[k]
            batch_dtypes[k] = f"{list(t.shape)},{t.dtype}"
    ntb_keys = list(dp.non_tensor_batch.keys()) if dp.non_tensor_batch else []
    mi_keys = list(dp.meta_info.keys()) if dp.meta_info else []
    return (
        f"DataProto(\n"
        f"  batch_shape={batch_shape}, batch_keys({len(batch_keys)})={batch_keys[:8]}{'...' if len(batch_keys) > 8 else ''},\n"
        f"  batch_dtypes_sample={batch_dtypes},\n"
        f"  non_tensor_keys={ntb_keys},\n"
        f"  meta_info_keys={mi_keys}\n"
        f")"
    )


def _ensure_tq_init():
    global _TQ_INITIALIZED
    if not _TQ_INITIALIZED:
        tq.init()
        _TQ_INITIALIZED = True


def init_tq(config=None):
    """Initialize TransferQueue with optional config.

    Should be called once at pipeline startup.
    """
    global _TQ_INITIALIZED
    if not _TQ_INITIALIZED:
        tq.init(config)
        _TQ_INITIALIZED = True


def dataproto_to_kv_batch_meta(
    data: DataProto,
    partition_id: str = "train",
    key_prefix: str = "",
    tags: Optional[List[dict]] = None,
) -> KVBatchMeta:
    """Write a DataProto's batch to TransferQueue and return a KVBatchMeta handle.

    Args:
        data: DataProto with a non-empty batch TensorDict.
        partition_id: TQ partition to write into.
        key_prefix: Optional prefix for generated keys.
        tags: Per-sample tag dicts (e.g. [{"domain": "math"}]).
              Defaults to empty dicts.

    Returns:
        KVBatchMeta referencing the data just written to TQ.
    """
    _ensure_tq_init()

    batch_size = len(data)
    keys = [f"{key_prefix}{uuid.uuid4().hex}" for _ in range(batch_size)]
    if tags is None:
        tags = [{}] * batch_size

    if data.batch is not None and data.batch.batch_size[0] > 0:
        tq.kv_batch_put(
            keys=keys,
            fields=data.batch,
            tags=tags,
            partition_id=partition_id,
        )

    kv_meta = KVBatchMeta(
        keys=keys,
        tags=tags,
        partition_id=partition_id,
        fields=list(data.batch.keys()) if data.batch is not None else [],
        extra_info=data.meta_info,
    )
    print(f"[TQ] dataproto_to_kv_batch_meta: DataProto → KVBatchMeta\n"
          f"  IN:  {fmt_dataproto(data)}\n"
          f"  OUT: {fmt_kv_meta(kv_meta)}")
    return kv_meta


def kv_batch_meta_to_dataproto(
    meta: KVBatchMeta,
    fields: Optional[List[str]] = None,
) -> DataProto:
    """Read fields from TQ and return a DataProto.

    Used when the pipeline controller needs to perform local computation
    (e.g. advantage calculation) on data that lives in TQ.

    Args:
        meta: KVBatchMeta handle.
        fields: Specific fields to fetch. None fetches all available fields.

    Returns:
        DataProto with the fetched TensorDict as batch.
    """
    _ensure_tq_init()

    fetch_fields = fields if fields is not None else meta.fields
    td = tq.kv_batch_get(
        keys=meta.keys,
        partition_id=meta.partition_id,
        fields=fetch_fields,
    )

    data = DataProto(batch=td, meta_info=meta.extra_info or {})
    print(f"[TQ] kv_batch_meta_to_dataproto: KVBatchMeta → DataProto\n"
          f"  IN:  {fmt_kv_meta(meta)}\n"
          f"  OUT: {fmt_dataproto(data)}")
    return data


def merge_kv_batch_metas(kv_metas: List[KVBatchMeta]) -> KVBatchMeta:
    """Merge multiple KVBatchMeta objects into one (same partition_id expected).

    Used to combine per-domain KVBatchMeta returned by separate schedulers
    into a single handle for downstream worker calls.
    """
    if len(kv_metas) == 1:
        print(f"[TQ] merge_kv_batch_metas: single KVBatchMeta, skip merge\n"
              f"  {fmt_kv_meta(kv_metas[0])}")
        return kv_metas[0]

    all_keys = []
    all_tags = []
    partition_id = kv_metas[0].partition_id
    all_fields: set = set()
    merged_extra: dict = {}

    for i, kv_meta in enumerate(kv_metas):
        assert kv_meta.partition_id == partition_id, (
            f"Cannot merge KVBatchMeta with different partition_ids: "
            f"{kv_meta.partition_id} vs {partition_id}"
        )
        all_keys.extend(kv_meta.keys)
        all_tags.extend(kv_meta.tags)
        all_fields.update(kv_meta.fields or [])

    merged = KVBatchMeta(
        keys=all_keys,
        tags=all_tags,
        partition_id=partition_id,
        fields=list(all_fields),
        extra_info=merged_extra,
    )
    print(f"[TQ] merge_kv_batch_metas: {len(kv_metas)} → 1\n"
          f"  OUT: {fmt_kv_meta(merged)}")
    return merged


def kv_batch_meta_put_fields(
    meta: KVBatchMeta,
    data: DataProto,
    field_keys: Optional[List[str]] = None,
):
    """Write specific fields from a DataProto back to TQ.

    Used when the pipeline controller computes something locally
    (e.g. advantages) and needs to store the results in TQ.

    Args:
        meta: KVBatchMeta handle.
        data: DataProto whose batch contains the fields to write.
        field_keys: Specific field names to write. None writes all batch keys.
    """
    _ensure_tq_init()

    if data.batch is None or data.batch.batch_size[0] == 0:
        return

    if field_keys is not None:
        td_to_write = data.batch.select(*field_keys)
    else:
        td_to_write = data.batch

    from roll.utils.transferqueue_utils import kv_batch_meta_put_tensordict
    kv_batch_meta_put_tensordict(meta, td_to_write, func_name="kv_batch_meta_put_fields")
    written_fields = field_keys or list(data.batch.keys())
    field_shapes = {k: list(data.batch[k].shape) for k in written_fields if k in data.batch.keys()}
    print(f"[TQ] kv_batch_meta_put_fields: wrote back to TQ\n"
          f"  target: {fmt_kv_meta(meta)}\n"
          f"  written_fields={written_fields}, shapes={field_shapes}")
