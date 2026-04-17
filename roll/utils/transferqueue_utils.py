"""TransferQueue utilities for ROLL.

Provides the @tqbridge decorator and BatchMeta/KVBatchMeta conversion tools,
adapted from the TQ prototype integration.

When TQ is not enabled, @tqbridge is effectively a no-op.
All transfer_queue imports are lazy so this module remains safe to import
even when the package is not installed.
"""

import asyncio
import copy
import functools
import inspect
import logging
import os
import threading
import time
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto

if TYPE_CHECKING:
    from roll.distributed.scheduler.decorator import Dispatch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ROLL_LOGGING_LEVEL", "WARN"))

TQ_INITIALIZED = False

_tq_mod = None
_BatchMeta_cls = None
_KVBatchMeta_cls = None


def _ensure_tq_imports():
    global _tq_mod, _BatchMeta_cls, _KVBatchMeta_cls
    if _tq_mod is None:
        import transfer_queue
        from transfer_queue import BatchMeta, KVBatchMeta

        _tq_mod = transfer_queue
        _BatchMeta_cls = BatchMeta
        _KVBatchMeta_cls = KVBatchMeta
    return _tq_mod, _BatchMeta_cls, _KVBatchMeta_cls


def init_tq(config=None):
    """Initialize TransferQueue once in the current process."""
    tq, _, _ = _ensure_tq_imports()
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init(config)
        TQ_INITIALIZED = True


def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="batchmeta-data converter",
        daemon=True,
    )

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()
        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()


def _find_meta(*args, **kwargs):
    _, BatchMeta, KVBatchMeta = _ensure_tq_imports()
    for arg in args:
        if isinstance(arg, (BatchMeta, KVBatchMeta)):
            return arg
    for value in kwargs.values():
        if isinstance(value, (BatchMeta, KVBatchMeta)):
            return value
    return None


def _find_batch_meta(*args, **kwargs):
    _, BatchMeta, _ = _ensure_tq_imports()
    for arg in args:
        if isinstance(arg, BatchMeta):
            return arg
    for value in kwargs.values():
        if isinstance(value, BatchMeta):
            return value
    return None


def _split_extra_info(extra_info):
    extra_info = copy.deepcopy(extra_info) if extra_info else {}
    if "meta_info" in extra_info or "non_tensor_batch" in extra_info:
        meta_info = copy.deepcopy(extra_info.get("meta_info", {}))
        non_tensor_batch = copy.deepcopy(extra_info.get("non_tensor_batch", {}))
        return meta_info, non_tensor_batch
    return extra_info, {}


def _pack_extra_info(meta_info, non_tensor_batch):
    return {
        "meta_info": copy.deepcopy(meta_info or {}),
        "non_tensor_batch": copy.deepcopy(non_tensor_batch or {}),
    }


def _apply_output_extra_info(meta, output):
    if isinstance(output, DataProto):
        meta.extra_info = _pack_extra_info(output.meta_info, output.non_tensor_batch)
    return meta


async def _async_meta_to_realdata(meta) -> TensorDict:
    tq, _, _ = _ensure_tq_imports()
    if meta.size == 0:
        return TensorDict({}, batch_size=(0,))

    tq_client = tq.get_client()
    return await tq_client.async_get_data(meta)


def _meta_to_realdata(meta) -> TensorDict:
    _, _, KVBatchMeta = _ensure_tq_imports()
    if isinstance(meta, KVBatchMeta):
        meta = kv_batch_meta2batch_meta(meta)
    return _run_async_in_temp_loop(_async_meta_to_realdata, meta)


def _meta_to_dataproto(meta) -> DataProto:
    td = _meta_to_realdata(meta)
    meta_info, non_tensor_batch = _split_extra_info(getattr(meta, "extra_info", None))
    return DataProto(batch=td, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


async def _async_meta_to_dataproto(meta) -> DataProto:
    _, _, KVBatchMeta = _ensure_tq_imports()
    if isinstance(meta, KVBatchMeta):
        meta = await async_kv_batch_meta2batch_meta(meta)
    td = await _async_meta_to_realdata(meta)
    meta_info, non_tensor_batch = _split_extra_info(getattr(meta, "extra_info", None))
    return DataProto(batch=td, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def meta_to_dataproto(meta) -> DataProto:
    return _meta_to_dataproto(meta)


async def _async_update_meta_with_output(output: TensorDict, meta, func_name=None):
    tq, _, _ = _ensure_tq_imports()
    fields = [key for key, value in output.items() if isinstance(value, torch.Tensor)]
    if not fields:
        return meta

    t1 = time.time()
    tq_client = tq.get_client()
    meta = await tq_client.async_put(data=output.select(*fields), metadata=meta)
    t2 = time.time()
    logger.info(f"Task {func_name} (pid={os.getpid()}) writing to TQ, cost: {t2 - t1:.3f}s")
    return meta


def _update_meta_with_output(output: TensorDict, meta, func_name=None):
    return _run_async_in_temp_loop(_async_update_meta_with_output, output, meta, func_name)


def _compute_need_collect(dispatch_mode: "dict | Dispatch", args: list) -> bool:
    from roll.distributed.scheduler.decorator import Dispatch

    if dispatch_mode is None or isinstance(dispatch_mode, Dispatch):
        return True

    assert isinstance(dispatch_mode, dict) and "collect_fn" in dispatch_mode
    collect_fn = dispatch_mode["collect_fn"]
    if isinstance(collect_fn, functools.partial) and len(args) >= 1 and hasattr(args[0], "query_collect_info"):
        return args[0].query_collect_info(collect_fn)
    return True


def _extract_writable_tensordict(output):
    if isinstance(output, DataProto):
        if output.batch is not None and output.batch.batch_size and output.batch.batch_size[0] > 0:
            return output.batch
        return None
    if isinstance(output, TensorDict):
        if output.batch_size and output.batch_size[0] > 0:
            return output
        return None
    return None


def _postprocess_common(output, put_data, need_collect):
    _, BatchMeta, _ = _ensure_tq_imports()
    if put_data and not need_collect:
        return BatchMeta()
    if not put_data and not need_collect and isinstance(output, DataProto):
        return DataProto()
    if not put_data and not need_collect and isinstance(output, TensorDict):
        return TensorDict({}, batch_size=(0,))
    return output


async def async_kv_batch_meta2batch_meta(meta):
    tq, _, _ = _ensure_tq_imports()
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True

    tq_client = tq.get_client()
    batch_meta = await tq_client.async_kv_retrieve_meta(
        keys=meta.keys,
        partition_id=meta.partition_id,
        create=False,
    )
    fields = meta.fields
    if fields is not None:
        if isinstance(fields, str):
            fields = [fields]
        batch_meta = batch_meta.select_fields(fields)
    batch_meta.extra_info = meta.extra_info
    return batch_meta


def kv_batch_meta2batch_meta(meta):
    return _run_async_in_temp_loop(async_kv_batch_meta2batch_meta, meta)


async def async_batch_meta2kv_batch_meta(meta):
    tq, _, KVBatchMeta = _ensure_tq_imports()
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True

    tq_client = tq.get_client()
    partition_id = meta.partition_ids[0]
    assert all(partition_id == pid for pid in meta.partition_ids)
    keys = await tq_client.async_kv_retrieve_keys(global_indexes=meta.global_indexes, partition_id=partition_id)
    return KVBatchMeta(
        keys=keys,
        tags=[{}] * meta.size,
        partition_id=partition_id,
        fields=meta.field_names,
        extra_info=meta.extra_info,
    )


def batch_meta2kv_batch_meta(meta):
    return _run_async_in_temp_loop(async_batch_meta2kv_batch_meta, meta)


def kv_batch_meta_put_tensordict(meta, td: TensorDict, func_name: str = "kv_batch_meta_put_tensordict"):
    if td is None or not td.batch_size or td.batch_size[0] == 0:
        return meta
    batch_meta = kv_batch_meta2batch_meta(meta)
    updated_batch_meta = _update_meta_with_output(td, batch_meta, func_name)
    updated_kv_meta = batch_meta2kv_batch_meta(updated_batch_meta)
    meta.fields = list(updated_kv_meta.fields) if updated_kv_meta.fields is not None else []
    meta.extra_info = updated_kv_meta.extra_info
    return meta


def tqbridge(dispatch_mode=None):
    """Bridge BatchMeta input/output with DataProto/TensorDict worker methods."""
    from roll.distributed.scheduler.decorator import _check_dispatch_mode

    _check_dispatch_mode(dispatch_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            tq, BatchMeta, _ = _ensure_tq_imports()
            batch_meta = _find_batch_meta(*args, **kwargs)
            if batch_meta is None:
                return func(*args, **kwargs)

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            io_meta = batch_meta

            args = [_meta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
            kwargs = {
                k: _meta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                for k, v in kwargs.items()
            }

            output = func(*args, **kwargs)

            writable_td = _extract_writable_tensordict(output)
            put_data = writable_td is not None
            if put_data:
                assert writable_td.batch_size[0] == io_meta.size, (
                    f"output batch_size {writable_td.batch_size} != meta size {io_meta.size}"
                )

            need_collect = _compute_need_collect(dispatch_mode, args)
            if put_data and need_collect:
                io_meta = _apply_output_extra_info(io_meta, output)
                return _update_meta_with_output(writable_td, io_meta, func.__name__)
            return _postprocess_common(output, put_data, need_collect)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            tq, BatchMeta, _ = _ensure_tq_imports()
            batch_meta = _find_batch_meta(*args, **kwargs)
            if batch_meta is None:
                return await func(*args, **kwargs)

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            io_meta = batch_meta

            args = [
                await _async_meta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg
                for arg in args
            ]
            kwargs = {
                k: await _async_meta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                for k, v in kwargs.items()
            }

            output = await func(*args, **kwargs)

            writable_td = _extract_writable_tensordict(output)
            put_data = writable_td is not None
            if put_data:
                assert writable_td.batch_size[0] == io_meta.size, (
                    f"output batch_size {writable_td.batch_size} != meta size {io_meta.size}"
                )

            need_collect = _compute_need_collect(dispatch_mode, args)
            if put_data and need_collect:
                io_meta = _apply_output_extra_info(io_meta, output)
                return await _async_update_meta_with_output(writable_td, io_meta, func.__name__)
            return _postprocess_common(output, put_data, need_collect)

        return async_inner if inspect.iscoroutinefunction(func) else inner

    return decorator


def dataproto_to_kv_batch_meta(
    data: DataProto,
    partition_id: str = "train",
    key_prefix: str = "",
    tags: Optional[List[dict]] = None,
):
    """Write a DataProto batch into TQ and return a KVBatchMeta handle."""
    tq, _, KVBatchMeta = _ensure_tq_imports()
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True

    batch_size = len(data)
    keys = [f"{key_prefix}{uuid.uuid4().hex}" for _ in range(batch_size)]
    if tags is None:
        tags = [{} for _ in range(batch_size)]

    if data.batch is not None and data.batch.batch_size[0] > 0:
        tq.kv_batch_put(
            keys=keys,
            fields=data.batch,
            tags=tags,
            partition_id=partition_id,
        )

    return KVBatchMeta(
        keys=keys,
        tags=tags,
        partition_id=partition_id,
        fields=list(data.batch.keys()) if data.batch is not None else [],
        extra_info=_pack_extra_info(data.meta_info, data.non_tensor_batch),
    )
