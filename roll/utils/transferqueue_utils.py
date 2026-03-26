"""
TransferQueue utilities for ROLL.

Provides the @tqbridge decorator and BatchMeta/KVBatchMeta conversion tools,
adapted from verl's transferqueue_utils.py for the ROLL framework.

When TQ is not enabled (no BatchMeta in function arguments), @tqbridge is a
zero-overhead pass-through.

All transfer_queue imports are lazy so that this module can be imported safely
even when the transfer_queue package is not installed (TQ disabled).
"""

import asyncio
import copy
import functools
import inspect
import logging
import os
import threading
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto

if TYPE_CHECKING:
    from roll.distributed.scheduler.decorator import Dispatch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ROLL_LOGGING_LEVEL", "WARN"))

TQ_INITIALIZED = False

# Lazy-loaded references (populated by _ensure_tq_imports)
_tq_mod = None
_BatchMeta_cls = None
_KVBatchMeta_cls = None


def _ensure_tq_imports():
    """Lazy-import transfer_queue to avoid hard dependency at module load."""
    global _tq_mod, _BatchMeta_cls, _KVBatchMeta_cls
    if _tq_mod is None:
        import transfer_queue
        from transfer_queue import BatchMeta, KVBatchMeta
        _tq_mod = transfer_queue
        _BatchMeta_cls = BatchMeta
        _KVBatchMeta_cls = KVBatchMeta
    return _tq_mod, _BatchMeta_cls, _KVBatchMeta_cls


def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    """Run an async function in a temporary event loop on a daemon thread."""
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="batchmeta tensordict converter",
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
    """Find BatchMeta in positional or keyword arguments."""
    _, BatchMeta, _ = _ensure_tq_imports()
    for arg in args:
        if isinstance(arg, BatchMeta):
            return arg
    for v in kwargs.values():
        if isinstance(v, BatchMeta):
            return v
    return None


async def _async_meta_to_realdata(meta) -> TensorDict:
    """Fetch actual tensor data from TQ storage given BatchMeta."""
    tq, _, _ = _ensure_tq_imports()
    if meta.size == 0:
        return TensorDict({}, batch_size=(0,))

    tq_client = tq.get_client()
    tensordict = await tq_client.async_get_data(meta)
    return tensordict


def _meta_to_realdata(meta) -> TensorDict:
    return _run_async_in_temp_loop(_async_meta_to_realdata, meta)


def _meta_to_dataproto(meta) -> DataProto:
    """Convert BatchMeta to DataProto by fetching TensorDict from TQ."""
    td = _meta_to_realdata(meta)
    extra = copy.deepcopy(meta.extra_info) if meta.extra_info else {}
    return DataProto(batch=td, meta_info=extra)


async def _async_meta_to_dataproto(meta) -> DataProto:
    td = await _async_meta_to_realdata(meta)
    extra = copy.deepcopy(meta.extra_info) if meta.extra_info else {}
    return DataProto(batch=td, meta_info=extra)


async def _async_update_meta_with_output(output: TensorDict, meta, func_name=None):
    """Write TensorDict output back to TQ storage and return updated BatchMeta."""
    tq, _, _ = _ensure_tq_imports()
    fields = []
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            fields.append(k)

    if fields:
        t1 = time.time()
        tq_client = tq.get_client()
        meta = await tq_client.async_put(data=output.select(*fields), metadata=meta)
        t2 = time.time()
        logger.info(f"Task {func_name} (pid={os.getpid()}) writing to TQ, cost: {t2 - t1:.3f}s")
    return meta


def _update_meta_with_output(output: TensorDict, meta, func_name=None):
    return _run_async_in_temp_loop(_async_update_meta_with_output, output, meta, func_name)


def _compute_need_collect(dispatch_mode, args: list) -> bool:
    """Determine whether the current worker should collect (write back) data.

    For ROLL's Dispatch enum modes, always returns True — the collect logic is
    handled by the Cluster's collect_fn. For custom dict dispatch_modes with a
    collect_fn partial, inspects worker rank to decide.
    """
    from roll.distributed.scheduler.decorator import Dispatch
    from roll.distributed.executor.worker import Worker

    if dispatch_mode is None or isinstance(dispatch_mode, Dispatch):
        return True

    assert isinstance(dispatch_mode, dict) and "collect_fn" in dispatch_mode

    collect_fn = dispatch_mode["collect_fn"]
    if isinstance(collect_fn, functools.partial):
        if len(args) < 1 or not isinstance(args[0], Worker):
            return True
        return args[0].query_collect_info(collect_fn)
    return True


def _extract_writable_tensordict(output):
    """Extract a non-empty TensorDict from output (DataProto or TensorDict)."""
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
    """Normalize return values when TQ bridge decides not to collect."""
    _, BatchMeta, _ = _ensure_tq_imports()
    if put_data and not need_collect:
        return BatchMeta()
    elif not put_data and not need_collect and isinstance(output, DataProto):
        return DataProto()
    elif not put_data and not need_collect and isinstance(output, TensorDict):
        return TensorDict({}, batch_size=(0,))
    return output


# ---------------------------------------------------------------------------
# KVBatchMeta <-> BatchMeta converters
# ---------------------------------------------------------------------------

async def async_kv_batch_meta2batch_meta(meta):
    tq, _, _ = _ensure_tq_imports()
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True
    tq_client = tq.get_client()
    batch_meta = await tq_client.async_kv_retrieve_meta(
        keys=meta.keys, partition_id=meta.partition_id, create=False
    )
    fields = meta.fields
    if fields is not None:
        if isinstance(fields, str):
            fields = [fields]
        batch_meta = batch_meta.select_fields(fields)
    batch_meta.extra_info = meta.extra_info
    return batch_meta


def kv_batch_meta2batch_meta(meta):
    result = _run_async_in_temp_loop(async_kv_batch_meta2batch_meta, meta)
    return result


async def async_batch_meta2kv_batch_meta(meta):
    tq, _, KVBatchMeta = _ensure_tq_imports()
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True
    tq_client = tq.get_client()
    partition_id = meta.partition_ids[0]
    assert all(partition_id == pid for pid in meta.partition_ids)
    keys = await tq_client.async_kv_retrieve_keys(
        global_indexes=meta.global_indexes, partition_id=partition_id
    )
    kv_batch_meta = KVBatchMeta(
        keys=keys,
        tags=[{}] * meta.size,
        partition_id=partition_id,
        fields=meta.field_names,
        extra_info=meta.extra_info,
    )
    return kv_batch_meta


def batch_meta2kv_batch_meta(meta):
    result = _run_async_in_temp_loop(async_batch_meta2kv_batch_meta, meta)
    return result


def kv_batch_meta_put_tensordict(meta, td: TensorDict, func_name: str = "kv_batch_meta_put_tensordict"):
    if td is None or not td.batch_size or td.batch_size[0] == 0:
        return meta
    batch_meta = kv_batch_meta2batch_meta(meta)
    updated_batch_meta = _update_meta_with_output(td, batch_meta, func_name)
    updated_kv_meta = batch_meta2kv_batch_meta(updated_batch_meta)
    meta.fields = list(updated_kv_meta.fields) if updated_kv_meta.fields is not None else []
    meta.extra_info = updated_kv_meta.extra_info
    return meta


# ---------------------------------------------------------------------------
# @kv_tqbridge decorator — for pipeline-local functions accepting KVBatchMeta
# ---------------------------------------------------------------------------

def kv_tqbridge(writeback_fields=None):
    """Decorator for pipeline-local functions to transparently accept KVBatchMeta.

    When the first ``data`` argument is a KVBatchMeta:
      1. Fetches tensor data from TQ into a DataProto
      2. Calls the original function with the DataProto
      3. Writes back *writeback_fields* to TQ
      4. Syncs DataProto.meta_info → KVBatchMeta.extra_info
      5. Returns KVBatchMeta in place of DataProto in the return value

    When the first ``data`` argument is a DataProto (TQ disabled), acts as a
    zero-overhead pass-through — the function is called unchanged.

    Args:
        writeback_fields: List of tensor field names to write back to TQ after
            the function completes.  Only fields that actually exist in the
            output DataProto's batch will be written.
    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            # --- Detect KVBatchMeta in arguments ---
            tq_mod, _, KVBatchMeta_cls = _ensure_tq_imports()

            kv_meta = None
            kv_arg_pos = None  # int → positional, str → kwarg key

            for idx, arg in enumerate(args):
                if isinstance(arg, KVBatchMeta_cls):
                    kv_meta = arg
                    kv_arg_pos = idx
                    break

            if kv_meta is None:
                for key in ("data",):
                    if key in kwargs and isinstance(kwargs[key], KVBatchMeta_cls):
                        kv_meta = kwargs[key]
                        kv_arg_pos = key
                        break

            if kv_meta is None:
                # No KVBatchMeta found → pure pass-through
                return func(*args, **kwargs)

            # --- KVBatchMeta path ---
            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq_mod.init()
                TQ_INITIALIZED = True

            from roll.utils.tq_pipeline_utils import (
                kv_batch_meta_to_dataproto as _kv2dp,
                kv_batch_meta_put_fields as _kv_put,
            )

            t0 = time.time()
            data = _kv2dp(kv_meta)
            data.meta_info = dict(kv_meta.extra_info) if kv_meta.extra_info else {}
            t1 = time.time()

            logger.info(
                f"[TQ] @kv_tqbridge [{func.__name__}] fetched "
                f"{len(kv_meta.keys)} samples from TQ, cost={t1 - t0:.3f}s"
            )

            # Replace KVBatchMeta with DataProto in the call
            if isinstance(kv_arg_pos, int):
                args = list(args)
                args[kv_arg_pos] = data
                args = tuple(args)
            else:
                kwargs[kv_arg_pos] = data

            # --- Call original function ---
            result = func(*args, **kwargs)

            # --- Unpack return value ---
            if isinstance(result, tuple):
                out_data = result[0]
                rest = result[1:]
            else:
                out_data = result
                rest = None

            # --- Write back fields to TQ ---
            if (
                writeback_fields
                and hasattr(out_data, "batch")
                and out_data.batch is not None
            ):
                existing = [
                    f for f in writeback_fields if f in out_data.batch.keys()
                ]
                if existing:
                    _kv_put(kv_meta, out_data, field_keys=existing)
                    for f in existing:
                        if kv_meta.fields is not None and f not in kv_meta.fields:
                            kv_meta.fields.append(f)
                    logger.info(
                        f"[TQ] @kv_tqbridge [{func.__name__}] wrote back "
                        f"fields={existing} to TQ"
                    )

            # --- Sync meta_info back to extra_info ---
            if hasattr(out_data, "meta_info") and out_data.meta_info:
                if kv_meta.extra_info is None:
                    kv_meta.extra_info = {}
                kv_meta.extra_info.update(out_data.meta_info)

            if rest is not None:
                return (kv_meta, *rest)
            return kv_meta

        return inner

    return decorator


# ---------------------------------------------------------------------------
# @tqbridge decorator — for distributed worker methods accepting BatchMeta
# ---------------------------------------------------------------------------

def tqbridge(dispatch_mode=None):
    """Decorator that bridges BatchMeta and DataProto/TensorDict for ROLL workers.

    When a decorated function receives a BatchMeta argument:
      - Input:  BatchMeta -> fetch data from TQ -> wrap as DataProto
      - Execute: call original function with DataProto
      - Output: if function returns DataProto with non-empty batch,
                write batch back to TQ, return BatchMeta

    When no BatchMeta is present (TQ disabled), the decorator is a no-op.
    Supports both sync and async functions.
    """
    from roll.distributed.scheduler.decorator import _check_dispatch_mode
    _check_dispatch_mode(dispatch_mode)

    def decorator(func):
        pid = os.getpid()

        @wraps(func)
        def inner(*args, **kwargs):
            tq, BatchMeta, _ = _ensure_tq_imports()
            batch_meta = _find_meta(*args, **kwargs)
            if batch_meta is None:
                return func(*args, **kwargs)

            print(f"[TQ] @tqbridge [{func.__name__}] input: {type(batch_meta).__name__}, "
                  f"size={batch_meta.size}, fields={batch_meta.field_names}")

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            t1 = time.time()
            args = [_meta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
            kwargs = {k: _meta_to_dataproto(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()}
            t2 = time.time()
            print(f"[TQ] @tqbridge [{func.__name__}] fetched {batch_meta.size} samples from TQ, cost={t2 - t1:.3f}s")

            output = func(*args, **kwargs)

            writable_td = _extract_writable_tensordict(output)
            put_data = writable_td is not None
            if put_data:
                assert writable_td.batch_size[0] == batch_meta.size, (
                    f"output batch_size {writable_td.batch_size} != meta size {batch_meta.size}"
                )

            need_collect = _compute_need_collect(dispatch_mode, args)
            if put_data and need_collect:
                result = _update_meta_with_output(writable_td, batch_meta, func.__name__)
                print(f"[TQ] @tqbridge [{func.__name__}] output: {type(result).__name__}, "
                      f"wrote back fields={list(writable_td.keys())}, "
                      f"shapes={{k: list(writable_td[k].shape) for k in list(writable_td.keys())[:5]}}")
                return result
            result = _postprocess_common(output, put_data, need_collect)
            print(f"[TQ] @tqbridge [{func.__name__}] output: {type(result).__name__}, put_data={put_data}, need_collect={need_collect}")
            return result

        @wraps(func)
        async def async_inner(*args, **kwargs):
            tq, BatchMeta, _ = _ensure_tq_imports()
            batch_meta = _find_meta(*args, **kwargs)
            if batch_meta is None:
                return await func(*args, **kwargs)

            print(f"[TQ] @tqbridge [{func.__name__}] async input: {type(batch_meta).__name__}, "
                  f"size={batch_meta.size}, fields={batch_meta.field_names}")

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            t1 = time.time()
            args = [await _async_meta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
            kwargs = {
                k: await _async_meta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                for k, v in kwargs.items()
            }
            t2 = time.time()
            print(f"[TQ] @tqbridge [{func.__name__}] async fetched {batch_meta.size} samples from TQ, cost={t2 - t1:.3f}s")

            output = await func(*args, **kwargs)

            writable_td = _extract_writable_tensordict(output)
            put_data = writable_td is not None
            if put_data:
                assert writable_td.batch_size[0] == batch_meta.size, (
                    f"output batch_size {writable_td.batch_size} != meta size {batch_meta.size}"
                )

            need_collect = _compute_need_collect(dispatch_mode, args)
            if put_data and need_collect:
                result = await _async_update_meta_with_output(writable_td, batch_meta, func.__name__)
                print(f"[TQ] @tqbridge [{func.__name__}] async output: {type(result).__name__}, "
                      f"wrote back fields={list(writable_td.keys())}, "
                      f"shapes={{k: list(writable_td[k].shape) for k in list(writable_td.keys())[:5]}}")
                return result
            result = _postprocess_common(output, put_data, need_collect)
            print(f"[TQ] @tqbridge [{func.__name__}] async output: {type(result).__name__}, put_data={put_data}, need_collect={need_collect}")
            return result

        return async_inner if inspect.iscoroutinefunction(func) else inner

    return decorator
