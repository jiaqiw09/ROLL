"""TransferQueue utilities for ROLL.

Provides the @tqbridge decorator and BatchMeta conversion tools, adapted from
the TQ prototype integration.

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
from typing import TYPE_CHECKING, Any, Callable, List

import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto

if TYPE_CHECKING:
    from roll.distributed.scheduler.decorator import Dispatch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("ROLL_TQ_LOGGING_LEVEL", "INFO"))

TQ_INITIALIZED = False

_tq_mod = None
_BatchMeta_cls = None

TQ_DISPATCH_TRACE_KEY = "_tq_dispatch_trace"
TQ_COLLECT_TRACE_KEY = "_tq_collect_trace"


def _calc_tensordict_size_mb(td) -> float:
    total = 0
    if td is None:
        return 0.0
    for value in td.values():
        if hasattr(value, "nbytes"):
            total += value.nbytes
    return total / (1024 * 1024)


def _calc_dataproto_size_mb(data) -> float:
    if not isinstance(data, DataProto):
        return 0.0
    return _calc_tensordict_size_mb(data.batch)


def _get_process_rss_gb() -> float:
    from roll.utils.context_managers import cpu_memory_info

    return cpu_memory_info().rss / 1024**3


def _clone_meta_info(meta_info):
    return copy.deepcopy(meta_info) if meta_info else {}


def _attach_trace(meta_info, trace_key: str, trace_payload: dict):
    meta = _clone_meta_info(meta_info)
    meta[trace_key] = copy.deepcopy(trace_payload)
    return meta


def _pop_trace(meta_info, trace_key: str):
    if not meta_info:
        return None
    trace = meta_info.pop(trace_key, None)
    return copy.deepcopy(trace) if isinstance(trace, dict) else None


def _extract_first_trace_from_dataprotos(trace_key: str, *containers):
    for container in containers:
        if isinstance(container, dict):
            iterable = container.values()
        else:
            iterable = container
        for item in iterable:
            if isinstance(item, DataProto):
                trace = _pop_trace(item.meta_info, trace_key)
                if trace is not None:
                    return trace
    return None


def _ensure_tq_imports():
    global _tq_mod, _BatchMeta_cls
    if _tq_mod is None:
        import transfer_queue
        from transfer_queue import BatchMeta

        _tq_mod = transfer_queue
        _BatchMeta_cls = BatchMeta
    return _tq_mod, _BatchMeta_cls


def init_tq(config=None):
    """Initialize TransferQueue once in the current process."""
    tq, _ = _ensure_tq_imports()
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init(config)
        TQ_INITIALIZED = True


def is_tq_runtime_enabled() -> bool:
    """Return whether TransferQueue has been enabled in the current process.

    This is the runtime single source of truth used by dispatch/collect paths.
    Once `init_tq(...)` succeeds in a process, all supported call sites in that
    process should consistently consider TQ enabled, without depending on which
    wrapper object they were called through.
    """
    return TQ_INITIALIZED


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


def _find_batch_meta(*args, **kwargs):
    _, BatchMeta = _ensure_tq_imports()
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


def _normalize_batch_meta_size(meta):
    """Best-effort fix for BatchMeta copies whose reported size drifts from their indexes."""
    rows = None
    global_indexes = getattr(meta, "global_indexes", None)
    if global_indexes is not None:
        try:
            rows = len(global_indexes)
        except Exception:
            rows = None
    if rows is None:
        return meta

    for attr_name in ("size", "_size"):
        try:
            current = getattr(meta, attr_name)
        except Exception:
            current = None
        if current != rows:
            try:
                setattr(meta, attr_name, rows)
            except Exception:
                pass
    return meta


def _is_empty_batch_meta(meta) -> bool:
    """Return True when BatchMeta is only a dispatch placeholder."""
    if meta is None:
        return False
    meta = _normalize_batch_meta_size(meta)
    size = getattr(meta, "size", None)
    if size == 0:
        return True
    global_indexes = getattr(meta, "global_indexes", None)
    try:
        return global_indexes is not None and len(global_indexes) == 0
    except Exception:
        return False


def _placeholder_meta_info(meta):
    """Build minimal meta_info for an empty dispatch placeholder."""
    meta_info, non_tensor_batch = _split_extra_info(getattr(meta, "extra_info", None))
    meta_info = copy.deepcopy(meta_info)
    meta_info["_broadcast_non_tensor_batch"] = True
    return meta_info


async def _async_meta_to_realdata(meta) -> TensorDict:
    tq, _ = _ensure_tq_imports()
    meta = _normalize_batch_meta_size(meta)
    if meta.size == 0:
        return TensorDict({}, batch_size=(0,))

    tq_client = tq.get_client()
    return await tq_client.async_get_data(meta)


def _meta_to_realdata(meta) -> TensorDict:
    meta = _normalize_batch_meta_size(meta)
    return _run_async_in_temp_loop(_async_meta_to_realdata, meta)


def _meta_to_dataproto(meta) -> DataProto:
    if _is_empty_batch_meta(meta):
        meta_info = _placeholder_meta_info(meta)
        return DataProto(batch=None, non_tensor_batch={}, meta_info=meta_info)
    td = _meta_to_realdata(meta)
    meta_info, non_tensor_batch = _split_extra_info(getattr(meta, "extra_info", None))
    return DataProto(batch=td, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


async def _async_meta_to_dataproto(meta) -> DataProto:
    if _is_empty_batch_meta(meta):
        meta_info = _placeholder_meta_info(meta)
        return DataProto(batch=None, non_tensor_batch={}, meta_info=meta_info)
    td = await _async_meta_to_realdata(meta)
    meta_info, non_tensor_batch = _split_extra_info(getattr(meta, "extra_info", None))
    return DataProto(batch=td, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def meta_to_dataproto(meta) -> DataProto:
    return _meta_to_dataproto(meta)


async def _async_update_meta_with_output(output: TensorDict, meta, func_name=None):
    tq, _ = _ensure_tq_imports()
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


def _empty_batch_meta():
    _, BatchMeta = _ensure_tq_imports()
    meta = BatchMeta(global_indexes=[], partition_ids=[])
    return _normalize_batch_meta_size(meta)


def _postprocess_common(output, put_data, need_collect):
    if put_data and not need_collect:
        return _empty_batch_meta()
    if not put_data and not need_collect and isinstance(output, DataProto):
        return DataProto()
    if not put_data and not need_collect and isinstance(output, TensorDict):
        return TensorDict({}, batch_size=(0,))
    return output


def tqbridge(dispatch_mode=None):
    """Bridge BatchMeta input/output with DataProto/TensorDict worker methods."""
    from roll.distributed.scheduler.decorator import _check_dispatch_mode

    _check_dispatch_mode(dispatch_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            import time
            tq, BatchMeta, _ = _ensure_tq_imports()
            batch_meta = _find_batch_meta(*args, **kwargs)
            if batch_meta is None:
                for arg in args:
                    if isinstance(arg, DataProto) and arg.batch is not None:
                        rows = arg.batch.batch_size[0]
                        data_mb = _calc_dataproto_size_mb(arg)
                        logger.info(
                            f"[NO TQ worker] {func.__name__} | INPUT VIA RAY | rows={rows} "
                            f"| data={data_mb:.2f}MB | rss={_get_process_rss_gb():.3f}GB"
                        )
                output = func(*args, **kwargs)
                writable_td = _extract_writable_tensordict(output)
                if writable_td is not None:
                    write_data_mb = _calc_tensordict_size_mb(writable_td)
                    logger.info(
                        f"[NO TQ worker] {func.__name__} | OUTPUT VIA RAY | rows={writable_td.batch_size[0]} "
                        f"| data={write_data_mb:.2f}MB | rss={_get_process_rss_gb():.3f}GB"
                    )
                return output

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            io_meta = batch_meta
            is_placeholder_input = _is_empty_batch_meta(io_meta)
            n_rows = getattr(io_meta, 'size', len(getattr(io_meta, 'global_indexes', [])))

            # Worker side: TQ READ
            logger.debug(
                f"[TQ worker] {func.__name__} | READ TRIGGERED | rows={n_rows} "
                f"(BatchMeta received instead of DataProto via Ray)"
            )
            t_read = time.time()
            args = [_meta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
            kwargs = {
                k: _meta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                for k, v in kwargs.items()
            }
            read_cost = time.time() - t_read
            # Compute size of data read
            read_data_mb = 0.0
            for a in args:
                if isinstance(a, DataProto) and a.batch is not None:
                    for v in a.batch.values():
                        if hasattr(v, 'nbytes'):
                            read_data_mb += v.nbytes
            read_data_mb /= (1024 * 1024)
            dispatch_trace = _extract_first_trace_from_dataprotos(
                TQ_DISPATCH_TRACE_KEY, args, kwargs
            )
            logger.debug(
                f"[TQ worker] {func.__name__} | READ DONE | rows={n_rows} "
                f"| data={read_data_mb:.2f}MB | TQ read cost={read_cost:.3f}s"
            )
            if dispatch_trace is not None and not is_placeholder_input:
                e2e_total = max(0.0, time.time() - dispatch_trace["dispatch_started_at"])
                logger.info(
                    f"[TQ dispatch E2E] {func.__name__} | trace_id={dispatch_trace['trace_id']} "
                    f"| rows={n_rows} | data={read_data_mb:.2f}MB | total={e2e_total:.3f}s "
                    f"| driver_put={dispatch_trace['driver_put_cost']:.3f}s "
                    f"| worker_read={read_cost:.3f}s"
                )

            output = func(*args, **kwargs)

            writable_td = _extract_writable_tensordict(output)
            put_data = writable_td is not None
            need_collect = _compute_need_collect(dispatch_mode, args)
            if is_placeholder_input:
                logger.info(
                    f"[TQ worker] {func.__name__} | SKIP WRITEBACK | rows={n_rows} "
                    f"(dispatch_first placeholder rank; output is not collected)"
                )
                return _postprocess_common(output, put_data, False)
            if put_data:
                assert writable_td.batch_size[0] == io_meta.size, (
                    f"output batch_size {writable_td.batch_size} != meta size {io_meta.size}"
                )

            if put_data and need_collect:
                # Worker side: TQ WRITE
                write_data_mb = sum(
                    v.nbytes for v in writable_td.values() if hasattr(v, 'nbytes')
                ) / (1024 * 1024)
                logger.debug(
                    f"[TQ worker] {func.__name__} | WRITE TRIGGERED | rows={writable_td.batch_size[0]} "
                    f"| data={write_data_mb:.2f}MB (writing output back to TQ instead of Ray)"
                )
                t_write = time.time()
                output.meta_info = _attach_trace(
                    output.meta_info,
                    TQ_COLLECT_TRACE_KEY,
                    {
                        "trace_id": uuid.uuid4().hex[:12],
                        "worker_write_started_at": t_write,
                        "worker_name": func.__name__,
                        "rows": writable_td.batch_size[0],
                        "data_mb": write_data_mb,
                    },
                )
                io_meta = _apply_output_extra_info(io_meta, output)
                result = _update_meta_with_output(writable_td, io_meta, func.__name__)
                write_cost = time.time() - t_write
                logger.debug(
                    f"[TQ worker] {func.__name__} | WRITE DONE | data={write_data_mb:.2f}MB "
                    f"| TQ write cost={write_cost:.3f}s"
                )
                return result
            return _postprocess_common(output, put_data, need_collect)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            import time
            tq, BatchMeta, _ = _ensure_tq_imports()
            batch_meta = _find_batch_meta(*args, **kwargs)
            if batch_meta is None:
                for arg in args:
                    if isinstance(arg, DataProto) and arg.batch is not None:
                        rows = arg.batch.batch_size[0]
                        data_mb = _calc_dataproto_size_mb(arg)
                        logger.info(
                            f"[NO TQ worker] {func.__name__} | INPUT VIA RAY | rows={rows} "
                            f"| data={data_mb:.2f}MB | rss={_get_process_rss_gb():.3f}GB"
                        )
                output = await func(*args, **kwargs)
                writable_td = _extract_writable_tensordict(output)
                if writable_td is not None:
                    write_data_mb = _calc_tensordict_size_mb(writable_td)
                    logger.info(
                        f"[NO TQ worker] {func.__name__} | OUTPUT VIA RAY | rows={writable_td.batch_size[0]} "
                        f"| data={write_data_mb:.2f}MB | rss={_get_process_rss_gb():.3f}GB"
                    )
                return output

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            io_meta = batch_meta
            is_placeholder_input = _is_empty_batch_meta(io_meta)
            n_rows = getattr(io_meta, 'size', len(getattr(io_meta, 'global_indexes', [])))

            # Worker side: TQ READ (async)
            logger.debug(
                f"[TQ worker] {func.__name__} | READ TRIGGERED | rows={n_rows} "
                f"(BatchMeta received instead of DataProto via Ray)"
            )
            t_read = time.time()
            args = [
                await _async_meta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg
                for arg in args
            ]
            kwargs = {
                k: await _async_meta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                for k, v in kwargs.items()
            }
            read_cost = time.time() - t_read
            read_data_mb = 0.0
            for a in args:
                if isinstance(a, DataProto) and a.batch is not None:
                    for v in a.batch.values():
                        if hasattr(v, 'nbytes'):
                            read_data_mb += v.nbytes
            read_data_mb /= (1024 * 1024)
            dispatch_trace = _extract_first_trace_from_dataprotos(
                TQ_DISPATCH_TRACE_KEY, args, kwargs
            )
            logger.debug(
                f"[TQ worker] {func.__name__} | READ DONE | rows={n_rows} "
                f"| data={read_data_mb:.2f}MB | TQ read cost={read_cost:.3f}s"
            )
            if dispatch_trace is not None and not is_placeholder_input:
                e2e_total = max(0.0, time.time() - dispatch_trace["dispatch_started_at"])
                logger.info(
                    f"[TQ dispatch E2E] {func.__name__} | trace_id={dispatch_trace['trace_id']} "
                    f"| rows={n_rows} | data={read_data_mb:.2f}MB | total={e2e_total:.3f}s "
                    f"| driver_put={dispatch_trace['driver_put_cost']:.3f}s "
                    f"| worker_read={read_cost:.3f}s"
                )

            output = await func(*args, **kwargs)

            writable_td = _extract_writable_tensordict(output)
            put_data = writable_td is not None
            need_collect = _compute_need_collect(dispatch_mode, args)
            if is_placeholder_input:
                logger.info(
                    f"[TQ worker] {func.__name__} | SKIP WRITEBACK | rows={n_rows} "
                    f"(dispatch_first placeholder rank; output is not collected)"
                )
                return _postprocess_common(output, put_data, False)
            if put_data:
                assert writable_td.batch_size[0] == io_meta.size, (
                    f"output batch_size {writable_td.batch_size} != meta size {io_meta.size}"
                )

            if put_data and need_collect:
                write_data_mb = sum(
                    v.nbytes for v in writable_td.values() if hasattr(v, 'nbytes')
                ) / (1024 * 1024)
                logger.debug(
                    f"[TQ worker] {func.__name__} | WRITE TRIGGERED | rows={writable_td.batch_size[0]} "
                    f"| data={write_data_mb:.2f}MB (writing output back to TQ instead of Ray)"
                )
                t_write = time.time()
                output.meta_info = _attach_trace(
                    output.meta_info,
                    TQ_COLLECT_TRACE_KEY,
                    {
                        "trace_id": uuid.uuid4().hex[:12],
                        "worker_write_started_at": t_write,
                        "worker_name": func.__name__,
                        "rows": writable_td.batch_size[0],
                        "data_mb": write_data_mb,
                    },
                )
                io_meta = _apply_output_extra_info(io_meta, output)
                result = await _async_update_meta_with_output(writable_td, io_meta, func.__name__)
                write_cost = time.time() - t_write
                logger.debug(
                    f"[TQ worker] {func.__name__} | WRITE DONE | data={write_data_mb:.2f}MB "
                    f"| TQ write cost={write_cost:.3f}s"
                )
                return result
            return _postprocess_common(output, put_data, need_collect)

        return async_inner if inspect.iscoroutinefunction(func) else inner

    return decorator
