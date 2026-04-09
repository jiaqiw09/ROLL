"""TransferQueue helpers for DataProto <-> BatchMeta transport."""

import asyncio
import copy
import dataclasses
import functools
import threading
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.transferqueue_trace import (
    NO_TQ_COLLECT_TRACE_KEY,
    NO_TQ_DISPATCH_TRACE_KEY,
    TQ_COLLECT_TRACE_KEY,
    TQ_DISPATCH_TRACE_KEY,
    _attach_trace,
    _calc_dataproto_size_mb,
    _calc_tensordict_size_mb,
    configure_tq_trace,
    _extract_first_trace_from_dataprotos,
    _log_no_tq_collect_e2e,
    _log_no_tq_dispatch_e2e,
    _log_tq_conversion,
    _log_tq_dispatch_e2e,
    _make_collect_trace,
    _pop_trace,
)

if TYPE_CHECKING:
    from roll.distributed.scheduler.decorator import Dispatch

TQ_INITIALIZED = False

_tq_mod = None
_BatchMeta_cls = None


def _ensure_tq_imports():
    global _tq_mod, _BatchMeta_cls
    if _tq_mod is None:
        import transfer_queue
        from transfer_queue import BatchMeta

        _tq_mod = transfer_queue
        _BatchMeta_cls = BatchMeta
    return _tq_mod, _BatchMeta_cls


def is_batch_meta(value) -> bool:
    try:
        _, BatchMeta = _ensure_tq_imports()
    except ImportError:
        return False
    return isinstance(value, BatchMeta)


def init_tq(config=None):
    if dataclasses.is_dataclass(config):
        config = dataclasses.asdict(config)
    configure_tq_trace(config)
    tq, _ = _ensure_tq_imports()
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init(config)
        TQ_INITIALIZED = True


def is_tq_runtime_enabled() -> bool:
    return TQ_INITIALIZED


def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="batchmeta-data-converter",
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
    for arg in args:
        if is_batch_meta(arg):
            return arg
    for value in kwargs.values():
        if is_batch_meta(value):
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


def _peek_trace_from_meta(meta, trace_key: str):
    meta_info, _ = _split_extra_info(getattr(meta, "extra_info", None))
    trace = meta_info.get(trace_key)
    return copy.deepcopy(trace) if isinstance(trace, dict) else None


def _normalize_batch_meta_size(meta):
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


def _set_batchmeta_row_count(meta, rows: int) -> None:
    for attr_name in ("size", "_size"):
        try:
            setattr(meta, attr_name, rows)
        except Exception:
            pass


def _is_empty_batch_meta(meta) -> bool:
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
    meta_info, _ = _split_extra_info(getattr(meta, "extra_info", None))
    meta_info = copy.deepcopy(meta_info)
    meta_info["_broadcast_non_tensor_batch"] = True
    return meta_info


def _batchmeta_to_aux_dataproto(meta) -> DataProto:
    meta_info, non_tensor_batch = _split_extra_info(getattr(meta, "extra_info", None))
    return DataProto(batch=None, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


async def _async_meta_to_realdata(meta) -> TensorDict:
    tq, _ = _ensure_tq_imports()
    meta = _normalize_batch_meta_size(meta)
    if getattr(meta, "size", 0) == 0:
        return TensorDict({}, batch_size=(0,))
    tq_client = tq.get_client()
    return await tq_client.async_get_data(meta)


def _meta_to_realdata(meta) -> TensorDict:
    meta = _normalize_batch_meta_size(meta)
    return _run_async_in_temp_loop(_async_meta_to_realdata, meta)


def _meta_to_dataproto(
    meta,
    *,
    scene: Optional[str] = None,
    trace_key: Optional[str] = None,
    worker_name: Optional[str] = None,
    domain: Optional[str] = None,
    prompt_id: Optional[int] = None,
    meta_count: Optional[int] = None,
) -> DataProto:
    if _is_empty_batch_meta(meta):
        return DataProto(batch=None, non_tensor_batch={}, meta_info=_placeholder_meta_info(meta))
    trace = _peek_trace_from_meta(meta, trace_key) if trace_key else None
    t0 = time.time()
    td = _meta_to_realdata(meta)
    cost = time.time() - t0
    meta_info, non_tensor_batch = _split_extra_info(getattr(meta, "extra_info", None))
    if trace_key and trace is not None:
        updated_trace = copy.deepcopy(trace)
        updated_trace["worker_read_cost"] = cost
        meta_info = _attach_trace(meta_info, trace_key, updated_trace)
    data = DataProto(batch=td, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    if scene is not None:
        _log_tq_conversion(
            direction="batchmeta_to_dataproto",
            scene=scene,
            rows=len(data),
            data_mb=_calc_dataproto_size_mb(data),
            cost=cost,
            trace_id=trace.get("trace_id") if trace else None,
            worker_name=worker_name,
            domain=domain,
            prompt_id=prompt_id,
            meta_count=meta_count,
            partition_id=trace.get("partition_id") if trace else None,
        )
    return data


async def _async_meta_to_dataproto(
    meta,
    *,
    scene: Optional[str] = None,
    trace_key: Optional[str] = None,
    worker_name: Optional[str] = None,
    domain: Optional[str] = None,
    prompt_id: Optional[int] = None,
    meta_count: Optional[int] = None,
) -> DataProto:
    if _is_empty_batch_meta(meta):
        return DataProto(batch=None, non_tensor_batch={}, meta_info=_placeholder_meta_info(meta))
    trace = _peek_trace_from_meta(meta, trace_key) if trace_key else None
    t0 = time.time()
    td = await _async_meta_to_realdata(meta)
    cost = time.time() - t0
    meta_info, non_tensor_batch = _split_extra_info(getattr(meta, "extra_info", None))
    if trace_key and trace is not None:
        updated_trace = copy.deepcopy(trace)
        updated_trace["worker_read_cost"] = cost
        meta_info = _attach_trace(meta_info, trace_key, updated_trace)
    data = DataProto(batch=td, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
    if scene is not None:
        _log_tq_conversion(
            direction="batchmeta_to_dataproto",
            scene=scene,
            rows=len(data),
            data_mb=_calc_dataproto_size_mb(data),
            cost=cost,
            trace_id=trace.get("trace_id") if trace else None,
            worker_name=worker_name,
            domain=domain,
            prompt_id=prompt_id,
            meta_count=meta_count,
            partition_id=trace.get("partition_id") if trace else None,
        )
    return data


def meta_to_dataproto(meta, **kwargs) -> DataProto:
    return _meta_to_dataproto(meta, **kwargs)


def merge_batch_metas(metas: List[Any], *, global_keys: Optional[set[str]] = None):
    if not metas:
        raise ValueError("merge_batch_metas requires at least one BatchMeta")

    normalized_metas = [_normalize_batch_meta_size(meta) for meta in metas]
    merged_meta = copy.deepcopy(normalized_metas[0])
    merged_global_indexes = []
    merged_partition_ids = []
    aux_data = []

    for meta in normalized_metas:
        merged_global_indexes.extend(list(getattr(meta, "global_indexes", [])))
        merged_partition_ids.extend(list(getattr(meta, "partition_ids", [])))
        aux_data.append(_batchmeta_to_aux_dataproto(meta))

    merged_aux = DataProto.concat(aux_data, global_keys=global_keys)
    merged_meta.global_indexes = merged_global_indexes
    merged_meta.partition_ids = merged_partition_ids
    _set_batchmeta_row_count(merged_meta, len(merged_global_indexes))
    merged_meta.extra_info = _pack_extra_info(merged_aux.meta_info, merged_aux.non_tensor_batch)
    return merged_meta


async def _async_update_meta_with_output(output: TensorDict, meta):
    tq, _ = _ensure_tq_imports()
    fields = [key for key, value in output.items() if isinstance(value, torch.Tensor)]
    if not fields:
        return meta
    tq_client = tq.get_client()
    return await tq_client.async_put(data=output.select(*fields), metadata=meta)


def _update_meta_with_output(output: TensorDict, meta):
    return _run_async_in_temp_loop(_async_update_meta_with_output, output, meta)


async def _async_put_dataproto_to_batchmeta(
    data: DataProto,
    *,
    partition_id: str,
    scene: str,
    trace_key: Optional[str] = None,
    trace_payload: Optional[dict] = None,
    worker_name: Optional[str] = None,
    domain: Optional[str] = None,
    prompt_id: Optional[int] = None,
):
    tq, _ = _ensure_tq_imports()
    td = data.batch
    if td is None or not td.batch_size or td.batch_size[0] == 0:
        return None

    tq_client = tq.get_client()
    t0 = time.time()
    meta = await tq_client.async_put(
        data=td,
        metadata=None,
        partition_id=partition_id,
    )
    cost = time.time() - t0
    rows = td.batch_size[0]
    data_mb = _calc_tensordict_size_mb(td)

    meta_info = copy.deepcopy(data.meta_info)
    if trace_key and trace_payload is not None:
        meta_info = _attach_trace(meta_info, trace_key, trace_payload)
    meta.extra_info = _pack_extra_info(meta_info, data.non_tensor_batch)

    _log_tq_conversion(
        direction="dataproto_to_batchmeta",
        scene=scene,
        rows=rows,
        data_mb=data_mb,
        cost=cost,
        trace_id=(trace_payload or {}).get("trace_id"),
        worker_name=worker_name,
        domain=domain,
        prompt_id=prompt_id,
        partition_id=partition_id,
    )
    return meta


def _put_dataproto_to_batchmeta(data: DataProto, **kwargs):
    return _run_async_in_temp_loop(_async_put_dataproto_to_batchmeta, data, **kwargs)


async def _async_update_batchmeta_from_dataproto(
    output: DataProto,
    meta,
    *,
    scene: str,
    worker_name: str,
    trace_key: Optional[str] = None,
):
    tq, _ = _ensure_tq_imports()
    writable_td = _extract_writable_tensordict(output)
    if writable_td is None:
        return meta

    fields = [key for key, value in writable_td.items() if isinstance(value, torch.Tensor)]
    if not fields:
        return meta

    rows = writable_td.batch_size[0]
    data_mb = _calc_tensordict_size_mb(writable_td)
    t0 = time.time()
    tq_client = tq.get_client()

    trace_payload = None
    output_meta_info = copy.deepcopy(output.meta_info)
    if trace_key:
        trace_payload = _make_collect_trace(
            rows=rows,
            data_mb=data_mb,
            worker_name=worker_name,
            partition_id=getattr(meta, "partition_ids", [None])[0] if getattr(meta, "partition_ids", None) else None,
            worker_write_started_at=t0,
        )
        output_meta_info = _attach_trace(output_meta_info, trace_key, trace_payload)

    io_meta = copy.deepcopy(meta)
    io_meta.extra_info = _pack_extra_info(output_meta_info, output.non_tensor_batch)
    updated_meta = await tq_client.async_put(data=writable_td.select(*fields), metadata=io_meta)
    cost = time.time() - t0

    if trace_payload is not None:
        trace_payload["worker_write_cost"] = cost
        output_meta_info = _attach_trace(output.meta_info, trace_key, trace_payload)
        updated_meta.extra_info = _pack_extra_info(output_meta_info, output.non_tensor_batch)

    _log_tq_conversion(
        direction="dataproto_to_batchmeta",
        scene=scene,
        rows=rows,
        data_mb=data_mb,
        cost=cost,
        trace_id=trace_payload.get("trace_id") if trace_payload else None,
        worker_name=worker_name,
        partition_id=(trace_payload or {}).get("partition_id"),
    )
    return updated_meta


def _update_batchmeta_from_dataproto(output: DataProto, meta, **kwargs):
    return _run_async_in_temp_loop(_async_update_batchmeta_from_dataproto, output, meta, **kwargs)


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
    """Bridge BatchMeta input/output with worker methods while preserving DataProto interfaces."""

    from roll.distributed.scheduler.decorator import _check_dispatch_mode

    _check_dispatch_mode(dispatch_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            tq, BatchMeta = _ensure_tq_imports()
            batch_meta = _find_batch_meta(*args, **kwargs)

            if batch_meta is None:
                no_tq_dispatch_trace = _extract_first_trace_from_dataprotos(NO_TQ_DISPATCH_TRACE_KEY, args, kwargs)
                if no_tq_dispatch_trace is not None:
                    for arg in args:
                        if isinstance(arg, DataProto) and arg.batch is not None:
                            _log_no_tq_dispatch_e2e(
                                func.__name__,
                                no_tq_dispatch_trace,
                                rows=arg.batch.batch_size[0],
                                data_mb=_calc_dataproto_size_mb(arg),
                            )
                            break
                output = func(*args, **kwargs)
                writable_td = _extract_writable_tensordict(output)
                if writable_td is not None and isinstance(output, DataProto):
                    output.meta_info = _attach_trace(output.meta_info, NO_TQ_COLLECT_TRACE_KEY, _make_collect_trace(
                        rows=writable_td.batch_size[0],
                        data_mb=_calc_tensordict_size_mb(writable_td),
                        worker_name=func.__name__,
                    ))
                return output

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            io_meta = batch_meta
            is_placeholder_input = _is_empty_batch_meta(io_meta)
            n_rows = getattr(io_meta, "size", len(getattr(io_meta, "global_indexes", [])))

            t_read = time.time()
            args = [
                _meta_to_dataproto(
                    arg,
                    scene="dispatch_worker_read",
                    trace_key=TQ_DISPATCH_TRACE_KEY,
                    worker_name=func.__name__,
                ) if isinstance(arg, BatchMeta) else arg
                for arg in args
            ]
            kwargs = {
                k: _meta_to_dataproto(
                    v,
                    scene="dispatch_worker_read",
                    trace_key=TQ_DISPATCH_TRACE_KEY,
                    worker_name=func.__name__,
                ) if isinstance(v, BatchMeta) else v
                for k, v in kwargs.items()
            }
            read_cost = time.time() - t_read
            dispatch_trace = _peek_trace_from_meta(io_meta, TQ_DISPATCH_TRACE_KEY)
            read_data_mb = sum(_calc_dataproto_size_mb(arg) for arg in args if isinstance(arg, DataProto))
            if dispatch_trace is not None and not is_placeholder_input:
                _log_tq_dispatch_e2e(
                    func.__name__,
                    dispatch_trace,
                    rows=n_rows,
                    data_mb=read_data_mb,
                    worker_read_cost=read_cost,
                )

            output = func(*args, **kwargs)
            writable_td = _extract_writable_tensordict(output)
            put_data = writable_td is not None
            need_collect = _compute_need_collect(dispatch_mode, args)

            if is_placeholder_input:
                return _postprocess_common(output, put_data, False)

            if put_data and need_collect:
                return _update_batchmeta_from_dataproto(
                    output,
                    io_meta,
                    scene="collect_worker_write",
                    worker_name=func.__name__,
                    trace_key=TQ_COLLECT_TRACE_KEY,
                )

            return _postprocess_common(output, put_data, need_collect)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            tq, BatchMeta = _ensure_tq_imports()
            batch_meta = _find_batch_meta(*args, **kwargs)

            if batch_meta is None:
                no_tq_dispatch_trace = _extract_first_trace_from_dataprotos(NO_TQ_DISPATCH_TRACE_KEY, args, kwargs)
                if no_tq_dispatch_trace is not None:
                    for arg in args:
                        if isinstance(arg, DataProto) and arg.batch is not None:
                            _log_no_tq_dispatch_e2e(
                                func.__name__,
                                no_tq_dispatch_trace,
                                rows=arg.batch.batch_size[0],
                                data_mb=_calc_dataproto_size_mb(arg),
                            )
                            break
                output = await func(*args, **kwargs)
                writable_td = _extract_writable_tensordict(output)
                if writable_td is not None and isinstance(output, DataProto):
                    output.meta_info = _attach_trace(output.meta_info, NO_TQ_COLLECT_TRACE_KEY, _make_collect_trace(
                        rows=writable_td.batch_size[0],
                        data_mb=_calc_tensordict_size_mb(writable_td),
                        worker_name=func.__name__,
                    ))
                return output

            global TQ_INITIALIZED
            if not TQ_INITIALIZED:
                tq.init()
                TQ_INITIALIZED = True

            io_meta = batch_meta
            is_placeholder_input = _is_empty_batch_meta(io_meta)
            n_rows = getattr(io_meta, "size", len(getattr(io_meta, "global_indexes", [])))

            t_read = time.time()
            args = [
                await _async_meta_to_dataproto(
                    arg,
                    scene="dispatch_worker_read",
                    trace_key=TQ_DISPATCH_TRACE_KEY,
                    worker_name=func.__name__,
                ) if isinstance(arg, BatchMeta) else arg
                for arg in args
            ]
            kwargs = {
                k: await _async_meta_to_dataproto(
                    v,
                    scene="dispatch_worker_read",
                    trace_key=TQ_DISPATCH_TRACE_KEY,
                    worker_name=func.__name__,
                ) if isinstance(v, BatchMeta) else v
                for k, v in kwargs.items()
            }
            read_cost = time.time() - t_read
            dispatch_trace = _peek_trace_from_meta(io_meta, TQ_DISPATCH_TRACE_KEY)
            read_data_mb = sum(_calc_dataproto_size_mb(arg) for arg in args if isinstance(arg, DataProto))
            if dispatch_trace is not None and not is_placeholder_input:
                _log_tq_dispatch_e2e(
                    func.__name__,
                    dispatch_trace,
                    rows=n_rows,
                    data_mb=read_data_mb,
                    worker_read_cost=read_cost,
                )

            output = await func(*args, **kwargs)
            writable_td = _extract_writable_tensordict(output)
            put_data = writable_td is not None
            need_collect = _compute_need_collect(dispatch_mode, args)

            if is_placeholder_input:
                return _postprocess_common(output, put_data, False)

            if put_data and need_collect:
                return await _async_update_batchmeta_from_dataproto(
                    output,
                    io_meta,
                    scene="collect_worker_write",
                    worker_name=func.__name__,
                    trace_key=TQ_COLLECT_TRACE_KEY,
                )

            return _postprocess_common(output, put_data, need_collect)

        return async_inner if asyncio.iscoroutinefunction(func) else inner

    return decorator
