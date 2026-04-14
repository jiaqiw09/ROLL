"""
ref: https://github.com/volcengine/verl/blob/main/single_controller/base/decorator.py
"""

import gc
import os
import traceback
from enum import Enum, auto
from functools import wraps, partial
from itertools import chain
from typing import Tuple, List, Dict
from more_itertools import chunked
import ray
import torch
import asyncio
import copy
import time
import uuid

from roll.distributed.scheduler.protocol import DataProto, ObjectRefWrap
from roll.utils.transferqueue_trace import (
    NO_TQ_COLLECT_TRACE_KEY,
    NO_TQ_DISPATCH_TRACE_KEY,
    TQ_COLLECT_TRACE_KEY,
    TQ_DISPATCH_TRACE_KEY,
    _attach_trace,
    _calc_dataproto_size_mb,
    _calc_tensordict_size_mb,
    _log_no_tq_collect_e2e,
    _log_tq_collect_e2e,
    _make_dispatch_trace,
)
from roll.utils.logging import get_logger
from roll.platforms import current_platform

logger = get_logger()

BIND_WORKER_METHOD_FLAG = "BIND_WORKER_METHOD_FLAG"
TQ_DISPATCH_PARTITION_ID = "train_dispatch"


class Dispatch(Enum):
    """
    dispatch 负责处理Cluster的输入list如何拆分分配到各worker上
    """

    ONE_TO_ALL = auto()
    ONE_TO_ALL_ONE = auto()
    ALL_TO_ALL = auto()
    DP_MP_COMPUTE = auto()
    DP_MP_DISPATCH_FIRST = auto()
    DP_MP_DISPATCH_FIRST_COLLECT_ALL = auto()


class Execute(Enum):
    ALL = 0
    RANK_ZERO = 1


def _split_args_kwargs(chunks, *args, **kwargs):
    """
    arg: List, 将List分成dp份
    """

    def split(arg, chunks):
        if isinstance(arg, list):
            return list(chunked(arg, len(arg) // chunks))
        else:
            assert hasattr(arg, "chunk"), f"Argument {arg} does not have a 'chunk' method."
            return arg.chunk(chunks=chunks)

    splitted_args = []
    for arg in args:
        splitted_args.append(split(arg, chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        splitted_kwargs[key] = split(val, chunks)

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(cluster, *args, **kwargs):
    """
    假定输入arg是一个值，分发到所有的worker上
    """
    args = tuple([arg] * cluster.world_size for arg in args)
    kwargs = {k: [v] * cluster.world_size for k, v in kwargs.items()}
    return args, kwargs


def collect_all_to_all(cluster, output):
    """
    collect 所有worker的输出
    """
    assert len(output) == cluster.world_size
    return output


def collect_all_to_one(cluster, output):
    """
    collect 所有worker的输出
    """
    assert len(output) == cluster.world_size

    if isinstance(output[0], ray.ObjectRef):
        output_in_dp = []
        for global_rank in range(cluster.world_size):
            output_in_dp.append(ObjectRefWrap(output[global_rank], collected=global_rank == 0))
        return output_in_dp
    return output[0]


def dispatch_all_to_all(cluster, *args, **kwargs):
    """
    假定输入arg是List, len(arg) = cluster.world_size
    """
    for arg in args:
        assert isinstance(arg, (Tuple, List)) and len(arg) == cluster.world_size
    for k, v in kwargs.items():
        assert isinstance(v, (Tuple, List)) and len(v) == cluster.world_size
    return args, kwargs


def _is_tq_enabled(cluster) -> bool:
    from roll.utils.transferqueue_utils import is_tq_runtime_enabled

    return is_tq_runtime_enabled()


def _is_representative_rank(rank_info) -> bool:
    return (
        getattr(rank_info, "tp_rank", 0) == 0
        and getattr(rank_info, "is_pipeline_last_stage", True)
        and getattr(rank_info, "cp_rank", 0) == 0
    )


def _collect_representative_results(cluster, raw_output):
    rep_results = []
    for global_rank in range(cluster.world_size):
        local_rank_info = cluster.get_rank_info(rank=global_rank)
        if _is_representative_rank(local_rank_info):
            rep_results.append(raw_output[global_rank])
    return rep_results


def _dispatch_dp_mp_compute(cluster, _dispatch_first, *args, **kwargs):
    """
    将输入chunk成dp_world_size份，按dp_rank为每个worker组织数据 -> 同一dp_rank收到的数据都是相同的
    """
    use_tq = _is_tq_enabled(cluster)

    if use_tq:
        from roll.utils.transferqueue_utils import (
            _pack_extra_info,
            _put_dataproto_to_batchmeta,
            _set_batchmeta_row_count,
        )

        def _dataproto_to_batchmeta_shards(data: DataProto, dp_size: int):
            td = data.batch
            data_mb = _calc_tensordict_size_mb(td)
            n_rows = td.batch_size[0] if td is not None and td.batch_size else 0
            dispatch_started_at = time.time()
            full_trace = _make_dispatch_trace(
                rows=n_rows,
                data_mb=data_mb,
                dp_size=dp_size,
                partition_id=TQ_DISPATCH_PARTITION_ID,
                driver_put_started_at=dispatch_started_at,
            )
            full_meta = _put_dataproto_to_batchmeta(
                data,
                partition_id=TQ_DISPATCH_PARTITION_ID,
                scene="dispatch_driver_put",
                trace_key=TQ_DISPATCH_TRACE_KEY,
                trace_payload=full_trace,
            )
            full_trace["driver_put_cost"] = time.time() - dispatch_started_at
            full_meta.extra_info = _pack_extra_info(
                _attach_trace(data.meta_info, TQ_DISPATCH_TRACE_KEY, full_trace),
                data.non_tensor_batch,
            )

            dp_shards = data.chunk(chunks=dp_size)
            all_idx = list(full_meta.global_indexes)
            all_partition_ids = list(full_meta.partition_ids)
            shards = []
            start = 0

            for shard_data in dp_shards:
                shard_rows = len(shard_data)
                end = start + shard_rows
                shard_idx = all_idx[start:end]
                if shard_rows == 0:
                    shard_partition_ids = []
                elif len(all_partition_ids) == 1:
                    shard_partition_ids = all_partition_ids * shard_rows
                else:
                    shard_partition_ids = all_partition_ids[start:end]
                shard_meta = copy.deepcopy(full_meta)
                shard_meta.global_indexes = shard_idx
                shard_meta.partition_ids = shard_partition_ids
                _set_batchmeta_row_count(shard_meta, shard_rows)
                shard_trace = _make_dispatch_trace(
                    rows=shard_rows,
                    data_mb=_calc_dataproto_size_mb(shard_data),
                    dp_size=dp_size,
                    partition_id=TQ_DISPATCH_PARTITION_ID,
                    trace_id=full_trace["trace_id"],
                    driver_put_started_at=full_trace["driver_put_started_at"],
                    driver_put_cost=full_trace["driver_put_cost"],
                )
                shard_meta.extra_info = _pack_extra_info(
                    _attach_trace(shard_data.meta_info, TQ_DISPATCH_TRACE_KEY, shard_trace),
                    shard_data.non_tensor_batch,
                )
                shards.append(shard_meta)
                start = end
            return shards

        new_args = []
        for arg in args:
            if isinstance(arg, DataProto):
                new_args.append(_dataproto_to_batchmeta_shards(arg, cluster.dp_size))
            else:
                new_args.append(arg.chunk(chunks=cluster.dp_size) if hasattr(arg, "chunk") else [arg] * cluster.dp_size)
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, DataProto):
                new_kwargs[key] = _dataproto_to_batchmeta_shards(value, cluster.dp_size)
            else:
                new_kwargs[key] = value.chunk(chunks=cluster.dp_size) if hasattr(value, "chunk") else [value] * cluster.dp_size

        def _get_tq_arg(arg_list, rank_info):
            shard = arg_list[rank_info.dp_rank]
            if _dispatch_first and not (
                getattr(rank_info, "tp_rank", 0) == 0
                and getattr(rank_info, "cp_rank", 0) == 0
                and getattr(rank_info, "pp_rank", 0) == 0
            ):
                empty = copy.deepcopy(shard)
                empty.global_indexes = []
                empty.partition_ids = []
                _set_batchmeta_row_count(empty, 0)
                return empty
            return shard

        all_args = tuple(
            [_get_tq_arg(arg_list, cluster.get_rank_info(rank=i)) for i in range(cluster.world_size)]
            for arg_list in new_args
        )
        all_kwargs = {
            key: [_get_tq_arg(value_list, cluster.get_rank_info(rank=i)) for i in range(cluster.world_size)]
            for key, value_list in new_kwargs.items()
        }
        return all_args, all_kwargs

    splitted_args, splitted_kwargs = _split_args_kwargs(cluster.dp_size, *args, **kwargs)
    all_args = []

    def get_arg_by_rank_info(arg, rank_info):
        local_dp_rank = rank_info.dp_rank
        if (
            _dispatch_first
            and isinstance(arg[local_dp_rank], DataProto)
            and not (rank_info.tp_rank == 0 and rank_info.cp_rank == 0 and rank_info.pp_rank == 0)
        ):
            return DataProto(batch=None, meta_info=arg[local_dp_rank].meta_info)
        return arg[local_dp_rank]

    for arg in splitted_args:
        if isinstance(arg, (Tuple, List)) and arg and isinstance(arg[0], DataProto):
            trace_id = uuid.uuid4().hex[:12]
            dispatch_started_at = time.time()
            traced = []
            for shard in arg:
                shard.meta_info = _attach_trace(
                    shard.meta_info,
                    NO_TQ_DISPATCH_TRACE_KEY,
                    {
                        "trace_id": trace_id,
                        "dispatch_started_at": dispatch_started_at,
                        "rows": len(shard),
                        "data_mb": _calc_dataproto_size_mb(shard),
                        "dp_size": cluster.dp_size,
                    },
                )
                traced.append(shard)
            arg = traced
        assert isinstance(arg, (Tuple, List)) and len(arg) == cluster.dp_size
        transformed_args = []
        for i in range(cluster.world_size):
            local_rank_info = cluster.get_rank_info(rank=i)
            transformed_args.append(get_arg_by_rank_info(arg, local_rank_info))
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in splitted_kwargs.items():
        if isinstance(v, (Tuple, List)) and v and isinstance(v[0], DataProto):
            trace_id = uuid.uuid4().hex[:12]
            dispatch_started_at = time.time()
            traced = []
            for shard in v:
                shard.meta_info = _attach_trace(
                    shard.meta_info,
                    NO_TQ_DISPATCH_TRACE_KEY,
                    {
                        "trace_id": trace_id,
                        "dispatch_started_at": dispatch_started_at,
                        "rows": len(shard),
                        "data_mb": _calc_dataproto_size_mb(shard),
                        "dp_size": cluster.dp_size,
                    },
                )
                traced.append(shard)
            v = traced
        assert isinstance(v, (Tuple, List)) and len(v) == cluster.dp_size
        transformed_v = []
        for i in range(cluster.world_size):
            local_rank_info = cluster.get_rank_info(rank=i)
            transformed_v.append(get_arg_by_rank_info(v, local_rank_info))
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def dispatch_dp_mp_compute(cluster, *args, **kwargs):
    return _dispatch_dp_mp_compute(cluster, False, *args, **kwargs)


def dispatch_dp_mp_dispatch_first(cluster, *args, **kwargs):
    return _dispatch_dp_mp_compute(cluster, True, *args, **kwargs)


def collect_dp_mp_compute(cluster, output):
    """
    只需要搜集tp=0, pipeline_last_stage的结果
    输入输出都是list, 是batch维度的
    """
    use_tq = _is_tq_enabled(cluster)
    is_refs = isinstance(output[0], ray.ObjectRef) if output else False
    ray_get_cost = 0.0
    raw_output = output

    if use_tq and is_refs:
        timeout = int(os.environ.get("roll_RPC_TIMEOUT", 3600)) if "roll_RPC_TIMEOUT" in os.environ else None
        t0 = time.time()
        raw_output = ray.get(list(output), timeout=timeout)
        ray_get_cost = time.time() - t0

    rep_results = _collect_representative_results(cluster, raw_output)

    if use_tq:
        from roll.utils.transferqueue_utils import (
            _meta_to_dataproto,
            _split_extra_info,
            is_batch_meta,
            merge_batch_metas,
        )

        if rep_results and is_batch_meta(rep_results[0]):
            collect_traces = []
            for meta in rep_results:
                meta_info, _ = _split_extra_info(getattr(meta, "extra_info", None))
                trace = meta_info.get(TQ_COLLECT_TRACE_KEY)
                if isinstance(trace, dict):
                    collect_traces.append(trace)

            merged_meta = merge_batch_metas(rep_results)
            total_rows = len(getattr(merged_meta, "global_indexes", []))
            t0 = time.time()
            result = _meta_to_dataproto(
                merged_meta,
                scene="collect_driver_read",
                trace_key=TQ_COLLECT_TRACE_KEY,
                meta_count=len(rep_results),
            )
            tq_cost = max(0.0, time.time() - t0)
            result_mb = _calc_dataproto_size_mb(result)
            _log_tq_collect_e2e(
                collect_traces,
                total_rows=total_rows,
                data_mb=result_mb,
                driver_read_cost=tq_cost,
            )
            return result

    if is_refs:
        output_in_dp = []
        for global_rank in range(cluster.world_size):
            local_rank_info = cluster.get_rank_info(rank=global_rank)
            output_in_dp.append(ObjectRefWrap(output[global_rank], collected=_is_representative_rank(local_rank_info)))
        return output_in_dp

    if isinstance(rep_results[0], list):
        return list(chain.from_iterable(rep_results))
    if isinstance(rep_results[0], DataProto):
        from roll.utils.transferqueue_utils import _pop_trace

        result_mb = sum(_calc_dataproto_size_mb(result) for result in rep_results)
        total_rows = sum(result.batch.batch_size[0] for result in rep_results if result.batch is not None)
        collect_traces = []
        for result in rep_results:
            trace = _pop_trace(result.meta_info, NO_TQ_COLLECT_TRACE_KEY)
            if isinstance(trace, dict):
                collect_traces.append(trace)
        _log_no_tq_collect_e2e(
            collect_traces,
            total_rows=total_rows,
            data_mb=result_mb,
            ray_get_cost=ray_get_cost,
        )
        return DataProto.concat(rep_results)
    raise NotImplementedError(f"output type {type(rep_results[0])}")


predefined_dispatch_mode_fn = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ONE_TO_ALL_ONE: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_one,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_MP_COMPUTE: {
        "dispatch_fn": dispatch_dp_mp_compute,
        "collect_fn": collect_dp_mp_compute,
    },
    Dispatch.DP_MP_DISPATCH_FIRST: {
        "dispatch_fn": dispatch_dp_mp_dispatch_first,
        "collect_fn": collect_dp_mp_compute,
    },
    Dispatch.DP_MP_DISPATCH_FIRST_COLLECT_ALL: {
        "dispatch_fn": dispatch_dp_mp_dispatch_first,
        "collect_fn": collect_all_to_all,
    }
}


def get_predefined_dispatch_fn(dispatch_mode):
    return predefined_dispatch_mode_fn[dispatch_mode]


predefined_execute_mode_fn = {
    Execute.ALL: {"execute_fn_name": "execute_all"},
    Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
}


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    return predefined_execute_mode_fn[execute_mode]


def func_generator(cls, method_name, dispatch_fn, collect_fn, execute_fn):
    def func(*args, blocking=True, **kwargs):
        bench_timing_enabled = (
            os.environ.get("ROLL_TRANSPORT_BENCHMARK_TIMING", "0") == "1" and method_name == "transport_benchmark"
        )
        driver_started_at = time.perf_counter() if bench_timing_enabled else None
        dispatch_prepare_cost = submit_cost = driver_wait_cost = driver_collect_cost = None

        if method_name == "initialize":
            setattr(cls, "initialized", True)

        dispatch_started_at = time.perf_counter() if bench_timing_enabled else None
        args, kwargs = dispatch_fn(cls, *args, **kwargs)
        if bench_timing_enabled:
            dispatch_prepare_cost = time.perf_counter() - dispatch_started_at

        submit_started_at = time.perf_counter() if bench_timing_enabled else None
        output = execute_fn(method_name, *args, **kwargs)
        if bench_timing_enabled:
            submit_cost = time.perf_counter() - submit_started_at
        if blocking:
            timeout = None
            if "roll_RPC_TIMEOUT" in os.environ:
                timeout = int(os.environ.get("roll_RPC_TIMEOUT"))
            wait_started_at = time.perf_counter() if bench_timing_enabled else None
            output = ray.get(output, timeout=timeout)
            if bench_timing_enabled:
                driver_wait_cost = time.perf_counter() - wait_started_at
        collect_started_at = time.perf_counter() if bench_timing_enabled else None
        output = collect_fn(cls, output)
        if bench_timing_enabled:
            driver_collect_cost = time.perf_counter() - collect_started_at
            total_cost = time.perf_counter() - driver_started_at
            mode = "tq" if _is_tq_enabled(cls) else "dataproto"
            logger.info(
                f"[BENCH driver timing] {method_name} | mode={mode} "
                f"| dispatch_prepare={dispatch_prepare_cost:.3f}s "
                f"| submit={submit_cost:.3f}s "
                f"| wait={(driver_wait_cost or 0.0):.3f}s "
                f"| collect={driver_collect_cost:.3f}s "
                f"| total={total_cost:.3f}s"
            )
        return output

    return func


def _check_dispatch_mode(dispatch_mode):
    assert isinstance(
        dispatch_mode, (Dispatch, Dict)
    ), f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    if isinstance(dispatch_mode, Dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, clear_cache=True):
    _check_dispatch_mode(dispatch_mode)
    _check_execute_mode(execute_mode)

    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode}
        bridge_dispatch_modes = {
            Dispatch.DP_MP_COMPUTE,
            Dispatch.DP_MP_DISPATCH_FIRST,
            Dispatch.DP_MP_DISPATCH_FIRST_COLLECT_ALL,
        }
        if isinstance(dispatch_mode, dict) or dispatch_mode in bridge_dispatch_modes:
            from roll.utils.transferqueue_utils import tqbridge

            wrapped_func = tqbridge(dispatch_mode)(func)
        else:
            wrapped_func = func
        if is_async:
            @wraps(func)
            async def inner_async(*args, **kwargs):
                try:
                    result = await wrapped_func(*args, **kwargs)
                    if clear_cache:
                        try:
                            current_platform.clear_cublas_workspaces()
                            gc.collect()
                            current_platform.empty_cache()
                        except Exception as oe:
                            pass

                except Exception as e:
                    logger.error(str(e))
                    logger.error(traceback.format_exc())
                    raise e
                return result

            setattr(inner_async, BIND_WORKER_METHOD_FLAG, attrs)
            return inner_async
        else:
            @wraps(func)
            def inner(*args, **kwargs):
                try:
                    result = wrapped_func(*args, **kwargs)
                    if clear_cache:
                        try:
                            current_platform.clear_cublas_workspaces()
                            gc.collect()
                            current_platform.empty_cache()
                        except Exception as oe:
                            pass

                except Exception as e:
                    logger.error(str(e))
                    logger.error(traceback.format_exc())
                    raise e
                return result

            setattr(inner, BIND_WORKER_METHOD_FLAG, attrs)
            return inner

    return decorator
