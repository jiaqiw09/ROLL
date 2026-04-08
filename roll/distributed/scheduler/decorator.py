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

from roll.distributed.scheduler.protocol import DataProto, ObjectRefWrap
from roll.utils.logging import get_logger
from roll.platforms import current_platform

logger = get_logger()

BIND_WORKER_METHOD_FLAG = "BIND_WORKER_METHOD_FLAG"
TQ_DISPATCH_PARTITION_ID = "train_dispatch"


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


def _set_batchmeta_row_count(meta, rows: int) -> None:
    """Best-effort sync of row-count fields used by TransferQueue metadata."""
    for attr_name in ("size", "_size"):
        try:
            setattr(meta, attr_name, rows)
        except Exception:
            pass


def _is_tq_enabled(cluster) -> bool:
    """Return True if TransferQueue is enabled in the current process.

    TQ is configured once at pipeline level, then initialized per process
    (driver / scheduler / worker). Dispatch and collect happen on the driver
    side, so they should check the driver's TQ runtime state instead of
    depending on whether the Cluster wrapper happens to carry pipeline_config.
    """
    from roll.utils.transferqueue_utils import is_tq_runtime_enabled

    return is_tq_runtime_enabled()


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


def _dispatch_dp_mp_compute(cluster, _dispatch_first, *args, **kwargs):
    """
    将输入chunk成dp_world_size份，按dp_rank为每个worker组织数据 -> 同一dp_rank收到的数据都是相同的

    When TQ is enabled:
      - Each DataProto arg is written to TQ in bulk; workers receive a BatchMeta
        slice instead of the full tensor payload.
      - tp/pp ranks that would receive None (dispatch_first mode) receive an empty
        BatchMeta so type checking in @tqbridge stays consistent.
    """
    use_tq = _is_tq_enabled(cluster)

    if use_tq:
        import transfer_queue as tq_mod
        tq_client = tq_mod.get_client()

        def _dataproto_to_batchmeta_shards(data: DataProto, dp_size: int):
            """Write *data* to TQ and return a list of dp_size BatchMeta shards."""
            import asyncio
            import copy
            import time

            td = data.batch  # TensorDict
            meta_info = data.meta_info
            non_tensor_batch = data.non_tensor_batch

            data_mb = _calc_tensordict_size_mb(td)
            n_rows = td.batch_size[0] if td.batch_size else 0
            logger.info(
                f"[TQ dispatch] TRIGGERED | rows={n_rows} | tensor_data={data_mb:.2f}MB "
                f"(this would have gone through Ray) | dp_size={dp_size}"
            )

            def _run(coro):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()

            t0 = time.time()
            from roll.utils.transferqueue_utils import _pack_extra_info
            full_meta = _run(tq_client.async_put(
                data=td,
                metadata=None,
                partition_id=TQ_DISPATCH_PARTITION_ID,
            ))
            full_meta.extra_info = _pack_extra_info(meta_info, non_tensor_batch)
            tq_cost = time.time() - t0
            logger.info(
                f"[TQ dispatch] PUT DONE | rows={n_rows} | data={data_mb:.2f}MB "
                f"| TQ write cost={tq_cost:.3f}s | Ray would have serialized ~{data_mb:.2f}MB on copy"
            )

            # Keep TQ shard boundaries aligned with DataProto.chunk(), so tensor rows,
            # non_tensor_batch rows and meta extra_info all describe the same slice.
            dp_shards = data.chunk(chunks=dp_size)
            all_idx = list(full_meta.global_indexes)
            all_partition_ids = list(full_meta.partition_ids)
            shards = []
            start = 0

            def _slice_partition_ids(end: int, shard_rows: int):
                if shard_rows == 0 or len(all_partition_ids) == 0:
                    return []
                if len(all_partition_ids) == 1:
                    return all_partition_ids * shard_rows
                return all_partition_ids[start:end]

            for dp_rank, shard_data in enumerate(dp_shards):
                shard_rows = len(shard_data)
                end = start + shard_rows
                shard_idx = all_idx[start:end]
                shard_partition_ids = _slice_partition_ids(end, shard_rows)
                shard_meta = copy.deepcopy(full_meta)
                shard_meta.global_indexes = shard_idx
                shard_meta.partition_ids = shard_partition_ids
                _set_batchmeta_row_count(shard_meta, shard_rows)
                shard_meta.extra_info = _pack_extra_info(
                    shard_data.meta_info,
                    shard_data.non_tensor_batch,
                )
                logger.info(
                    f"[TQ dispatch] SHARD READY | dp_rank={dp_rank} | rows={shard_rows} "
                    f"| indexes={len(shard_idx)} | partition_ids={len(shard_partition_ids)} "
                    f"| meta_size={getattr(shard_meta, 'size', None)} "
                    f"| fields={getattr(shard_meta, 'field_names', None)} "
                    f"| non_tensor_keys={list(shard_data.non_tensor_batch.keys())}"
                )
                shards.append(shard_meta)
                start = end

            if start != len(all_idx):
                logger.warning(
                    f"[TQ dispatch] shard index accounting mismatch: consumed={start}, "
                    f"available={len(all_idx)}, dp_size={dp_size}"
                )
            return shards

        # Replace DataProto positional args with BatchMeta shard lists
        new_args = []
        for arg in args:
            if isinstance(arg, DataProto):
                shards = _dataproto_to_batchmeta_shards(arg, cluster.dp_size)
                new_args.append(shards)
            else:
                # Non-DataProto args: split as usual
                new_args.append(arg.chunk(chunks=cluster.dp_size) if hasattr(arg, 'chunk') else [arg] * cluster.dp_size)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, DataProto):
                new_kwargs[k] = _dataproto_to_batchmeta_shards(v, cluster.dp_size)
            else:
                new_kwargs[k] = v.chunk(chunks=cluster.dp_size) if hasattr(v, 'chunk') else [v] * cluster.dp_size

        # Assign per-worker: tp/pp non-primary ranks get an empty BatchMeta
        from transfer_queue import BatchMeta as _BatchMeta

        def _get_tq_arg(arg_list, rank_info):
            shard = arg_list[rank_info.dp_rank]
            if _dispatch_first and not (rank_info.tp_rank == 0 and rank_info.cp_rank == 0 and rank_info.pp_rank == 0):
                # Non-primary rank: send an empty BatchMeta placeholder, but keep
                # extra_info so tqbridge can restore the original DataProto meta_info.
                empty = copy.deepcopy(shard)
                empty.global_indexes = []
                empty.partition_ids = []
                for attr_name in ("size", "_size"):
                    try:
                        setattr(empty, attr_name, 0)
                    except Exception:
                        pass
                return empty
            return shard

        all_args = tuple(
            [_get_tq_arg(arg_list, cluster.get_rank_info(rank=i)) for i in range(cluster.world_size)]
            for arg_list in new_args
        )
        all_kwargs = {
            k: [_get_tq_arg(v_list, cluster.get_rank_info(rank=i)) for i in range(cluster.world_size)]
            for k, v_list in new_kwargs.items()
        }
        return all_args, all_kwargs

    # ---- original path (no TQ) ----
    splitted_args, splitted_kwargs = _split_args_kwargs(cluster.dp_size, *args, **kwargs)
    for arg in args:
        if isinstance(arg, DataProto):
            rows = arg.batch.batch_size[0] if arg.batch is not None else 0
            data_mb = _calc_dataproto_size_mb(arg)
            logger.info(
                f"[NO TQ dispatch] TRIGGERED | rows={rows} | tensor_data={data_mb:.2f}MB "
                f"| dp_size={cluster.dp_size} | ray_payload={data_mb:.2f}MB | rss={_get_process_rss_gb():.3f}GB"
            )
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
        assert isinstance(arg, (Tuple, List)) and len(arg) == cluster.dp_size
        transformed_args = []
        for i in range(cluster.world_size):
            local_rank_info = cluster.get_rank_info(rank=i)
            transformed_args.append(get_arg_by_rank_info(arg, local_rank_info))
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in splitted_kwargs.items():
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

    When TQ is enabled:
      - Representative ranks (tp=0, last pp stage, cp=0) have written their output
        to TQ via @tqbridge and returned a BatchMeta.
      - We merge those BatchMeta global_indexes and do a single get_data() call.
    """
    use_tq = _is_tq_enabled(cluster)

    # Materialise Ray refs first when blocking=False
    raw_output = output
    is_refs = isinstance(output[0], ray.ObjectRef) if output else False
    if is_refs:
        import os, ray as _ray
        import time
        timeout = int(os.environ.get("roll_RPC_TIMEOUT", 3600))
        t0 = time.time()
        raw_output = _ray.get(list(output), timeout=timeout)
        ray_get_cost = time.time() - t0
        logger.info(
            f"[NO TQ collect] ray.get DONE | world_size={cluster.world_size} "
            f"| cost={ray_get_cost:.3f}s | rss={_get_process_rss_gb():.3f}GB"
        )

    # Identify representative ranks
    rep_results = []
    rep_metas = []
    for global_rank in range(cluster.world_size):
        local_rank_info = cluster.get_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.is_pipeline_last_stage and local_rank_info.cp_rank == 0:
            val = raw_output[global_rank]
            rep_results.append(val)

    if use_tq:
        from transfer_queue import BatchMeta as _BatchMeta
        # Check if results are BatchMeta (TQ path)
        if rep_results and isinstance(rep_results[0], _BatchMeta):
            import time
            from roll.utils.transferqueue_utils import _meta_to_dataproto, _split_extra_info

            # Merge global_indexes from all dp representative ranks
            merged_idx = []
            merged_partition_ids = []
            extra_info = None
            for meta in rep_results:
                merged_idx.extend(list(meta.global_indexes))
                merged_partition_ids.extend(list(meta.partition_ids))
                if extra_info is None:
                    extra_info = getattr(meta, "extra_info", None)

            import copy

            merged_meta = copy.deepcopy(rep_results[0]) if rep_results else _BatchMeta(
                global_indexes=[],
                partition_ids=[],
            )
            merged_meta.global_indexes = merged_idx
            merged_meta.partition_ids = merged_partition_ids
            _set_batchmeta_row_count(merged_meta, len(merged_idx))
            if extra_info is not None:
                merged_meta.extra_info = extra_info

            logger.info(
                f"[TQ collect] TRIGGERED | merging {len(rep_results)} dp ranks "
                f"| total rows={len(merged_idx)}"
            )
            t0 = time.time()
            result = _meta_to_dataproto(merged_meta)
            tq_cost = time.time() - t0

            # Compute result size for comparison
            result_mb = 0.0
            if result.batch is not None:
                for v in result.batch.values():
                    if hasattr(v, 'nbytes'):
                        result_mb += v.nbytes
                result_mb /= (1024 * 1024)
            logger.info(
                f"[TQ collect] GET DONE | rows={len(merged_idx)} | data={result_mb:.2f}MB "
                f"| TQ read cost={tq_cost:.3f}s | rss={_get_process_rss_gb():.3f}GB "
                f"(without TQ, Ray would have returned ~{result_mb:.2f}MB from {len(rep_results)} workers)"
            )

            return result

    # ---- original path (no TQ / non-BatchMeta results) ----
    if is_refs:
        # Re-wrap as ObjectRefWrap for blocking=False callers
        output_in_dp = []
        for global_rank in range(cluster.world_size):
            local_rank_info = cluster.get_rank_info(rank=global_rank)
            collected = (
                local_rank_info.tp_rank == 0
                and local_rank_info.is_pipeline_last_stage
                and local_rank_info.cp_rank == 0
            )
            output_in_dp.append(ObjectRefWrap(output[global_rank], collected=collected))
        return output_in_dp

    if isinstance(rep_results[0], list):
        return list(chain.from_iterable(rep_results))
    elif isinstance(rep_results[0], DataProto):
        result_mb = sum(_calc_dataproto_size_mb(result) for result in rep_results)
        total_rows = sum(result.batch.batch_size[0] for result in rep_results if result.batch is not None)
        logger.info(
            f"[NO TQ collect] CONCAT TRIGGERED | dp_results={len(rep_results)} "
            f"| total_rows={total_rows} | data={result_mb:.2f}MB | rss={_get_process_rss_gb():.3f}GB"
        )
        return DataProto.concat(rep_results)
    else:
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
        if method_name == "initialize":
            setattr(cls, "initialized", True)

        args, kwargs = dispatch_fn(cls, *args, **kwargs)
        output = execute_fn(method_name, *args, **kwargs)
        if blocking:
            timeout = None
            if "roll_RPC_TIMEOUT" in os.environ:
                timeout = int(os.environ.get("roll_RPC_TIMEOUT"))
            output = ray.get(output, timeout=timeout)
        output = collect_fn(cls, output)
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
        if is_async:
            @wraps(func)
            async def inner_async(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
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
                    result = func(*args, **kwargs)
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
