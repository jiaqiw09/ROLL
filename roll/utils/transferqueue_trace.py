"""Logging and trace helpers for TransferQueue transport."""

import copy
import dataclasses
import os
import time
import uuid
from typing import Any, Optional

from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()

TQ_DISPATCH_TRACE_KEY = "_tq_dispatch_trace"
TQ_COLLECT_TRACE_KEY = "_tq_collect_trace"
NO_TQ_DISPATCH_TRACE_KEY = "_no_tq_dispatch_trace"
NO_TQ_COLLECT_TRACE_KEY = "_no_tq_collect_trace"

TRACE_CONFIG = {
    "log_enable": True,
    "log_level": os.getenv("ROLL_TQ_LOGGING_LEVEL", "INFO"),
    "log_scenes": None,
}


def _config_to_dict(config: Any) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return copy.deepcopy(config)
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if hasattr(config, "__dict__"):
        return copy.deepcopy(vars(config))
    return {}


def configure_tq_trace(config: Any = None) -> None:
    config_dict = _config_to_dict(config)
    TRACE_CONFIG["log_enable"] = config_dict.get("log_enable", TRACE_CONFIG["log_enable"])
    TRACE_CONFIG["log_level"] = config_dict.get("log_level", TRACE_CONFIG["log_level"])
    TRACE_CONFIG["log_scenes"] = config_dict.get("log_scenes", TRACE_CONFIG["log_scenes"])
    logger.setLevel(TRACE_CONFIG["log_level"])


def _scene_enabled(scene: str) -> bool:
    if not TRACE_CONFIG["log_enable"]:
        return False
    log_scenes = TRACE_CONFIG.get("log_scenes")
    return log_scenes is None or scene in log_scenes


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
        iterable = container.values() if isinstance(container, dict) else container
        for item in iterable:
            if isinstance(item, DataProto):
                trace = _pop_trace(item.meta_info, trace_key)
                if trace is not None:
                    return trace
    return None


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def _make_dispatch_trace(
    *,
    rows: int,
    data_mb: float,
    dp_size: Optional[int] = None,
    partition_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    driver_put_started_at: Optional[float] = None,
    driver_put_cost: Optional[float] = None,
) -> dict:
    return {
        "trace_id": trace_id or _new_trace_id(),
        "rows": rows,
        "data_mb": data_mb,
        "dp_size": dp_size,
        "partition_id": partition_id,
        "driver_put_started_at": driver_put_started_at if driver_put_started_at is not None else time.time(),
        "driver_put_cost": driver_put_cost,
    }


def _make_collect_trace(
    *,
    rows: int,
    data_mb: float,
    worker_name: str,
    partition_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    worker_write_started_at: Optional[float] = None,
    worker_write_cost: Optional[float] = None,
) -> dict:
    return {
        "trace_id": trace_id or _new_trace_id(),
        "rows": rows,
        "data_mb": data_mb,
        "worker_name": worker_name,
        "partition_id": partition_id,
        "worker_write_started_at": worker_write_started_at if worker_write_started_at is not None else time.time(),
        "worker_write_cost": worker_write_cost,
    }


def _log_tq_conversion(
    *,
    direction: str,
    scene: str,
    rows: int,
    data_mb: float,
    cost: float,
    trace_id: Optional[str] = None,
    worker_name: Optional[str] = None,
    domain: Optional[str] = None,
    prompt_id: Optional[int] = None,
    meta_count: Optional[int] = None,
    partition_id: Optional[str] = None,
):
    if not _scene_enabled(scene):
        return
    fields = [
        f"direction={direction}",
        f"scene={scene}",
    ]
    if trace_id:
        fields.append(f"trace_id={trace_id}")
    if worker_name:
        fields.append(f"worker={worker_name}")
    if domain:
        fields.append(f"domain={domain}")
    if prompt_id is not None:
        fields.append(f"prompt_id={prompt_id}")
    if meta_count is not None:
        fields.append(f"meta_count={meta_count}")
    if partition_id:
        fields.append(f"partition={partition_id}")
    fields.extend(
        [
            f"rows={rows}",
            f"data={data_mb:.2f}MB",
            f"cost={cost:.3f}s",
        ]
    )
    logger.info("[TQ convert] " + " | ".join(fields))


def _log_tq_dispatch_e2e(func_name: str, trace: dict, *, rows: int, data_mb: float, worker_read_cost: float):
    if not _scene_enabled("dispatch_e2e"):
        return
    total = max(0.0, time.time() - trace["driver_put_started_at"])
    driver_put_cost = trace.get("driver_put_cost") or 0.0
    transfer_overhead = max(0.0, total - driver_put_cost - worker_read_cost)
    logger.info(
        f"[TQ dispatch E2E] {func_name} | trace_id={trace['trace_id']} | rows={rows} | data={data_mb:.2f}MB "
        f"| total={total:.3f}s | driver_put={driver_put_cost:.3f}s | worker_read={worker_read_cost:.3f}s "
        f"| transfer={transfer_overhead:.3f}s"
    )


def _log_tq_collect_e2e(trace_group: list[dict], *, total_rows: int, data_mb: float, driver_read_cost: float):
    if not _scene_enabled("collect_e2e"):
        return
    if not trace_group:
        logger.info(
            f"[TQ collect E2E] rows={total_rows} | data={data_mb:.2f}MB | total={driver_read_cost:.3f}s "
            f"| max_worker_write=0.000s | driver_read={driver_read_cost:.3f}s | transfer=0.000s"
        )
        return

    worker_name = trace_group[0].get("worker_name", "unknown")
    first_write_started_at = min(trace["worker_write_started_at"] for trace in trace_group)
    max_worker_write_cost = max((trace.get("worker_write_cost") or 0.0) for trace in trace_group)
    total = max(0.0, time.time() - first_write_started_at)
    transfer_overhead = max(0.0, total - max_worker_write_cost - driver_read_cost)
    logger.info(
        f"[TQ collect E2E] {worker_name} | traces={len(trace_group)} | rows={total_rows} | data={data_mb:.2f}MB "
        f"| total={total:.3f}s | max_worker_write={max_worker_write_cost:.3f}s "
        f"| driver_read={driver_read_cost:.3f}s | transfer={transfer_overhead:.3f}s"
    )


def _log_no_tq_dispatch_e2e(func_name: str, trace: dict, *, rows: int, data_mb: float):
    if not _scene_enabled("no_tq_dispatch_e2e"):
        return
    total = max(0.0, time.time() - trace["dispatch_started_at"])
    logger.info(
        f"[NO TQ dispatch E2E] {func_name} | trace_id={trace['trace_id']} | rows={rows} | data={data_mb:.2f}MB "
        f"| total={total:.3f}s"
    )


def _log_no_tq_collect_e2e(trace_group: list[dict], *, total_rows: int, data_mb: float, ray_get_cost: float):
    if not _scene_enabled("no_tq_collect_e2e"):
        return
    if not trace_group:
        return
    worker_name = trace_group[0].get("worker_name", "unknown")
    first_write_started_at = min(trace["worker_write_started_at"] for trace in trace_group)
    total = max(0.0, time.time() - first_write_started_at)
    logger.info(
        f"[NO TQ collect E2E] {worker_name} | traces={len(trace_group)} | rows={total_rows} | data={data_mb:.2f}MB "
        f"| total={total:.3f}s | ray_get={ray_get_cost:.3f}s"
    )


def _log_tq_replay_write(*, prompt_id: int, response_count: int, total_rows: int, data_mb: float, total_write_cost: float):
    if not _scene_enabled("replay_summary"):
        return
    logger.info(
        f"[TQ replay] prompt_id={prompt_id} | responses={response_count} | rows={total_rows} "
        f"| data={data_mb:.2f}MB | total_write={total_write_cost:.3f}s"
    )
