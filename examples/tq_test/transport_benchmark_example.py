"""Example worker and batch builder for pure transport benchmarks.

This example is intentionally isolated from the training pipeline:
it lets you construct a large DataProto and send it through the same
dispatch/tqbridge/collect path without running model inference.

The benchmark is intentionally simple:
- build one large payload tensor on the driver
- dispatch it through Ray/TQ
- echo the same payload size back from the worker

So whatever size you dispatch is also the size you collect.
"""

import numpy as np
import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.transferqueue_utils import tqbridge


def _tensor_size_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def _make_payload(rows: int, payload_width: int, *, dtype=torch.float16) -> torch.Tensor:
    if payload_width <= 0:
        return torch.zeros((rows, 1), dtype=dtype)
    return torch.zeros((rows, payload_width), dtype=dtype)


def payload_width_from_total_bytes(total_bytes: int, batch_size: int, *, dtype=torch.float16) -> int:
    """Convert a desired total payload size into per-row width.

    The benchmark payload is a dense tensor of shape [batch_size, payload_width].
    This helper computes the smallest payload_width that reaches the requested
    total bytes for the whole batch.
    """
    if total_bytes <= 0:
        return 1
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    bytes_per_row = max((total_bytes + batch_size - 1) // batch_size, 1)
    width = (bytes_per_row + bytes_per_element - 1) // bytes_per_element
    return max(int(width), 1)


def build_transport_benchmark_batch(
    *,
    batch_size: int = 16,
    seq_len: int = 4096,
    payload_width: int = 4 * 1024 * 1024,
    payload_total_bytes: int | None = None,
) -> DataProto:
    """Build a large synthetic batch for transport-only benchmarking.

    Args:
        batch_size: Number of rows in the batch.
        seq_len: Sequence length for model-like fields.
        payload_width: Number of float16 elements per row in the synthetic
            transport payload. Larger values increase both dispatch and collect
            pressure. Since the worker echoes the payload back, the collect size
            matches the dispatched size.
        payload_total_bytes: Optional target size for the whole payload tensor.
            When provided, it overrides payload_width and computes the minimum
            per-row width that reaches the requested total batch size.
    """
    if payload_total_bytes is not None:
        payload_width = payload_width_from_total_bytes(
            total_bytes=payload_total_bytes,
            batch_size=batch_size,
            dtype=torch.float16,
        )

    input_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64)
    position_ids = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
    response_mask = torch.ones((batch_size, seq_len), dtype=torch.int64)
    payload = _make_payload(batch_size, payload_width, dtype=torch.float16)

    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "response_mask": response_mask,
        # Extra bulk payload that is ignored by the worker logic but still
        # travels through DataProto / Ray / TQ.
        "transport_payload": payload,
    }
    non_tensors = {
        "sample_uuid": np.array([f"bench-{i}" for i in range(batch_size)], dtype=object),
    }
    meta_info = {
        "global_step": 0,
        "benchmark_note": "synthetic transport benchmark",
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)


class TransportBenchmarkWorker(Worker):
    """Minimal worker that measures transport without model compute."""

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.logger.info(f"{self.worker_name} transport benchmark worker initialized")

    @register(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    @tqbridge(dispatch_mode=Dispatch.DP_MP_DISPATCH_FIRST)
    def transport_benchmark(self, data: DataProto) -> DataProto:
        """Round-trip a synthetic payload through the scheduler transport path.

        The method avoids model forward/backward entirely. It simply returns a
        clone of the large synthetic payload, so collect pressure matches
        dispatch pressure.
        """
        rows = len(data)
        input_payload_mb = sum(_tensor_size_mb(v) for v in data.batch.values())
        output_payload = data.batch["transport_payload"].clone()
        summary = torch.full((rows, 1), fill_value=input_payload_mb, dtype=torch.float32)

        output = DataProto.from_dict(
            tensors={
                "benchmark_input_mb": summary,
                "benchmark_output_payload": output_payload,
            },
            meta_info={
                "metrics": {
                    "benchmark/rows": rows,
                    "benchmark/input_payload_mb": input_payload_mb,
                    "benchmark/output_payload_mb": _tensor_size_mb(output_payload),
                }
            },
        )
        return output


def example_usage() -> str:
    return (
        "1. Build a batch with build_transport_benchmark_batch(...)\n"
        "2. Send it to worker.transport_benchmark(batch, blocking=True)\n"
        "3. Compare TQ dispatch/collect E2E vs NO TQ dispatch/collect E2E logs\n"
        "4. In multi-GPU runs, the batch is sharded by dp_size and each shard is echoed back"
    )
