import argparse
import os
from types import SimpleNamespace

os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/torch_extensions"
os.makedirs("/tmp/torch_extensions", exist_ok=True)

from roll.configs import ModelArguments
from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.utils.transferqueue_utils import init_tq

from examples.tq_test.transport_benchmark_example import (
    TransportBenchmarkWorker,
    build_transport_benchmark_batch,
    payload_width_from_total_bytes,
)


def _parse_gpus(gpus: str) -> list[int]:
    values = [item.strip() for item in gpus.split(",") if item.strip()]
    if not values:
        raise ValueError("--gpus must contain at least one GPU id, for example: 0,1")
    return [int(item) for item in values]


def _build_tq_config(total_storage_size: int, num_data_storage_units: int) -> dict:
    return {
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "total_storage_size": total_storage_size,
                "num_data_storage_units": num_data_storage_units,
            },
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Pure transport benchmark for DataProto vs TQ")
    parser.add_argument(
        "--gpus",
        "--devices",
        dest="gpus",
        type=str,
        required=True,
        help="Comma-separated device ids, for example: 0,1. Works for both GPU and NPU runs.",
    )
    parser.add_argument(
        "--num-gpus-per-node",
        "--num-devices-per-node",
        dest="num_gpus_per_node",
        type=int,
        required=True,
        help="Physical accelerator count per node in the Ray cluster, for example: 8",
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use from the Ray cluster")
    parser.add_argument("--batch-size", type=int, default=16, help="Synthetic batch size before DP sharding")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length for model-like input fields")
    parser.add_argument(
        "--payload-width",
        type=int,
        default=4 * 1024 * 1024,
        help="Number of float16 elements per row in transport_payload",
    )
    parser.add_argument(
        "--payload-total-mb",
        type=float,
        default=None,
        help="Target size in MB for the whole transport_payload tensor. Overrides --payload-width.",
    )
    parser.add_argument("--use-tq", action="store_true", help="Enable TransferQueue on the driver side")
    parser.add_argument(
        "--tq-total-storage-size",
        type=int,
        default=100000,
        help="TransferQueue SimpleStorage total_storage_size",
    )
    parser.add_argument(
        "--tq-num-data-storage-units",
        type=int,
        default=8,
        help="TransferQueue SimpleStorage num_data_storage_units",
    )
    args = parser.parse_args()

    gpu_ids = _parse_gpus(args.gpus)
    world_size = len(gpu_ids)
    device_mapping_expr = str(gpu_ids)

    init()

    if args.use_tq:
        init_tq(
            _build_tq_config(
                total_storage_size=args.tq_total_storage_size,
                num_data_storage_units=args.tq_num_data_storage_units,
            )
        )

    resource_manager = ResourceManager(
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
    )
    worker_config = WorkerConfig(
        name="transport-benchmark",
        worker_cls="examples.tq_test.transport_benchmark_example.TransportBenchmarkWorker",
        model_args=ModelArguments(model_name_or_path=None),
        strategy_args=StrategyArguments(strategy_name="hf_infer", strategy_config={}),
        world_size=world_size,
        device_mapping=device_mapping_expr,
        system_envs={},
        infer_batch_size=args.batch_size,
    )
    cluster = Cluster(
        name=worker_config.name,
        worker_cls=TransportBenchmarkWorker,
        resource_manager=resource_manager,
        worker_config=worker_config,
    )
    cluster.initialize(SimpleNamespace(resume_from_checkpoint=False))

    batch = build_transport_benchmark_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        payload_width=args.payload_width,
        payload_total_bytes=(
            None if args.payload_total_mb is None else int(args.payload_total_mb * 1024 * 1024)
        ),
    )
    result = cluster.transport_benchmark(batch, blocking=True)

    payload = result.batch["benchmark_output_payload"]
    input_summary = result.batch["benchmark_input_mb"]
    effective_payload_width = payload_width_from_total_bytes(
        total_bytes=int(args.payload_total_mb * 1024 * 1024),
        batch_size=args.batch_size,
    ) if args.payload_total_mb is not None else args.payload_width
    print("transport benchmark finished")
    print(f"use_tq={args.use_tq}")
    print(f"world_size={world_size}")
    print(f"rows={len(result)}")
    print(f"payload_total_mb={args.payload_total_mb}")
    print(f"effective_payload_width={effective_payload_width}")
    print(f"output_payload_shape={tuple(payload.shape)}")
    print(f"benchmark_input_mb_shape={tuple(input_summary.shape)}")

    resource_manager.destroy_placement_group()


if __name__ == "__main__":
    main()
