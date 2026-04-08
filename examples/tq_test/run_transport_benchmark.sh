#!/bin/bash
set +x

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

# NPU / Ascend hints:
# 1. DEVICE_IDS is the logical device list used by this benchmark worker group.
# 2. NUM_DEVICES_PER_NODE is the total physical accelerator count on the node.
# 3. For NO TQ runs, leave USE_TQ=0. For TQ runs, set USE_TQ=1.

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

DEVICE_IDS=${DEVICE_IDS:-0,1,2,3}
NUM_DEVICES_PER_NODE=${NUM_DEVICES_PER_NODE:-4}
NUM_NODES=${NUM_NODES:-1}

BATCH_SIZE=${BATCH_SIZE:-16}
SEQ_LEN=${SEQ_LEN:-4096}
PAYLOAD_WIDTH=${PAYLOAD_WIDTH:-4194304}
PAYLOAD_TOTAL_MB=${PAYLOAD_TOTAL_MB:-}

USE_TQ=${USE_TQ:-0}
TQ_TOTAL_STORAGE_SIZE=${TQ_TOTAL_STORAGE_SIZE:-100000}
TQ_NUM_DATA_STORAGE_UNITS=${TQ_NUM_DATA_STORAGE_UNITS:-8}

CMD=(
  python -m examples.tq_test.start_transport_benchmark
  --devices ${DEVICE_IDS}
  --num-devices-per-node ${NUM_DEVICES_PER_NODE}
  --num-nodes ${NUM_NODES}
  --batch-size ${BATCH_SIZE}
  --seq-len ${SEQ_LEN}
)

if [ -n "${PAYLOAD_TOTAL_MB}" ]; then
  CMD+=(--payload-total-mb ${PAYLOAD_TOTAL_MB})
else
  CMD+=(--payload-width ${PAYLOAD_WIDTH})
fi

if [ "${USE_TQ}" = "1" ]; then
  CMD+=(
    --use-tq
    --tq-total-storage-size ${TQ_TOTAL_STORAGE_SIZE}
    --tq-num-data-storage-units ${TQ_NUM_DATA_STORAGE_UNITS}
  )
fi

cd "$(dirname "$0")/../.."
"${CMD[@]}"
