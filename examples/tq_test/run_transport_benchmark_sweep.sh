#!/bin/bash
set +x

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

DEVICE_IDS=${DEVICE_IDS:-0,1,2,3}
NUM_DEVICES_PER_NODE=${NUM_DEVICES_PER_NODE:-4}
NUM_NODES=${NUM_NODES:-1}

BATCH_SIZE=${BATCH_SIZE:-16}
SEQ_LEN=${SEQ_LEN:-4096}

USE_TQ=${USE_TQ:-0}
TQ_TOTAL_STORAGE_SIZE=${TQ_TOTAL_STORAGE_SIZE:-100000}
TQ_NUM_DATA_STORAGE_UNITS=${TQ_NUM_DATA_STORAGE_UNITS:-8}

# Default sweep:
# 16MB -> 32MB -> 64MB ... -> 256GB
SIZE_SERIES_MB=${SIZE_SERIES_MB:-"16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144"}

# 1 means continue sweeping after a failed size, 0 means stop immediately.
CONTINUE_ON_FAILURE=${CONTINUE_ON_FAILURE:-1}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-2}

cd "$(dirname "$0")/../.."

for PAYLOAD_TOTAL_MB in ${SIZE_SERIES_MB}; do
  echo "============================================================"
  echo "transport benchmark sweep start"
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "use_tq=${USE_TQ}"
  echo "payload_total_mb=${PAYLOAD_TOTAL_MB}"
  echo "device_ids=${DEVICE_IDS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "seq_len=${SEQ_LEN}"
  echo "============================================================"

  CMD=(
    python -m examples.tq_test.start_transport_benchmark
    --devices ${DEVICE_IDS}
    --num-devices-per-node ${NUM_DEVICES_PER_NODE}
    --num-nodes ${NUM_NODES}
    --batch-size ${BATCH_SIZE}
    --seq-len ${SEQ_LEN}
    --payload-total-mb ${PAYLOAD_TOTAL_MB}
  )

  if [ "${USE_TQ}" = "1" ]; then
    CMD+=(
      --use-tq
      --tq-total-storage-size ${TQ_TOTAL_STORAGE_SIZE}
      --tq-num-data-storage-units ${TQ_NUM_DATA_STORAGE_UNITS}
    )
  fi

  "${CMD[@]}"
  STATUS=$?

  echo "transport benchmark sweep end | payload_total_mb=${PAYLOAD_TOTAL_MB} | status=${STATUS}"

  if [ "${STATUS}" != "0" ] && [ "${CONTINUE_ON_FAILURE}" != "1" ]; then
    exit ${STATUS}
  fi

  sleep ${SLEEP_BETWEEN_RUNS}
done
