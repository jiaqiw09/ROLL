#!/bin/bash
set +x

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
mkdir -p $TORCH_EXTENSIONS_DIR

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_rlvr_pipeline.py --config_path $CONFIG_PATH  --config_name rlvr_zero3_sp2
