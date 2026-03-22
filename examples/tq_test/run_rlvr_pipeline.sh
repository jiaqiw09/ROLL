#!/bin/bash
set +x
ulimit -n 65536
CONFIG_PATH=$(basename $(dirname $0))
python examples/start_rlvr_pipeline.py --config_path $CONFIG_PATH  --config_name rlvr_zero3_sp2
