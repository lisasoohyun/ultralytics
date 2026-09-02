#!/usr/bin/env bash
# Postprocess raw YOLO26 pose SoC tensors and save image overlays.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BIN="/home/haiqv/dell_shared/model/Behavior_AI/0_HPE/yolo26n-pose_0901/inf/yolo26n-pose_pcq/inf_rename_bin"
DEFAULT_OUTPUT="${SCRIPT_DIR}/runs/pose/train-7/soc_predict"

if (($# == 0)); then
    echo "Usage: $0 /path/to/image-dir-or-list [--soc-bin /path/to/dat/root] [run_onnx_pose.py options]" >&2
    exit 2
fi

SOURCE="$1"
shift

python "${SCRIPT_DIR}/run_onnx_pose.py" \
    --source "${SOURCE}" \
    --soc-bin "${DEFAULT_BIN}" \
    --output "${DEFAULT_OUTPUT}" \
    "$@"
