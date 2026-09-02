#!/usr/bin/env bash
# Run raw YOLO26 pose ONNX on video; decoding and NMS run in run_onnx_pose.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="/home/haiqv/exp/pose_estimation/ultralytics/runs/pose/train-7/export/best.onnx"
OUTPUT="/home/haiqv/exp/pose_estimation/ultralytics/runs/pose/train-7/onnx_predict"

if (($# == 0)); then
    echo "Usage: $0 /path/to/video [run_onnx_pose.py options]" >&2
    exit 2
fi

SOURCE="$1"
shift

python "${SCRIPT_DIR}/run_onnx_pose.py" \
    --model "${MODEL}" \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    "$@"
