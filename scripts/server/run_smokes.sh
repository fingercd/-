#!/usr/bin/env bash
set -euo pipefail

project_root="${VAD_PROJECT_ROOT:-/users/fotile/VAD}"
python_bin="${VAD_PYTHON:-$project_root/.venv/bin/python}"
video_path="${VAD_SMOKE_VIDEO:-$project_root/data/smoke/surveillance-smoke.mp4}"
gpu="${VAD_GPU:-5}"
selection="${VAD_SMOKE_ENCODER:-both}"
runtime_lib="${VAD_RUNTIME_LIB:-/users/fotile/miniconda3/envs/mllm-comp-platform/lib}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment missing: $python_bin" >&2
  exit 2
fi
if [[ ! -f "$video_path" ]]; then
  echo "Smoke video missing: $video_path" >&2
  exit 2
fi

used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
if (( used_mib > 1024 )); then
  echo "GPU $gpu is already using ${used_mib} MiB; refusing to start" >&2
  exit 3
fi

cd "$project_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export LD_LIBRARY_PATH="$runtime_lib:${LD_LIBRARY_PATH:-}"

if [[ "$selection" == "both" || "$selection" == "videomaev2" ]]; then
  "$python_bin" -m vadbench smoke \
    -c configs/experiments/ucf_videomaev2_weak.yaml \
    --video "$video_path" \
    --chunks 1 \
    --output outputs/server-smoke/videomaev2.json
fi

if [[ "$selection" == "both" || "$selection" == "hermes" ]]; then
  "$python_bin" -m vadbench smoke \
    -c configs/experiments/ucf_hermes_stream.yaml \
    --video "$video_path" \
    --chunks 2 \
    --output outputs/server-smoke/hermes.json
fi

if [[ "$selection" != "both" && "$selection" != "videomaev2" && "$selection" != "hermes" ]]; then
  echo "VAD_SMOKE_ENCODER must be both, videomaev2, or hermes" >&2
  exit 2
fi
