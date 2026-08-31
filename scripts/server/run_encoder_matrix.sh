#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VIDEO="${VADBENCH_VIDEO:-${1:-$ROOT/data/smoke/mlvu-surveil-8.mp4}}"
OUTPUT_ROOT="${VADBENCH_OUTPUT_ROOT:-$ROOT/outputs/encoder-integration/current-video}"
DEVICE="${VADBENCH_DEVICE:-cpu}"
MAIN_PYTHON="${VADBENCH_MAIN_PYTHON:-$ROOT/.venv/bin/python}"
PTV_PYTHON="${VADBENCH_PTV_PYTHON:-/users/fotile/miniconda3/envs/mllm-comp-awarevln/bin/python}"
HERMES_PYTHON="${VADBENCH_HERMES_PYTHON:-$ROOT/.venv-hermes/bin/python}"

if [[ ! -f "$VIDEO" ]]; then
  echo "视频不存在：$VIDEO" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

run_group() {
  local label="$1" python_bin="$2" ld_path="$3"
  shift 3
  local matrix_path="$OUTPUT_ROOT/matrix-${label}.json"
  echo "[$label] python=$python_bin device=$DEVICE output=$OUTPUT_ROOT"
  if [[ ! -x "$python_bin" ]]; then
    echo "[$label] Python 不存在或不可执行：$python_bin" >&2
    return 2
  fi
  if [[ -n "$ld_path" ]]; then
    LD_LIBRARY_PATH="$ld_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$python_bin" -m vadbench integrations matrix \
      --video "$VIDEO" --device "$DEVICE" --execute \
      --output-root "$OUTPUT_ROOT" --matrix-path "$matrix_path" "$@"
  else
    env -u LD_LIBRARY_PATH "$python_bin" -m vadbench integrations matrix \
      --video "$VIDEO" --device "$DEVICE" --execute \
      --output-root "$OUTPUT_ROOT" --matrix-path "$matrix_path" "$@"
  fi
}

# These are the routes backed by the explicit verified R(2+1)D compatibility
# checkpoint. The result aux/registry notes identify this scope per route.
run_group compatibility "$MAIN_PYTHON" "" \
  --id c3d --id timesformer --id videomae --id uniformerv2 --id umt \
  --id internvideo2 --id videomamba --id vjepa2 --id longvu --id videochat \
  --id videochat_online --id videochat_flash --id ma_lmm --id moviechat \
  --id streaming_vlm --id infinipot_v --id mukv

# Native PyTorchVideo assets live in the isolated awarevln environment.
run_group pytorchvideo "$PTV_PYTHON" "/users/fotile/miniconda3/envs/mllm-comp-awarevln/lib" \
  --id i3d --id x3d --id slowfast

# Native HERMES is kept in its own dependency environment. The CPU smoke is
# intentional: the script never claims an occupied GPU and never kills peers.
run_group hermes "$HERMES_PYTHON" "" --id hermes_llava_ov

# VideoMAE V2 imports Transformers audio helpers; include the environment's
# codec libraries explicitly, then reuse all successful per-item results.
H3_LIB="/users/fotile/miniconda3/envs/h3/lib:/users/fotile/miniconda3/envs/mllm-comp-internav/lib:/users/fotile/miniconda3/pkgs/libsndfile-1.2.2-hc7d488a_2/lib:/users/fotile/miniconda3/pkgs/libflac-1.5.0-he200343_1/lib:/users/fotile/miniconda3/pkgs/libopus-1.6.1-h280c20c_0/lib:/users/fotile/miniconda3/pkgs/libvorbis-1.3.7-h54a6638_2/lib:/users/fotile/miniconda3/pkgs/libogg-1.3.5-hd0c01bc_1/lib"
run_group videomaev2 "$MAIN_PYTHON" "$H3_LIB" --id videomaev2

# One final no-forward aggregation confirms that all 25 item results are
# present and successful under one canonical output root.
"$MAIN_PYTHON" -m vadbench integrations matrix \
  --video "$VIDEO" --device "$DEVICE" --execute \
  --output-root "$OUTPUT_ROOT" --matrix-path "$OUTPUT_ROOT/matrix.json" \
  --id r2plus1d_18 --id x3d --id mvitv2 --id slowfast --id c3d --id i3d \
  --id timesformer --id video_swin --id videomae --id videomaev2 --id uniformerv2 \
  --id umt --id internvideo2 --id videomamba --id vjepa2 --id longvu --id videochat \
  --id videochat_online --id videochat_flash --id ma_lmm --id moviechat \
  --id streaming_vlm --id infinipot_v --id hermes_llava_ov --id mukv

export VADBENCH_MATRIX_PATH="$OUTPUT_ROOT/matrix.json"
"$MAIN_PYTHON" - <<'PY'
import json, os, sys
path=os.environ["VADBENCH_MATRIX_PATH"]
data=json.load(open(path,encoding="utf-8"))
counts=data.get("counts",{})
print(f"final matrix: selected={data.get('selected_count')} counts={counts}")
sys.exit(0 if counts.get("smoke_pass") == 25 and len(counts) == 1 else 1)
PY
