#!/usr/bin/env bash
set -euo pipefail

project_root="${VAD_PROJECT_ROOT:-/users/fotile/VAD}"
python_bin="${VAD_PYTHON:-$project_root/.venv/bin/python}"

cd "$project_root"
"$python_bin" scripts/fetch_upstreams.py --verify-only
"$python_bin" -m vadbench weights verify videomaev2-base-hf weights/videomaev2-base-hf
"$python_bin" -m vadbench weights verify hermes-llava-ov-0.5b weights/hermes-llava-ov-0.5b
"$python_bin" -m vadbench doctor --project-root "$project_root"

