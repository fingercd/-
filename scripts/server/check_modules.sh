#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib.util
import json

names = [
    "accelerate",
    "av",
    "decord",
    "logzero",
    "qwen_vl_utils",
    "pandas",
    "flash_attn",
    "safetensors",
    "yaml",
    "sklearn",
]
print(json.dumps({name: importlib.util.find_spec(name) is not None for name in names}, sort_keys=True))
PY

