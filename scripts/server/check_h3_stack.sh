#!/usr/bin/env bash
set -euo pipefail

/users/fotile/miniconda3/envs/h3/bin/python - <<'PY'
import importlib.util
import json

result = {}
for name in (
    "numpy",
    "PIL",
    "psutil",
    "safetensors",
    "tokenizers",
    "huggingface_hub",
    "torch",
    "torchvision",
    "transformers",
):
    if importlib.util.find_spec(name) is None:
        result[name] = None
        continue
    module = __import__(name)
    result[name] = getattr(module, "__version__", "installed")
print(json.dumps(result, sort_keys=True))
PY

