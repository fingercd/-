#!/usr/bin/env bash
set -euo pipefail

candidates=(
  /users/fotile/miniconda3/envs/h3/bin/python
  /users/fotile/miniconda3/envs/mllm-comp-vln-unified/bin/python
  /users/fotile/miniconda3/envs/mllm-comp-platform/bin/python
)

for python_bin in "${candidates[@]}"; do
  [[ -x "$python_bin" ]] || continue
  "$python_bin" - <<'PY' || true
import importlib.util
import json
import sys

names = ["torch", "torchvision", "transformers", "accelerate", "logzero", "cv2", "yaml"]
result = {"python": sys.version.split()[0], "executable": sys.executable}
for name in names:
    if importlib.util.find_spec(name) is None:
        result[name] = None
        continue
    try:
        module = __import__(name)
        result[name] = getattr(module, "__version__", "installed")
    except Exception as exc:
        result[name] = f"import-error:{type(exc).__name__}"
if result.get("torch") and not str(result["torch"]).startswith("import-error"):
    import torch
    result["cuda_available"] = torch.cuda.is_available()
    result["torch_cuda"] = torch.version.cuda
print(json.dumps(result, sort_keys=True))
PY
done

