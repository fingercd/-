#!/usr/bin/env python3
"""Run native encoder smoke tests only through the isolated v2 environments."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vadbench.environment_registry import (  # noqa: E402
    assert_new_environment_executable,
    load_encoder_candidates,
    load_encoder_environment_registry,
)
from vadbench.integrations.catalog import load_default_integration_catalog  # noqa: E402


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_environment(
    encoder_id: str,
    python: Path,
    overlay: Path | None,
    cache_root: Path,
) -> dict[str, str]:
    env = dict(os.environ)
    paths = [str(PROJECT_ROOT / "src")]
    if overlay is not None:
        paths.append(str(overlay))
    if encoder_id == "videomamba":
        paths.extend(
            [
                str(PROJECT_ROOT / "external-v2/videomamba/mamba"),
                str(PROJECT_ROOT / "external-v2/videomamba/causal-conv1d"),
            ]
        )
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HOME"] = str(cache_root / "huggingface")
    env["TORCH_HOME"] = str(cache_root / "torch")
    env["PIP_CACHE_DIR"] = str(cache_root / "pip")
    env["PYTHONPYCACHEPREFIX"] = str(cache_root / "pycache" / encoder_id)
    env["TMPDIR"] = str(cache_root / "tmp" / encoder_id)
    env["CONDA_PREFIX"] = str(python.parents[1])
    env.pop("VIRTUAL_ENV", None)
    env["LD_LIBRARY_PATH"] = str(python.parents[1] / "lib")
    for key in ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "PYTHONPYCACHEPREFIX", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def candidate_is_runnable(
    candidate: dict[str, Any], include_license_blocked: bool
) -> tuple[bool, str | None]:
    if candidate["registration_state"] == "candidate_only":
        return False, "candidate_only"
    if candidate["registration_state"] == "awaiting_manual_asset":
        return False, "manual_asset_missing"
    if candidate["license_state"] != "verified" and not include_license_blocked:
        return False, "license_blocked"
    return True, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", action="append")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-license-blocked", action="store_true")
    parser.add_argument(
        "--video",
        default="data/smoke/mlvu-surveil-8.mp4",
    )
    parser.add_argument(
        "--output-root",
        default=None,
    )
    args = parser.parse_args(argv)

    environment = load_encoder_environment_registry(PROJECT_ROOT)
    catalog = load_default_integration_catalog(PROJECT_ROOT)
    by_candidate = {item["id"]: item for item in load_encoder_candidates(PROJECT_ROOT)}
    selected = list(args.id or catalog.ids)
    unknown = sorted(set(selected) - set(catalog.ids))
    if unknown:
        raise SystemExit(f"not registered runtime encoders: {unknown}")
    run_id = (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    output_root = (
        Path(args.output_root) if args.output_root else PROJECT_ROOT / "outputs/encoder-v2" / run_id
    )
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    if PROJECT_ROOT not in output_root.parents:
        raise RuntimeError("output_root must remain under project root")
    output_root.mkdir(parents=True, exist_ok=True)

    items = []
    for encoder_id in selected:
        candidate = by_candidate[encoder_id]
        runnable, skip_reason = candidate_is_runnable(candidate, args.include_license_blocked)
        group = environment.group_for(encoder_id)
        python = assert_new_environment_executable(group.prefix / "bin/python", environment)
        overlay = environment.overlay_for(encoder_id)
        item = {
            "integration_id": encoder_id,
            "group": group.id,
            "python_executable": str(python),
            "overlay": str(overlay) if overlay is not None else None,
            "base_marker_sha256": file_hash(group.prefix / ".vadbench-env-v2.json"),
            "overlay_marker_sha256": (
                file_hash(overlay / ".overlay-v2.json") if overlay is not None else None
            ),
        }
        if not runnable:
            item.update({"status": "skipped", "reason": skip_reason})
            items.append(item)
            continue
        if not python.is_file():
            item.update({"status": "blocked", "reason": "new_environment_missing"})
            items.append(item)
            continue
        item_root = output_root / encoder_id
        item_root.mkdir(parents=True, exist_ok=True)
        log_path = item_root / "launcher.log"
        command = [
            str(python),
            "-m",
            "vadbench",
            "integrations",
            "smoke",
            "--video",
            str(PROJECT_ROOT / args.video),
            "--id",
            encoder_id,
            "--device",
            args.device,
            "--output-root",
            str(item_root),
        ]
        env = build_environment(encoder_id, python, overlay, environment.cache_root)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        log_path.write_text(
            completed.stdout + "\n--- stderr ---\n" + completed.stderr,
            encoding="utf-8",
        )
        result_files = sorted(item_root.glob("**/result.json"))
        status = "failed"
        result_path = None
        if result_files:
            result_path = result_files[-1]
            try:
                status = str(
                    json.loads(result_path.read_text(encoding="utf-8")).get("status", "failed")
                )
            except Exception:
                status = "failed"
        technical_status = status
        if candidate["license_state"] != "verified" and status == "smoke_pass":
            status = "blocked_license"
        item.update(
            {
                "status": status,
                "technical_status": technical_status,
                "reason": (
                    "license_blocked_after_technical_pass" if status == "blocked_license" else None
                ),
                "exit_code": completed.returncode,
                "result_path": (
                    result_path.relative_to(PROJECT_ROOT).as_posix()
                    if result_path is not None
                    else None
                ),
                "log_path": log_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
        items.append(item)

    counts = {
        status: sum(item["status"] == status for item in items)
        for status in sorted({item["status"] for item in items})
    }
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "device": args.device,
        "video": args.video,
        "counts": counts,
        "items": items,
    }
    (output_root / "matrix-v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not counts.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
