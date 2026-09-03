#!/usr/bin/env python3
"""Consolidate v2 smoke attempts into the authoritative 25-target matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vadbench.environment_registry import load_encoder_candidates  # noqa: E402


def load_attempts(root: Path) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[str, list[dict[str, Any]]] = {}
    for path in root.glob("**/matrix-v2.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data.get("items", []):
            record = dict(item)
            record["matrix_path"] = path.relative_to(PROJECT_ROOT).as_posix()
            record["mtime_ns"] = path.stat().st_mtime_ns
            attempts.setdefault(str(record["integration_id"]), []).append(record)
    return attempts


def choose_attempt(values: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not values:
        return None
    successful = [
        value
        for value in values
        if value.get("technical_status", value.get("status")) == "smoke_pass"
    ]
    pool = successful or values
    return max(pool, key=lambda value: int(value.get("mtime_ns", 0)))


def consolidate(smoke_root: Path) -> dict[str, Any]:
    candidates = load_encoder_candidates(PROJECT_ROOT)
    attempts = load_attempts(smoke_root)
    items = []
    for candidate in candidates:
        encoder_id = candidate["id"]
        state = candidate["registration_state"]
        license_state = candidate["license_state"]
        selected = choose_attempt(attempts.get(encoder_id, []))
        if state == "candidate_only":
            status = "unregistered"
        elif state == "awaiting_manual_asset":
            status = "manual_required"
        elif license_state != "verified":
            status = "blocked_license"
        elif selected is None:
            status = "not_run"
        else:
            status = str(selected.get("technical_status", selected.get("status", "failed")))
        items.append(
            {
                "integration_id": encoder_id,
                "group": candidate["group"],
                "registration_state": state,
                "asset_state": candidate["asset_state"],
                "license_state": license_state,
                "status": status,
                "attempt": selected,
            }
        )
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in sorted({item["status"] for item in items})
    }
    return {"schema_version": 2, "target_count": 25, "counts": counts, "items": items}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", default="outputs/encoder-v2")
    parser.add_argument(
        "--output",
        default="outputs/environment-migration-v2/native-smoke-matrix.json",
    )
    args = parser.parse_args(argv)
    smoke_root = Path(args.smoke_root)
    if not smoke_root.is_absolute():
        smoke_root = PROJECT_ROOT / smoke_root
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    payload = consolidate(smoke_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"counts": payload["counts"], "output": str(output)}, ensure_ascii=False))
    return 0 if payload["counts"].get("smoke_pass") == 14 else 1


if __name__ == "__main__":
    raise SystemExit(main())
