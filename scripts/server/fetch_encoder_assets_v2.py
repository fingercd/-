#!/usr/bin/env python3
"""Verify local native checkpoints and acquire permitted missing assets on node2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vadbench.environment_registry import (  # noqa: E402
    load_encoder_candidates,
    load_encoder_environment_registry,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_asset_file(local_path: Path, relative: str) -> Path:
    if local_path.is_file():
        return local_path
    return local_path / relative


def verify_entry(entry: dict[str, Any]) -> dict[str, Any]:
    local = Path(str(entry["local_path"]))
    if not local.is_absolute():
        local = PROJECT_ROOT / local
    files = []
    passed = bool(entry.get("sha256"))
    for relative, expected in dict(entry.get("sha256", {})).items():
        path = resolve_asset_file(local, str(relative))
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        match = exists and actual == str(expected).lower()
        passed = passed and match
        files.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix()
                if PROJECT_ROOT in path.parents
                else str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": match,
            }
        )
    return {"status": "verified" if passed else "missing_or_mismatch", "files": files}


def disk_guard(expected_bytes: int = 0) -> None:
    registry = load_encoder_environment_registry(PROJECT_ROOT)
    free = shutil.disk_usage(PROJECT_ROOT).free
    if free - max(0, expected_bytes) < registry.minimum_free_bytes:
        raise RuntimeError(
            f"projected free disk {free - expected_bytes} is below floor "
            f"{registry.minimum_free_bytes}"
        )


def acquire_huggingface(
    candidate: dict[str, Any],
    entry: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    checkpoint = candidate["checkpoint"]
    repo_id = checkpoint.get("repo_id")
    revision = checkpoint.get("revision")
    if not repo_id or not revision:
        raise RuntimeError("missing Hugging Face repo_id or revision")
    expected_size = int(checkpoint.get("expected_size_bytes") or 0)
    disk_guard(expected_size)
    final = PROJECT_ROOT / str(entry["local_path"])
    if final.exists():
        raise RuntimeError(f"refusing to overwrite existing asset path: {final}")
    temporary = cache_root / "downloads" / candidate["id"]
    temporary.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="snapshot-", dir=temporary))
    try:
        snapshot_download(
            repo_id=str(repo_id),
            revision=str(revision),
            local_dir=str(staging),
            allow_patterns=list(checkpoint.get("allow_patterns") or []),
            cache_dir=str(cache_root / "huggingface"),
            max_workers=1,
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        return {"status": "downloaded_pending_registry_hash", "path": str(final)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--id", action="append")
    parser.add_argument(
        "--output-root",
        default="outputs/environment-migration-v2",
    )
    args = parser.parse_args(argv)

    environment = load_encoder_environment_registry(PROJECT_ROOT)
    candidates = load_encoder_candidates(PROJECT_ROOT)
    selected = set(args.id or [item["id"] for item in candidates])
    checkpoint_data = yaml.safe_load(
        (PROJECT_ROOT / "registry/checkpoints.yaml").read_text(encoding="utf-8")
    )["checkpoints"]
    runtime_candidates = [
        item
        for item in candidates
        if item["registration_state"] != "candidate_only" and item["id"] in selected
    ]
    items = []
    manual = []
    for candidate in runtime_candidates:
        checkpoint_id = candidate["checkpoint"]["registry_id"]
        entry = checkpoint_data[checkpoint_id]
        verified = verify_entry(entry)
        if verified["status"] == "verified":
            items.append(
                {
                    "integration_id": candidate["id"],
                    "checkpoint_id": checkpoint_id,
                    "status": "verified_existing",
                    "files": verified["files"],
                }
            )
            continue
        reason = None
        acquired = None
        can_auto = (
            args.execute
            and candidate["license_state"] == "verified"
            and entry.get("source") == "huggingface"
        )
        if can_auto:
            try:
                acquired = acquire_huggingface(candidate, entry, environment.cache_root)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = (
                "automatic download disabled"
                if not args.execute
                else "manual or license-gated asset"
            )
        if acquired is not None:
            items.append(
                {
                    "integration_id": candidate["id"],
                    "checkpoint_id": checkpoint_id,
                    **acquired,
                }
            )
        else:
            record = {
                "integration_id": candidate["id"],
                "checkpoint_id": checkpoint_id,
                "status": "manual_required",
                "reason": reason,
                "official_repo": candidate["checkpoint"].get("repo_url"),
                "artifact_url": candidate["checkpoint"].get("artifact_url"),
                "revision": candidate["checkpoint"].get("revision"),
                "license": candidate["checkpoint"].get("license"),
                "allow_patterns": candidate["checkpoint"].get("allow_patterns") or [],
                "expected_size_bytes": candidate["checkpoint"].get("expected_size_bytes"),
                "incoming_path": (
                    PROJECT_ROOT / ".incoming/encoder-v2" / candidate["id"]
                ).as_posix(),
                "final_path": str(entry.get("local_path")),
            }
            manual.append(record)
            items.append(record)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    if PROJECT_ROOT not in output_root.parents:
        raise RuntimeError("output_root must remain under project root")
    output_root.mkdir(parents=True, exist_ok=True)
    asset_payload = {
        "schema_version": 2,
        "hostname": socket.gethostname(),
        "selected_count": len(runtime_candidates),
        "counts": {
            status: sum(item["status"] == status for item in items)
            for status in sorted({item["status"] for item in items})
        },
        "items": items,
    }
    (output_root / "asset-matrix.json").write_text(
        json.dumps(asset_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manual_payload = {
        "schema_version": 1,
        "manual_count": len(manual),
        "items": manual,
    }
    (output_root / "manual-download-manifest.json").write_text(
        json.dumps(manual_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asset_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
