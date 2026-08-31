from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读核验25路视频模型 catalog 中声明的本地 checkpoint；不联网、不删除。"
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--id", dest="integration_ids", action="append")
    parser.add_argument("--output")
    return parser


def _asset_file(root: Path, local_path: str, relative: str) -> Path:
    selected = Path(local_path).expanduser()
    if not selected.is_absolute():
        selected = root / selected
    selected = selected.resolve()
    if selected.is_file():
        if relative not in {selected.name, "model.pth", "model.pyth", "model.safetensors"}:
            return selected.parent / relative
        return selected
    return selected / relative


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inferred_root = Path(__file__).resolve().parents[2]
    root = (
        inferred_root
        if args.project_root is None
        else Path(args.project_root).expanduser().resolve()
    )
    sys.path.insert(0, str(root / "src"))
    from vadbench.integrations.catalog import load_default_integration_catalog

    catalog = load_default_integration_catalog(root)
    registry_path = root / "registry" / "checkpoints.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))["checkpoints"]
    selected = set(args.integration_ids or catalog.ids)
    unknown = selected - set(catalog.ids)
    if unknown:
        raise SystemExit(f"catalog 中不存在 integration：{sorted(unknown)}")

    items: list[dict[str, Any]] = []
    for record in catalog.integrations:
        if record.id not in selected:
            continue
        entry = registry[record.checkpoint.registry_id]
        local_path = str(entry.get("local_path", ""))
        expected_hashes = dict(entry.get("sha256", {}))
        file_results: list[dict[str, Any]] = []
        passed = bool(local_path and expected_hashes)
        for relative, expected in expected_hashes.items():
            path = _asset_file(root, local_path, str(relative))
            result: dict[str, Any] = {
                "path": path.relative_to(root).as_posix()
                if root in path.parents
                else path.as_posix(),
                "exists": path.is_file(),
                "expected_sha256": str(expected),
                "actual_sha256": None,
                "size_bytes": None,
                "match": False,
            }
            if path.is_file():
                actual = _sha256(path)
                result["actual_sha256"] = actual
                result["size_bytes"] = int(path.stat().st_size)
                result["match"] = actual == str(expected).lower()
            passed = passed and bool(result["match"])
            file_results.append(result)
        items.append(
            {
                "integration_id": record.id,
                "checkpoint_id": record.checkpoint.registry_id,
                "registry_status": entry.get("status"),
                "validation_scope": entry.get("validation_scope", "native"),
                "local_path": local_path or None,
                "status": "verified" if passed else "missing_or_mismatch",
                "files": file_results,
            }
        )
    payload = {
        "schema_version": "vadbench.encoder-assets-preflight.v1",
        "project_root": str(root),
        "selected_count": len(items),
        "verified_count": sum(item["status"] == "verified" for item in items),
        "items": items,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = root / output
        output = output.resolve()
        if root not in output.parents:
            raise SystemExit(f"output 必须位于 project_root 内：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if payload["verified_count"] == payload["selected_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
