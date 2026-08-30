"""显式、可校验的模型权重注册与下载。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


class CheckpointError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointSpec:
    id: str
    adapter: str
    source: str
    repo_id: str
    revision: str
    license: str
    allow_patterns: tuple[str, ...]
    sha256: Mapping[str, str]
    notes: str = ""

    @classmethod
    def from_mapping(cls, checkpoint_id: str, data: Mapping[str, Any]) -> CheckpointSpec:
        required = ("adapter", "source", "repo_id", "revision", "license")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise CheckpointError(f"checkpoint {checkpoint_id!r} 缺少字段：{missing}")
        return cls(
            id=checkpoint_id,
            adapter=str(data["adapter"]),
            source=str(data["source"]),
            repo_id=str(data["repo_id"]),
            revision=str(data["revision"]),
            license=str(data["license"]),
            allow_patterns=tuple(str(item) for item in data.get("allow_patterns", ())),
            sha256={str(k): str(v).lower() for k, v in dict(data.get("sha256", {})).items()},
            notes=str(data.get("notes", "")),
        )


def load_checkpoint_registry(path: str | Path) -> dict[str, CheckpointSpec]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    entries = data.get("checkpoints", data)
    if not isinstance(entries, Mapping):
        raise CheckpointError("checkpoint registry 顶层必须包含 checkpoints 对象")
    return {
        str(checkpoint_id): CheckpointSpec.from_mapping(str(checkpoint_id), value)
        for checkpoint_id, value in entries.items()
    }


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(spec: CheckpointSpec, root: str | Path) -> dict[str, str]:
    checkpoint_root = Path(root)
    if not checkpoint_root.is_dir():
        raise CheckpointError(f"权重目录不存在：{checkpoint_root}")
    actual: dict[str, str] = {}
    errors: list[str] = []
    for relative, expected in spec.sha256.items():
        file_path = checkpoint_root / relative
        if not file_path.is_file():
            errors.append(f"缺少 {relative}")
            continue
        actual_digest = sha256_file(file_path)
        actual[relative] = actual_digest
        if actual_digest.lower() != expected.lower():
            errors.append(f"{relative}: expected={expected}, actual={actual_digest}")
    if errors:
        raise CheckpointError("权重校验失败：\n- " + "\n- ".join(errors))
    return actual


def fetch_checkpoint(
    spec: CheckpointSpec,
    destination: str | Path,
    *,
    accepted_license: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """下载冻结 revision；许可证必须被调用方显式确认。"""

    if accepted_license != spec.license:
        raise CheckpointError(
            f"下载 {spec.id!r} 前必须显式确认许可证：accepted_license={spec.license!r}"
        )
    if spec.source != "huggingface":
        raise CheckpointError(f"暂不支持 checkpoint source={spec.source!r}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise CheckpointError("缺少 huggingface_hub；请安装 vadbench[videomaev2]") from exc

    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=str(destination_path),
        allow_patterns=list(spec.allow_patterns) or None,
        local_files_only=local_files_only,
    )
    actual = verify_checkpoint(spec, destination_path)
    provenance = {
        "schema": "vadbench.checkpoint/v1",
        "spec": asdict(spec),
        "verified_sha256": actual,
    }
    (destination_path / "vadbench-checkpoint.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination_path
