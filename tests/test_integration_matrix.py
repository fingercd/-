from __future__ import annotations

import json
from pathlib import Path

import pytest

from vadbench.engine.integration_matrix import (
    filter_integrations,
    preflight_integrations,
    run_integration_matrix,
    write_matrix_result,
)
from vadbench.integrations import DEFAULT_INTEGRATION_CATALOG


def test_filter_and_preflight_support_id_runtime_and_mode() -> None:
    selected = filter_integrations(
        DEFAULT_INTEGRATION_CATALOG,
        integration_ids=["videomaev2", "hermes_llava_ov"],
    )
    assert [item.id for item in selected] == ["videomaev2", "hermes_llava_ov"]
    assert len(preflight_integrations(DEFAULT_INTEGRATION_CATALOG, run_modes=["streaming"])) == 5


def test_matrix_uses_runtime_hooks_and_continues_after_failure(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(record):
        calls.append(record.id)
        if record.id == "videomaev2":
            raise RuntimeError("synthetic failure")
        return {"status": "smoke_pass", "encoder": {"id": record.id}}

    result = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {"schema_version": 1},
        tmp_path / "video.mp4",
        project_root=tmp_path,
        integration_ids=["videomaev2", "hermes_llava_ov"],
        skip_preflight=True,
        run_one=runner,
        output_root=tmp_path / "matrix",
    )
    assert calls == ["videomaev2", "hermes_llava_ov"]
    assert [item["status"] for item in result["items"]] == ["failed", "smoke_pass"]
    assert (tmp_path / "matrix" / "matrix.json").is_file()
    assert (tmp_path / "matrix" / "hermes_llava_ov" / "result.json").is_file()


def test_matrix_preserves_existing_success_and_rejects_escape(tmp_path: Path) -> None:
    output_root = tmp_path / "matrix"
    output_root.mkdir()
    existing = output_root / "videomaev2" / "result.json"
    existing.parent.mkdir()
    existing.write_text(json.dumps({"status": "smoke_pass", "sentinel": 1}), encoding="utf-8")
    calls: list[str] = []

    def runner(record):
        calls.append(record.id)
        return {"status": "smoke_pass", "sentinel": 2}

    result = run_integration_matrix(
        DEFAULT_INTEGRATION_CATALOG,
        {},
        None,
        project_root=tmp_path,
        integration_ids=["videomaev2"],
        skip_preflight=True,
        run_one=runner,
        output_root=output_root,
    )
    assert calls == []
    assert result["items"][0]["reused"] is True
    assert json.loads(existing.read_text())["sentinel"] == 1
    with pytest.raises(ValueError):
        write_matrix_result(
            {"status": "completed"}, tmp_path / "escape.json", output_root=output_root
        )
