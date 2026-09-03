from __future__ import annotations

import importlib.util
from pathlib import Path

from vadbench.environment_registry import (
    load_encoder_candidates,
    load_encoder_environment_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts/server" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_policy_skips_manual_candidate_and_license_gate() -> None:
    runner = load_script("run_native_encoder_matrix_v2.py")
    candidates = {item["id"]: item for item in load_encoder_candidates(ROOT)}
    assert runner.candidate_is_runnable(candidates["c3d"], False) == (
        False,
        "manual_asset_missing",
    )
    assert runner.candidate_is_runnable(candidates["videochat_online"], False) == (
        False,
        "license_blocked",
    )
    assert runner.candidate_is_runnable(candidates["videochat_online"], True) == (True, None)
    assert runner.candidate_is_runnable(candidates["videomae"], False) == (
        True,
        None,
    )


def test_asset_verifier_accepts_exact_hash_and_rejects_mismatch(tmp_path: Path) -> None:
    assets = load_script("fetch_encoder_assets_v2.py")
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"native-checkpoint")
    digest = assets.sha256_file(payload)
    exact = {
        "local_path": str(payload),
        "sha256": {"model.bin": digest},
    }
    wrong = {
        "local_path": str(payload),
        "sha256": {"model.bin": "0" * 64},
    }
    assert assets.verify_entry(exact)["status"] == "verified"
    assert assets.verify_entry(wrong)["status"] == "missing_or_mismatch"


def test_environment_manager_fingerprints_missing_prefix(tmp_path: Path) -> None:
    manager = load_script("manage_encoder_envs_v2.py")
    result = manager.fingerprint_environment(tmp_path / "missing")
    assert result == {
        "prefix": str((tmp_path / "missing").resolve()),
        "exists": False,
    }


def test_all_server_tools_use_v2_roots() -> None:
    registry = load_encoder_environment_registry(ROOT)
    assert registry.root == (ROOT / ".encoder-envs/v2").resolve()
    for name in (
        "manage_encoder_envs_v2.py",
        "fetch_encoder_assets_v2.py",
        "run_native_encoder_matrix_v2.py",
        "prepare_encoder_overlays_v2.py",
        "consolidate_encoder_v2_results.py",
    ):
        text = (ROOT / "scripts/server" / name).read_text(encoding="utf-8")
        assert (
            ".encoder-envs/v2" in text
            or "load_encoder_environment_registry" in text
            or "load_encoder_candidates" in text
        )


def test_consolidator_keeps_license_and_registration_gates(tmp_path: Path) -> None:
    consolidate = load_script("consolidate_encoder_v2_results.py")
    payload = consolidate.consolidate(tmp_path)
    assert payload["target_count"] == 25
    assert payload["counts"] == {
        "blocked_license": 2,
        "manual_required": 5,
        "not_run": 14,
        "unregistered": 4,
    }


def test_overlay_specs_include_observed_runtime_dependencies() -> None:
    overlays = load_script("prepare_encoder_overlays_v2.py")
    videomae_extra = overlays.COPY_SPECS["videomaev2"]["extra_sources"]
    assert any("easydict" in names for _, names in videomae_extra)
    assert "logzero" in overlays.COPY_SPECS["hermes_llava_ov"]["names"]
