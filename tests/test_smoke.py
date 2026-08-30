from __future__ import annotations

import json
from pathlib import Path

import yaml

import vadbench.cli as cli
from vadbench.smoke import write_smoke_result


def test_write_smoke_result_is_utf8_json(tmp_path: Path) -> None:
    output = write_smoke_result({"adapter": "测试", "ok": True}, tmp_path / "smoke.json")
    assert '"adapter": "测试"' in output.read_text(encoding="utf-8")


def test_cli_smoke_writes_selected_output(tmp_path: Path, monkeypatch, capsys) -> None:
    config = {
        "schema_version": 1,
        "dataset": {"root": str(tmp_path)},
        "encoder": {"adapter": "fake"},
        "streaming": {"enabled": False},
        "task": {"supervision": "video"},
        "output": {"root": str(tmp_path / "outputs"), "run_name": "smoke"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    captured = {}

    def fake_smoke(config, *args, **kwargs):
        captured["config"] = config
        return {"adapter": "fake", "mode": "fixed", "ok": True}

    monkeypatch.setattr(cli, "run_encoder_smoke", fake_smoke)
    output = tmp_path / "result.json"
    assert (
        cli.main(
            [
                "smoke",
                "-c",
                str(config_path),
                "--video",
                str(tmp_path / "video.mp4"),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--kv-size",
                "64",
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True
    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    assert captured["config"]["encoder"]["params"] == {"device": "cpu", "kv_size": 64}
