import json
import os

import pytest

from experiment.launch_server import launch_logssms, launch_sglang
from experiment.random_multiturn import client, profile


def test_sglang_launcher_keeps_its_own_environment(tmp_path, monkeypatch):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    source = tmp_path / "sglang/python/sglang"
    source.mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", "/some/logssms/checkout/python")
    command, env = launch_sglang.make_command([
        "--sglang-python", str(python),
        "--sglang-root", str(tmp_path / "sglang"),
        "--model-path", "/models/model", "--tp-size", "2",
    ])
    assert command == [
        str(python), "-m", "sglang.launch_server",
        "--model-path", "/models/model", "--tp-size", "2",
    ]
    assert env["PYTHONPATH"] == str(tmp_path / "sglang/python")


def test_sglang_launcher_preserves_venv_python_symlink(tmp_path):
    base_python = tmp_path / "base/python"
    base_python.parent.mkdir()
    base_python.touch()
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    command, _ = launch_sglang.make_command([
        "--sglang-python", str(venv_python), "--model-path", "/models/model",
    ])
    assert command[0] == str(venv_python)


def test_logssms_launcher_forwards_server_arguments(monkeypatch):
    received = {}

    def fake_exec(path, argv, env):
        received.update(path=path, argv=argv, env=env)

    monkeypatch.setattr(os, "execvpe", fake_exec)
    launch_logssms.main(["--model-path", "/models/model", "--cache-type", "radix"])
    assert received["argv"][1:] == [
        "-m", "logssms", "--model-path", "/models/model",
        "--cache-type", "radix",
    ]
    assert received["env"]["PYTHONPATH"].split(os.pathsep)[0].endswith(
        "/LogSSMs/python"
    )


def test_client_rejects_server_flags_and_multi_point_runs():
    common = [
        "--api", "sglang", "--base-url", "http://127.0.0.1:30000",
        "--tokenizer", "/models/model",
    ]
    with pytest.raises(SystemExit):
        client._parse_args(common + ["--cache-type", "radix"])
    with pytest.raises(SystemExit):
        client._parse_args(common + ["--num-concurrent-sessions", "8,16"])


def test_cache_profile_can_be_attached_after_server_shutdown(tmp_path):
    run = tmp_path / "one-run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({
        "api": "logssms", "points": {"1": {"cache": None}},
    }))
    (run / "history-c1.json").write_text(json.dumps({"sessions": [{
        "turns": [
            {"tag": "s/turn-1", "expected_reusable_len": 0},
            {"tag": "s/turn-2", "expected_reusable_len": 10},
        ],
    }]}))
    profile_path = tmp_path / "profile.rank0.json"
    profile_path.write_text(json.dumps({"cache": {"match_samples": [
        {"tag": "s/turn-1", "tree_match_len": 0, "admitted_cached_len": 0},
        {"tag": "s/turn-2", "tree_match_len": 9, "admitted_cached_len": 7},
    ]}}))
    profile.main(["--run-dir", str(run), "--profile", str(profile_path)])
    result = json.loads((run / "report.json").read_text())
    assert result["points"]["1"]["cache"]["admitted_cached_tokens"] == 7
