import importlib.util
from pathlib import Path
import sys


def test_compare_cache_keeps_the_workload_identical(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "benchmarks/throughput/compare_cache.py"
    spec = importlib.util.spec_from_file_location("compare_cache", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **_: commands.append(command))
    monkeypatch.setattr(sys, "argv", [
        str(path), "--model-path", "/models/qwen", "--num-concurrent-sessions", "1,8",
        "--num-sessions", "8", "--tensor-parallel-size", "2",
    ])
    module.main()
    assert len(commands) == 2
    assert commands[0][:commands[0].index("--cache-type")] == (
        commands[1][:commands[1].index("--cache-type")]
    )
    assert commands[0][commands[0].index("--cache-type") + 1] == "radix"
    assert commands[1][commands[1].index("--cache-type") + 1] == "logstate"
    assert "--history-mode" in commands[0] and "fixed" in commands[0]
    assert "--ignore-eos" in commands[0]
