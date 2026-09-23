# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.setitem(sys.modules, "tools.aisbench", SimpleNamespace(maybe_download_from_modelscope=lambda x: x))
    spec = importlib.util.spec_from_file_location(
        "tools._gate_bench_test", Path(__file__).resolve().parents[4] / "tools/vllm_bench.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_verification_and_collection_mode(bench, monkeypatch, tmp_path):
    commands = []

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            commands.append(command)
            result = Path(command[command.index("--result-filename") + 1])
            result.write_text(json.dumps({"output_throughput": 90}))

        def communicate(self, **kwargs):
            return "", ""

    monkeypatch.setattr(bench.subprocess, "Popen", Process)
    config = {"ignore_eos": True, "save_detailed": True, "unused_flag": False}
    with pytest.raises(AssertionError, match="Performance verification failed"):
        bench.VllmbenchRunner("model", 8000, config, 100, result_dir=str(tmp_path))
    assert commands[0][commands[0].index("--backend") + 1] == "openai-chat"
    collected = bench.VllmbenchRunner("model", 8000, config, 100, verify=False, result_dir=str(tmp_path))
    assert collected.result["output_throughput"] == 90
    assert config == {"ignore_eos": True, "save_detailed": True, "unused_flag": False}
    assert "--ignore-eos" in commands[0] and "--save-detailed" in commands[0]
    assert "" not in commands[0] and "--unused-flag" not in commands[0]
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_benchmark_timeout_terminates_its_process(bench):
    events = []

    class Process:
        def communicate(self, **kwargs):
            events.append("communicate")
            if events.count("communicate") <= 2:
                raise bench.subprocess.TimeoutExpired("vllm bench", 1)
            return "", ""

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

    runner = object.__new__(bench.VllmbenchRunner)
    runner.proc = Process()
    runner.timeout_seconds = 1
    with pytest.raises(bench.subprocess.TimeoutExpired):
        runner._wait_for_task()
    assert events == ["communicate", "terminate", "communicate", "kill", "communicate"]
