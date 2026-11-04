# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the benchmark CLI and reports without a model or NPU."""

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOL = Path(__file__).resolve().parents[3] / "examples/swa_replay/bench.py"
FAKE = r"""
import json, os, sys
from pathlib import Path
if '--version' in sys.argv:
    print('0.30.0')
    sys.exit(0)
args = sys.argv[3:]
def value(flag):
    return args[args.index(flag) + 1]
assert value('--random-range-ratio') == '0'
assert value('--random-prefix-len') == '0'
assert '--ignore-eos' in args and '--save-detailed' in args
n = int(value('--num-prompts'))
length = int(value('--random-output-len'))
scale = float(os.environ.get('BENCH_TEST_SCALE', '1'))
result = {'completed': n, 'failed': 0, 'input_lens': [int(value('--random-input-len'))] * n,
          'output_lens': [length] * n, 'ttfts': [0.01] * n, 'itls': [[0.001] * (length - 1)] * n,
          'errors': [''] * n, 'output_throughput': 100 / scale, 'request_throughput': 10 / scale}
for metric in ('ttft', 'tpot', 'itl', 'e2el'):
    for stat in ('mean', 'p50', 'p95', 'p99'):
        result[f'{stat}_{metric}_ms'] = 10 * scale
if os.environ.get('BENCH_TEST_FAIL'):
    result['failed'], result['completed'] = 1, n - 1
Path(value('--result-dir'), value('--result-filename')).write_text(json.dumps(result))
"""


@pytest.fixture
def bench(tmp_path):
    native = tmp_path / "fake vllm"
    native.write_text(f"#!{sys.executable}\n" + FAKE)
    native.chmod(0o755)
    root = tmp_path / "results with spaces"

    def invoke(*args, expected=0, **env):
        result = subprocess.run(
            [sys.executable, str(TOOL), *map(str, args)],
            capture_output=True,
            text=True,
            env={**os.environ, **env},
            timeout=30,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result

    def run(mode, **env):
        return invoke(
            "run",
            "--mode",
            mode,
            "--vllm-bin",
            native,
            "--skip-server-info",
            "--tokenizer",
            "model <ROWS>& path",
            "--input-lens",
            256,
            512,
            "--concurrencies",
            1,
            2,
            "--num-prompts",
            2,
            "--output-len",
            4,
            "--rounds",
            2,
            "--warmups",
            1,
            "--output-dir",
            root,
            **env,
        )

    return root, invoke, run


def test_off_on_report_and_native_parameters(bench):
    root, invoke, run = bench
    run("off")
    run("on", BENCH_TEST_SCALE="0.8")
    invoke("compare", "--results-dir", root)
    comparison = json.loads((root / "comparison/comparison.json").read_text())
    ttft = next(row for row in comparison["rows"] if row["metric"] == "p50_ttft_ms")
    assert ttft["rounds"] == 2
    assert ttft["off_median"] == 10 and ttft["on_median"] == 8
    assert ttft["change_percent"] == pytest.approx(-20)
    report = (root / "comparison/report.html").read_text()
    assert "&lt;ROWS&gt;&amp;" in report and "model <ROWS>" not in report
    off = Path(json.loads((root / "latest-off.json").read_text())["directory"])
    manifest = json.loads((off / "manifest.json").read_text())
    assert len(manifest["cases"]) == 8
    command = json.loads((off / "i256-c2-r2.command.json").read_text())
    assert command[command.index("--seed") + 1] == "43"
    assert (root / "comparison/comparison.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_failed_requests_preserve_data_without_updating_latest(bench):
    root, _, run = bench
    result = run("off", expected=1, BENCH_TEST_FAIL="1")
    assert "Incomplete benchmark" in result.stderr
    assert not (root / "latest-off.json").exists()
    folder = next(root.glob("off-*"))
    assert json.loads((folder / "manifest.json").read_text())["status"] == "failed"
    assert list(folder.glob("*.log")) and list(folder.glob("i*.json"))


@pytest.mark.parametrize("mutation", ["workload", "lengths", "missing_case"])
def test_compare_rejects_unfair_or_incomplete_runs(bench, mutation):
    root, invoke, run = bench
    run("off")
    run("on")
    on = Path(json.loads((root / "latest-on.json").read_text())["directory"])
    manifest = json.loads((on / "manifest.json").read_text())
    if mutation == "workload":
        manifest["workload"]["seed"] = 99
    elif mutation == "missing_case":
        manifest["cases"].pop()
    else:
        result = on / manifest["cases"][0]["result"]
        data = json.loads(result.read_text())
        data["input_lens"][0] += 1
        result.write_text(json.dumps(data))
    (on / "manifest.json").write_text(json.dumps(manifest))
    invoke("compare", "--results-dir", root, expected=1)
    assert not (root / "comparison/report.html").exists()


@pytest.mark.parametrize("invalid", [None, "wrong_mode", "prefix_cache", "max_len"])
def test_server_config_checks_and_filtered_provenance(monkeypatch, invalid):
    spec = importlib.util.spec_from_file_location("swa_bench_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        skip_server_info=False,
        base_url="http://localhost",
        mode="off",
        allow_prefix_caching=False,
        input_lens=[256],
        output_len=128,
    )
    config = {
        "cache_config": {"swa_bounded_replay": False, "enable_prefix_caching": False},
        "model_config": {"max_model_len": 4096, "hf_token": "DO_NOT_PERSIST"},
        "parallel_config": {"tensor_parallel_size": 4},
    }
    if invalid == "wrong_mode":
        args.mode = "on"
    elif invalid == "prefix_cache":
        config["cache_config"]["enable_prefix_caching"] = True
    elif invalid == "max_len":
        config["model_config"]["max_model_len"] = 256
    response = {
        "vllm_config": config,
        "system_env": {"credential": "DO_NOT_PERSIST"},
        "vllm_env": {"VLLM_USE_V2_MODEL_RUNNER": "0", "credential": "DO_NOT_PERSIST"},
    }
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.StringIO(json.dumps(response)))
    if invalid:
        with pytest.raises(ValueError):
            module.server_info(args)
    else:
        info, signature = module.server_info(args)
        assert "DO_NOT_PERSIST" not in json.dumps(info)
        assert signature["VLLM_USE_V2_MODEL_RUNNER"] == "0"
        assert signature["parallel_config"]["tensor_parallel_size"] == 4
