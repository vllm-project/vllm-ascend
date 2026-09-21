# SPDX-License-Identifier: Apache-2.0
"""CLI tests: inventory/profiles/sources/plan/build/verify/compare entry points."""

import json
from pathlib import Path

from tools.glm_reduced.cli import main

from .conftest import make_dsa_checkpoint


def test_inventory_cli(capsys):
    assert main(["inventory"]) == 0
    out = capsys.readouterr().out
    assert "GLM-5.2" in out and "glm-moe-dsa" in out and "unsupported" in out
    assert "GLM-4.1V" in out and "chatglm3-6b" in out
    assert main(["inventory", "--format", "json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert any(e["model_id"] == "zai-org/GLM-5.3-Flash" for e in entries)
    assert any(e["model_id"] == "zai-org/GLM-Image" and e["status"] == "unsupported" for e in entries)


def test_profiles_cli(capsys):
    assert main(["profiles"]) == 0
    out = capsys.readouterr().out
    assert "glm5-next" in out and "glm4-moe" in out and "chatglm" in out and "glm4v" in out


def test_sources_cli(capsys):
    assert main(["sources", "--model", "zai-org/GLM-5.2"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert entries[0]["revision"]
    assert entries[0]["config_url"].startswith("https://huggingface.co/")
    assert main(["sources", "--model", "nonexistent/model"]) == 2


def test_required_shards_cli(tmp_path, capsys):
    src = make_dsa_checkpoint(tmp_path / "src")
    assert (
        main(
            [
                "required-shards",
                "--config",
                str(src / "config.json"),
                "--index",
                str(src / "model.safetensors.index.json"),
                "--profile",
                "glm-moe-dsa",
                "--layers",
                "8",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["shard_count"] >= 1
    assert result["source_layers"] == 12
    assert set(result["shards"]) <= {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_plan_build_verify_cli(tmp_path, capsys):
    src = make_dsa_checkpoint(tmp_path / "src")
    dst = tmp_path / "out"

    assert main(["plan", str(src)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["profile"] == "glm-moe-dsa"  # auto-matched from architectures
    assert summary["keep_layers"] == 8
    assert summary["actions"]["remap"] > 0  # MTP tensors
    assert summary["actions"]["drop-layer"] > 0

    assert main(["build", str(src), str(dst), "--layers", "8"]) == 0
    assert (dst / "reduction_manifest.json").is_file()
    capsys.readouterr()  # flush build output
    assert main(["verify", str(dst)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["keep_layers"] == 8


def test_build_cli_error_exit_code(tmp_path, capsys):
    src = make_dsa_checkpoint(tmp_path / "src")
    dst = tmp_path / "out"
    assert main(["build", str(src), str(dst), "--layers", "3"]) == 2  # below profile minimum
    assert "keep_layers >= 8" in capsys.readouterr().err
    assert not dst.exists()


def _write_dump(path: Path, mode: str, logprob: float, checkpoint_id: str = "sha256:x", run_id: str = "r1") -> None:
    meta = {
        "format": "glm-reduced-logprobs-v2",
        "model": "/m",
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "dtype": "bfloat16",
        "mode": mode,
        "seed": 0,
        "prompt_count": 1,
        "output_tokens": 1,
        "runtime": {"vllm": "0.13.0"},
        "engine": {"enforce_eager": mode == "eager"},
    }
    record = {
        "prompt_index": 0,
        "prompt_token_ids": [1],
        "token_ids": [5],
        "top_logprobs": [[{"token_id": 5, "logprob": logprob}]],
    }
    path.write_text(json.dumps(meta) + "\n" + json.dumps(record) + "\n", encoding="utf-8")


def test_compare_logits_cli(tmp_path):
    baseline = tmp_path / "b.jsonl"
    candidate = tmp_path / "c.jsonl"
    _write_dump(baseline, "eager", -0.5, run_id="r1")
    _write_dump(candidate, "graph", -0.5001, run_id="r2")
    assert main(["compare-logits", str(baseline), str(candidate), "--atol", "1e-3", "--rtol", "1e-3"]) == 0
    assert main(["compare-logits", str(baseline), str(candidate), "--atol", "1e-9", "--rtol", "1e-9"]) == 1
    # Same file on both sides is the same run.
    assert main(["compare-logits", str(baseline), str(baseline), "--atol", "1", "--rtol", "1"]) == 2


def test_compare_perf_cli(tmp_path):
    def write(path: Path, runtime: str, latency: float, checkpoint_id: str = "sha256:x", run_id: str = "r1") -> None:
        payload = {
            "format": "glm-reduced-perf-v2",
            "meta": {
                "model": "/m-reduced",
                "checkpoint_id": checkpoint_id,
                "run_id": run_id,
                "hardware": "hw",
                "workload_id": "w",
                "mode": "run",
                "warmup_iterations": 1,
                "measured_iterations": 2,
                "expected_iteration_tokens": 100,
                "runtime": {"vllm": runtime},
                "engine": {"tensor_parallel_size": 8},
            },
            "latencies_s": [latency, latency],
            "total_tokens": [100, 100],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    baseline = tmp_path / "b.json"
    candidate = tmp_path / "c.json"
    write(baseline, "0.13.0", 1.0, run_id="r1")
    write(candidate, "0.14.0", 0.5, run_id="r2")
    assert main(["compare-perf", str(baseline), str(candidate), "--max-latency-regression-pct", "50"]) == 0
    # Fail closed without thresholds.
    assert main(["compare-perf", str(baseline), str(candidate)]) == 1
    # Improvement beyond the allowed "regression" floor still passes.
    assert main(["compare-perf", str(baseline), str(candidate), "--min-throughput-change-pct", "0"]) == 0
