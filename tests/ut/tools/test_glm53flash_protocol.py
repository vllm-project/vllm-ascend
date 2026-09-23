# SPDX-License-Identifier: Apache-2.0
"""Scenario regressions and real-file replay through the offline gates."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from tools.ci.glm53flash_analyze import analyze
from tools.ci.glm53flash_launch import effective_settings
from tools.ci.glm53flash_make_policy import candidate_policy
from tools.ci.glm53flash_protocol import precision_lengths, validate_precision_lengths
from tools.ci.run_glm53flash_gates import evaluate


@pytest.mark.parametrize("block_size,count", [(128, 16), (512, 16), (1152, 19)])
def test_overlapping_boundaries(block_size, count):
    lengths = precision_lengths(block_size)
    assert len(lengths) == count
    assert lengths == sorted(set(lengths))
    assert {block_size - 1, block_size, block_size + 1} <= set(lengths)
    validate_precision_lengths(lengths, block_size)


@pytest.mark.parametrize("block_size", [None, True, 0, 1, -1, 128.0, "128"])
def test_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="block_size"):
        precision_lengths(block_size)


@pytest.mark.parametrize(
    "lengths",
    [
        [128],
        list(range(1, 20)),
        precision_lengths(128)[:-1],
        precision_lengths(128) + [2177],
        precision_lengths(128) + [3000],
        list(reversed(precision_lengths(128))),
        [float(n) for n in precision_lengths(128)],
        precision_lengths(1152),
    ],
)
def test_wrong_scenarios_rejected(lengths):
    with pytest.raises(ValueError, match="Full boundary collection"):
        validate_precision_lengths(lengths, 128)


def test_collector_uses_shared_protocol():
    # Inspect without importing torch/vllm on the CPU-only test host.
    path = Path(__file__).resolve().parents[3] / "tools/ci/glm53flash_collect.py"
    tree = ast.parse(path.read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tools.ci.glm53flash_protocol"
        and any(alias.name == "precision_lengths" for alias in node.names)
        for node in ast.walk(tree)
    )
    selection = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "lengths" for target in node.targets)
    )
    assert isinstance(selection, ast.IfExp)
    assert isinstance(selection.orelse, ast.Call)
    assert selection.orelse.func.id == "precision_lengths"
    assert selection.orelse.args[0].id == "block_size"


def write_collection(root, block_size):
    """Write collector-format artifacts; model outputs are CPU test fixtures."""
    settings, _, _ = effective_settings("text_tp4")
    settings["enforce_eager"] = False
    logits = np.zeros((8, 154880), dtype=np.float32)
    logits[:, 42] = 1.0
    lengths = precision_lengths(block_size)
    for run in ("graph-0", "graph-1", "graph-2", "eager-0"):
        folder = root / run
        folder.mkdir(parents=True)
        graph = run.startswith("graph")
        current_settings = dict(settings)
        if not graph:
            current_settings["enforce_eager"] = True
            current_settings.pop("compilation_config")
        records = {
            "result.json": {"status": "PASS", "lengths": lengths, "replays": [7 if graph else 0] * 4},
            "settings.json": current_settings,
            "runtime-settings.json": {"block_size": block_size},
            "path-evidence.json": [{"layers": 9}] * 4,
            "weights.json": [{"rank": rank, "sha256": "a" * 64} for rank in range(4)],
        }
        if graph:
            records["performance.json"] = [
                {
                    "iteration": i,
                    "wall_ms": 1000,
                    "output_tokens_s": 256,
                    "requests": [{"preemptions": 0, "ttft_ms": 2, "tpot_ms": 3}] * 4,
                }
                for i in range(15)
            ]
        for name, record in records.items():
            (folder / name).write_text(json.dumps(record))
        for length in lengths:
            np.save(folder / f"n{length}-cold.npy", logits, allow_pickle=False)


@pytest.mark.parametrize("block_size", [128, 1152])
def test_real_artifact_replay(tmp_path, block_size):
    baseline, current = tmp_path / "baseline", tmp_path / "current"
    write_collection(baseline, block_size)
    write_collection(current, block_size)
    summary = analyze(baseline)
    assert summary["performance"]["samples"] == 45
    policy = candidate_policy(summary, "cpu-fixture-only")
    environment = {"scope": "cpu-fixture-only", "validated": True, "exclusive": True, "checked_by": "unit-test"}
    result = evaluate(baseline, current, policy, environment=environment)
    assert result["status"] == "PASS"
    assert len(result["precision"]["cases"]) == 3 * len(precision_lengths(block_size))
    # A malformed current collection must fail even when it has >=19 entries.
    path = current / "graph-0/result.json"
    record = json.loads(path.read_text())
    record["lengths"] = list(range(1, 20))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Full boundary collection"):
        evaluate(baseline, current, policy, environment=environment)
