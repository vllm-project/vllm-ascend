# SPDX-License-Identifier: Apache-2.0
"""CLI contracts using the frozen real ten-pair acceptance data."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CLI = ROOT / "benchmarks/check_mooncake_store_reuse.py"
DATA = ROOT / "benchmarks/data/mooncake_store_reuse_qwen38_27b.json"
PAIR_COUNT = 10


def run_cli(input_path: Path, expected_passed: bool):
    result = subprocess.run(
        [sys.executable, str(CLI), str(input_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == (0 if expected_passed else 1), (result.stdout, result.stderr)
    report = json.loads(result.stdout)
    assert report["passed"] is expected_passed
    assert isinstance(report["checks"], dict)
    assert report["checks"]


def test_frozen_real_ten_pairs_pass():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    assert len(data["pairs"]) == PAIR_COUNT
    run_cli(DATA, expected_passed=True)


def test_slower_b_restore_ttft_fails(tmp_path: Path):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for pair in data["pairs"]:
        pair["B"]["restore"]["ttft_seconds"] = pair["A"]["restore"]["ttft_seconds"] * 2
    modified = tmp_path / "slower-restore.json"
    modified.write_text(json.dumps(data), encoding="utf-8")
    run_cli(modified, expected_passed=False)


def test_b_local_cache_cannot_count_as_external_reuse(tmp_path: Path):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for pair in data["pairs"]:
        evidence = pair["B"]["restore"]["evidence"]
        evidence["local"] = evidence["external"]
        evidence["external"] = 0
        evidence["get_calls"] = 0
        evidence["get_bytes"] = 0
    modified = tmp_path / "local-cache-only.json"
    modified.write_text(json.dumps(data), encoding="utf-8")
    run_cli(modified, expected_passed=False)
