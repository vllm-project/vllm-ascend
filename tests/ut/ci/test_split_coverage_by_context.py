# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for per-test coverage directory splitting."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from coverage.sqldata import CoverageData

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / ".github" / "workflows" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from split_coverage_by_context import (  # noqa: E402
    sanitize_target,
    split_covdata,
)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            "tests/e2e/pull_request/one_card/test_sampler.py",
            "tests__e2e__pull_request__one_card__test_sampler",
        ),
        (
            "tests/e2e/pull_request/one_card/test_sampler.py::test_foo",
            "tests__e2e__pull_request__one_card__test_sampler--test_foo",
        ),
        (
            "tests/e2e/pull_request/one_card/test_sampler.py::test_foo[param]",
            "tests__e2e__pull_request__one_card__test_sampler--test_foo[param]",
        ),
    ],
)
def test_sanitize_target(target: str, expected: str):
    assert sanitize_target(target) == expected


def test_split_covdata_writes_per_test_dirs(tmp_path: Path):
    rcfile = REPO_ROOT / "tests" / "coveragerc"
    staging = tmp_path / "tests__e2e__pull_request__one_card__test_sampler" / "covdata"
    staging.mkdir(parents=True)
    data_file = staging / "coverage"

    # Build a tiny coverage DB with two pytest contexts.
    src_mod = tmp_path / "vllm_ascend" / "dummy.py"
    src_mod.parent.mkdir(parents=True)
    src_mod.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    filename = str(src_mod.resolve())

    data = CoverageData(basename=str(data_file))
    data.set_context("tests/e2e/pull_request/one_card/test_sampler.py::test_a")
    data.add_lines({filename: {1, 2}})
    data.set_context("tests/e2e/pull_request/one_card/test_sampler.py::test_b")
    data.add_lines({filename: {1, 3}})
    data.write()

    outputs_root = tmp_path
    count = split_covdata(staging, outputs_root, rcfile)
    assert count == 2

    dir_a = outputs_root / "tests__e2e__pull_request__one_card__test_sampler--test_a"
    dir_b = outputs_root / "tests__e2e__pull_request__one_card__test_sampler--test_b"
    assert (dir_a / "covdata" / "coverage").is_file()
    assert (dir_b / "covdata" / "coverage").is_file()
    assert (dir_a / "covdata" / "context.txt").read_text(encoding="utf-8").strip().endswith(
        "::test_a"
    )

    # Staging file-level directory should be removed after split.
    assert not staging.parent.exists()

    cov_a = CoverageData(basename=str(dir_a / "covdata" / "coverage"))
    cov_a.read()
    assert set(cov_a.lines(filename) or []) == {1, 2}

    cov_b = CoverageData(basename=str(dir_b / "covdata" / "coverage"))
    cov_b.read()
    assert set(cov_b.lines(filename) or []) == {1, 3}


def test_split_keeps_dir_when_already_single_test(tmp_path: Path):
    rcfile = REPO_ROOT / "tests" / "coveragerc"
    nodeid = "tests/ut/ops/test_foo.py::test_only"
    key = sanitize_target(nodeid)
    staging = tmp_path / key / "covdata"
    staging.mkdir(parents=True)
    data_file = staging / "coverage"

    src_mod = tmp_path / "vllm_ascend" / "only.py"
    src_mod.parent.mkdir(parents=True)
    src_mod.write_text("x = 1\n", encoding="utf-8")
    filename = str(src_mod.resolve())

    data = CoverageData(basename=str(data_file))
    data.set_context(nodeid)
    data.add_lines({filename: {1}})
    data.write()

    count = split_covdata(staging, tmp_path, rcfile)
    assert count == 1
    assert (tmp_path / key / "covdata" / "coverage").is_file()
    assert (tmp_path / key).exists()
