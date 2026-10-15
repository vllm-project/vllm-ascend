# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER = REPO_ROOT / ".github" / "workflows" / "scripts" / "resolve_latest_release_branch.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("resolve_latest_release_branch", HELPER)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def test_filter_keeps_versioned_release_branches_only():
    helper = _load_helper()
    names = [
        "main",
        "releases/v0.30.0",
        "releases/v0.30.0rc2",
        "revert-13905-releases/v0.23.0_bugbix",
        "releases/not-a-version",
    ]
    assert helper.filter_release_branches(names) == [
        "releases/v0.30.0",
        "releases/v0.30.0rc2",
    ]


def test_select_latest_uses_creation_timestamp():
    helper = _load_helper()
    selected = helper.select_latest(
        {
            "releases/v0.30.0": "2026-09-29T10:01:04Z",
            "releases/v0.30.0rc2": "2026-10-13T08:57:19Z",
            "releases/v0.29.0rc": "2026-09-28T03:00:14Z",
        }
    )
    assert selected == "releases/v0.30.0rc2"


def test_select_latest_breaks_timestamp_ties_by_branch_name():
    helper = _load_helper()
    selected = helper.select_latest(
        {
            "releases/v0.30.0": "2026-10-13T08:57:19Z",
            "releases/v0.30.0rc2": "2026-10-13T08:57:19Z",
        }
    )
    assert selected == "releases/v0.30.0rc2"


def test_select_latest_rejects_empty_input():
    helper = _load_helper()
    with pytest.raises(ValueError, match="creation timestamp"):
        helper.select_latest({})


def test_created_at_from_search_reads_earliest_item():
    helper = _load_helper()
    assert helper.created_at_from_search({"items": []}) is None
    assert (
        helper.created_at_from_search({"items": [{"created_at": "2026-10-13T08:57:19Z", "number": 18571}]})
        == "2026-10-13T08:57:19Z"
    )


def test_release_coverage_uses_filename_suffix_in_the_same_directory():
    helper = _load_helper()
    assert helper.coverage_obs_prefix("main") == "ci/precision-test"
    assert helper.coverage_obs_prefix("releases/v0.30.0rc2") == "ci/precision-test"
    assert helper.coverage_file_suffix("main") == ""
    assert helper.coverage_file_suffix("releases/v0.30.0rc2") == "v0.30.0rc2"
    assert helper.coverage_object_name("coverage.tar", "") == "coverage.tar"
    assert helper.coverage_object_name("coverage.tar", "v0.30.0rc2") == "coverage-v0.30.0rc2.tar"
    assert helper.coverage_object_name("test_case_map.json", "v0.30.0rc2") == "test_case_map-v0.30.0rc2.json"
    assert helper.is_allowed_coverage_prefix("ci/precision-test")
    assert not helper.is_allowed_coverage_prefix("ci/precision-test/releases/v0.30.0rc2")
    with pytest.raises(ValueError, match="unsupported coverage branch"):
        helper.coverage_obs_prefix("feature/foo")
