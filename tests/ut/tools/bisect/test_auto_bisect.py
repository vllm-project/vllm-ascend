import argparse
from pathlib import Path

import pytest

from tools.bisect.auto_bisect import Bisector, _parse_args, _resolve_num_nodes
from tools.bisect.config import SCENE_MULTI


def test_pick_mid_prefers_midpoint_then_nearest_unskipped_index():
    assert Bisector._pick_mid(0, 8, skipped=set()) == 4
    assert Bisector._pick_mid(0, 8, skipped={4}) == 5
    assert Bisector._pick_mid(0, 8, skipped={4, 5}) == 3
    assert Bisector._pick_mid(0, 3, skipped={0, 1, 2}) is None


def test_parse_args_maps_no_assume_built_head_flag():
    args = _parse_args(
        [
            "--scene",
            "single_node",
            "--config-yaml",
            "case.yaml",
            "--good-commit",
            "good",
            "--no-assume-built-head",
            "--native-check",
            "since-build",
        ]
    )

    assert args.scene == "single_node"
    assert args.config_yaml == "case.yaml"
    assert args.good_commit == "good"
    assert args.no_assume_built_head is True
    assert args.native_check == "since-build"


def test_resolve_num_nodes_prefers_explicit_value(tmp_path: Path):
    args = argparse.Namespace(
        num_nodes=4,
        scene=SCENE_MULTI,
        config_base_path=None,
        config_yaml="missing.yaml",
    )

    assert _resolve_num_nodes(args, tmp_path) == 4


def test_resolve_num_nodes_reads_multi_node_yaml(tmp_path: Path):
    config = tmp_path / "configs" / "case.yaml"
    config.parent.mkdir()
    config.write_text("num_nodes: 2\n", encoding="utf-8")
    args = argparse.Namespace(
        num_nodes=None,
        scene=SCENE_MULTI,
        config_base_path="configs",
        config_yaml="case.yaml",
    )

    assert _resolve_num_nodes(args, tmp_path) == 2


def test_resolve_num_nodes_fails_when_multi_node_yaml_has_no_node_count(tmp_path: Path):
    config = tmp_path / "configs" / "case.yaml"
    config.parent.mkdir()
    config.write_text("test_cases: []\n", encoding="utf-8")
    args = argparse.Namespace(
        num_nodes=None,
        scene=SCENE_MULTI,
        config_base_path="configs",
        config_yaml="case.yaml",
    )

    with pytest.raises(SystemExit, match="Could not determine --num-nodes"):
        _resolve_num_nodes(args, tmp_path)
