import argparse
from pathlib import Path

import pytest

from tools.bisect.auto_bisect import Bisector, _parse_args, _resolve_num_nodes
from tools.bisect.config import SCENE_MULTI, BisectInput, BisectOptions


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
            "--soc",
            "a2",
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


def test_parse_args_requires_soc():
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--scene",
                "single_node",
                "--config-yaml",
                "case.yaml",
            ]
        )


def test_parse_args_rejects_empty_or_placeholder_soc():
    for bad_soc in ("", "unknown", "none", "null"):
        with pytest.raises(SystemExit):
            _parse_args(
                [
                    "--scene",
                    "single_node",
                    "--config-yaml",
                    "case.yaml",
                    "--soc",
                    bad_soc,
                ]
            )


def _bisector_with_good_table(
    table_path: str,
    *,
    name: str | None = None,
    soc: str | None = None,
    good_commit: str | None = None,
) -> Bisector:
    bisector = Bisector.__new__(Bisector)
    bisector.inp = BisectInput(
        scene="single_node",
        config_yaml="model.yaml",
        bad_commit="bad",
        name=name,
        soc=soc,
        good_commit=good_commit,
    )
    bisector.opt = BisectOptions(good_table_path=table_path)
    return bisector


def test_resolve_good_selects_row_matching_soc_and_records_paired_vllm(tmp_path: Path):
    table = tmp_path / "good_table.csv"
    table.write_text(
        "name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time\n"
        "shared,cases/model.yaml,a2,success,vllm-a2,asc-a2,a2,single_node,2026-01-01 01:00:00 +0800\n"
        "shared,cases/model.yaml,a3,success,vllm-a3,asc-a3,a3,single_node,2026-01-02 01:00:00 +0800\n",
        encoding="utf-8",
    )

    bisector = _bisector_with_good_table(str(table), name="shared", soc="a3")

    good = bisector._resolve_good()

    assert good == "asc-a3"
    assert bisector.inp.good_vllm_commit == "vllm-a3"


def test_resolve_good_explicit_commit_skips_table_lookup(tmp_path: Path):
    bisector = _bisector_with_good_table(
        str(tmp_path / "missing.csv"),
        name="shared",
        soc="a2",
        good_commit="explicit-good",
    )

    assert bisector._resolve_good() == "explicit-good"


def test_resolve_good_raises_when_no_matching_success_row(tmp_path: Path):
    table = tmp_path / "good_table.csv"
    table.write_text(
        "name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time\n"
        "shared,cases/model.yaml,a3,success,vllm-a3,asc-a3,a3,single_node,2026-01-02 01:00:00 +0800\n",
        encoding="utf-8",
    )

    bisector = _bisector_with_good_table(str(table), name="shared", soc="a2")

    with pytest.raises(SystemExit, match="No successful good-table row"):
        bisector._resolve_good()


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


def test_resolve_num_nodes_rejects_non_mapping_yaml(tmp_path: Path):
    """A list/scalar YAML would make ``data.get`` raise AttributeError instead of a clear error."""
    config = tmp_path / "configs" / "case.yaml"
    config.parent.mkdir()
    config.write_text("- num_nodes: 2\n", encoding="utf-8")
    args = argparse.Namespace(
        num_nodes=None,
        scene=SCENE_MULTI,
        config_base_path="configs",
        config_yaml="case.yaml",
    )

    with pytest.raises(SystemExit, match="must contain a mapping"):
        _resolve_num_nodes(args, tmp_path)


@pytest.mark.parametrize("value", [0, -1, -2])
def test_resolve_num_nodes_rejects_non_positive_explicit_value(tmp_path: Path, value: int):
    """The ready barrier is ``len(readies) >= num_nodes``: at zero or below it passes
    with no workers at all, so the master starts a distributed run that cannot work."""
    args = argparse.Namespace(
        num_nodes=value,
        scene=SCENE_MULTI,
        config_base_path=None,
        config_yaml="missing.yaml",
    )

    with pytest.raises(SystemExit, match=r"--num-nodes must be >= 1"):
        _resolve_num_nodes(args, tmp_path)


def test_resolve_num_nodes_treats_yaml_zero_as_unset(tmp_path: Path):
    """``num_nodes: 0`` means "no node count here", so it falls through to the
    same error as a missing key rather than being used as a node count."""
    config = tmp_path / "configs" / "case.yaml"
    config.parent.mkdir()
    config.write_text("num_nodes: 0\n", encoding="utf-8")
    args = argparse.Namespace(
        num_nodes=None,
        scene=SCENE_MULTI,
        config_base_path="configs",
        config_yaml="case.yaml",
    )

    with pytest.raises(SystemExit, match="Could not determine --num-nodes"):
        _resolve_num_nodes(args, tmp_path)


@pytest.mark.parametrize("value", [-1, -2])
def test_resolve_num_nodes_rejects_negative_yaml_value(tmp_path: Path, value: int):
    """A negative count is truthy, so it used to be returned and then made the
    ``len(readies) >= num_nodes`` barrier pass with no workers."""
    config = tmp_path / "configs" / "case.yaml"
    config.parent.mkdir()
    config.write_text(f"num_nodes: {value}\n", encoding="utf-8")
    args = argparse.Namespace(
        num_nodes=None,
        scene=SCENE_MULTI,
        config_base_path="configs",
        config_yaml="case.yaml",
    )

    with pytest.raises(SystemExit, match=r"must be >= 1"):
        _resolve_num_nodes(args, tmp_path)


def test_resolve_num_nodes_reports_non_integer_yaml_value(tmp_path: Path):
    """``int("abc")`` used to escape as an uncaught ValueError traceback."""
    config = tmp_path / "configs" / "case.yaml"
    config.parent.mkdir()
    config.write_text("num_nodes: abc\n", encoding="utf-8")
    args = argparse.Namespace(
        num_nodes=None,
        scene=SCENE_MULTI,
        config_base_path="configs",
        config_yaml="case.yaml",
    )

    with pytest.raises(SystemExit, match="is not an integer"):
        _resolve_num_nodes(args, tmp_path)
