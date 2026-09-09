import argparse
from pathlib import Path

import pytest

from tools.bisect import git_ops
from tools.bisect.auto_bisect import Bisector, _parse_args, _resolve_num_nodes, main
from tools.bisect.config import SCENE_MULTI, BisectInput, BisectOptions, Candidate
from tools.bisect.runner import BisectFatalError
from tools.bisect.version_compat import PackageVersions


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


def test_run_trial_maps_deploy_errors_to_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A git/build failure while deploying one candidate must become a SKIP
    for that commit, not crash the whole search."""
    monkeypatch.setattr("tools.bisect.runner.kill_stray_servers", lambda: None)
    inp = BisectInput(scene="single_node", config_yaml="case.yaml", bad_commit="bad", soc="a2", good_commit="a" * 40)
    opt = BisectOptions(repo_dir=tmp_path, work_dir=str(tmp_path / "work"), assume_built_head=False)
    bisector = Bisector(inp, opt)

    def boom(candidate, round_idx, log_dir):
        raise git_ops.GitError("git checkout failed")

    monkeypatch.setattr(bisector.runner, "validate", boom)

    result = bisector._run_trial(Candidate(commit="b" * 40, pr_number=None, subject="bad"))

    assert result.verdict == "SKIP"
    assert result.note.startswith("build failed -> SKIP")


def test_run_writes_report_when_aborting_on_fatal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The fatal-abort path (e.g. no multi-node worker ever joined) must still
    write report.json so the run is auditable, and re-raise to main()."""
    monkeypatch.setattr("tools.bisect.runner.kill_stray_servers", lambda: None)
    monkeypatch.setattr(git_ops, "describe", lambda repo, ref: Candidate(commit=ref, pr_number=None, subject="s"))
    monkeypatch.setattr(
        git_ops,
        "candidate_list",
        lambda repo, good, bad: [
            Candidate(commit=good, pr_number=None, subject="good"),
            Candidate(commit=bad, pr_number=None, subject="bad"),
        ],
    )
    monkeypatch.setattr("tools.bisect.auto_bisect.expected_versions", lambda repo, commit=None: PackageVersions())
    inp = BisectInput(scene="single_node", config_yaml="case.yaml", bad_commit="b" * 40, soc="a2", good_commit="a" * 40)
    opt = BisectOptions(repo_dir=tmp_path, work_dir=str(tmp_path / "work"), assume_built_head=False)
    bisector = Bisector(inp, opt)

    def fatal(good, candidates, state):
        raise BisectFatalError("no worker ever joined")

    monkeypatch.setattr(bisector, "_verify_endpoints", fatal)

    with pytest.raises(BisectFatalError, match="no worker"):
        bisector.run()

    assert bisector.report_path.exists()


def test_main_returns_2_on_fatal_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fatal(self):
        raise BisectFatalError("no worker ever joined")

    monkeypatch.setattr(Bisector, "run", fatal)

    rc = main(
        [
            "--scene",
            "single_node",
            "--config-yaml",
            "case.yaml",
            "--soc",
            "a2",
            "--good-commit",
            "a" * 40,
            "--repo-dir",
            str(tmp_path),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    assert rc == 2
