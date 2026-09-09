import json
from pathlib import Path

import pytest

from tools.bisect import git_ops
from tools.bisect.build_manager import BuildDecision
from tools.bisect.config import BisectInput, BisectOptions, Candidate
from tools.bisect.coordinator import Coordinator
from tools.bisect.runner import BisectFatalError, MultiNodeRunner, SingleNodeRunner, _safe_name
from tools.bisect.version_compat import VersionAdaptationError


def test_safe_name_replaces_path_and_space_separators():
    assert _safe_name("configs/my case.yaml") == "configs_my_case.yaml"


def test_base_env_includes_case_and_config_base(tmp_path: Path):
    inp = BisectInput(
        scene="single_node",
        config_yaml="case.yaml",
        bad_commit="bad",
        soc="a2",
        config_base_path="configs",
    )
    opt = BisectOptions(repo_dir=tmp_path)
    runner = SingleNodeRunner(inp, opt, builder=None)  # type: ignore[arg-type]

    env = runner._base_env()

    assert env["CONFIG_YAML_PATH"] == "case.yaml"
    assert env["CONFIG_BASE_PATH"] == "configs"


def test_multi_node_runner_selects_external_dp_test_path(tmp_path: Path):
    inp = BisectInput(
        scene="multi_node",
        config_yaml="case.yaml",
        bad_commit="bad",
        soc="a3",
        config_base_path="tests/e2e/nightly/multi_node/external_dp/config",
    )
    opt = BisectOptions(repo_dir=tmp_path)
    runner = MultiNodeRunner(inp, opt, builder=None, coordinator=None)  # type: ignore[arg-type]

    assert runner._test_path().endswith("external_dp/scripts/test_external_dp.py")


class _StubBuilder:
    """Deploy stub: every commit deploys instantly without touching git."""

    def decide(self, commit: str) -> BuildDecision:
        return BuildDecision(rebuild=False, reinstall_reqs=False, native_hits=[], reason="stub")

    def prepare(self, commit: str, log_file=None) -> BuildDecision:
        return self.decide(commit)


def _multi_runner(tmp_path: Path, *, num_nodes: int = 2) -> MultiNodeRunner:
    inp = BisectInput(scene="multi_node", config_yaml="case.yaml", bad_commit="bad", soc="a3")
    opt = BisectOptions(
        repo_dir=tmp_path,
        coord_dir=str(tmp_path / "coord"),
        num_nodes=num_nodes,
        node_index=0,
        barrier_timeout_s=0.2,
        assume_built_head=False,
    )
    coord = Coordinator(opt.coord_dir, num_nodes, node_index=0)
    runner = MultiNodeRunner(inp, opt, builder=_StubBuilder(), coordinator=coord)  # type: ignore[arg-type]
    return runner


def test_barrier_timeout_without_any_worker_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When the barrier times out and no worker has EVER signalled ready, the
    bisect must abort instead of SKIP-scanning the whole range (each remaining
    round would burn the full barrier timeout)."""
    monkeypatch.setattr(git_ops, "current_commit", lambda repo: "b" * 40)
    runner = _multi_runner(tmp_path)
    candidate = Candidate(commit="b" * 40, pr_number=None, subject="bad")

    with pytest.raises(BisectFatalError, match="no worker node has ever signalled ready"):
        runner.validate(candidate, 1, tmp_path)

    # The round is still recorded as SKIP so a (late) worker can move on.
    assert (tmp_path / "coord" / "round_1" / "verdict.json").exists()


def test_barrier_timeout_with_late_worker_ready_still_skips_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A worker that signalled ready in an EARLIER round (late joiner whose
    marker landed after the master moved on) proves workers exist; the round
    is SKIPped as before rather than aborting the search."""
    monkeypatch.setattr(git_ops, "current_commit", lambda repo: "b" * 40)
    runner = _multi_runner(tmp_path)
    stale_round = tmp_path / "coord" / "round_1"
    stale_round.mkdir(parents=True)
    (stale_round / "ready_1.json").write_text('{"node": 1, "head": "b"}', encoding="utf-8")
    candidate = Candidate(commit="b" * 40, pr_number=None, subject="bad")

    with pytest.raises(VersionAdaptationError, match="Barrier failed"):
        runner.validate(candidate, 2, tmp_path)

    assert (tmp_path / "coord" / "round_2" / "verdict.json").exists()


def test_validate_publishes_skip_command_when_decision_fails_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A master-side deploy-decision failure BEFORE the command is published
    must still publish a SKIP command (and verdict) for the round -- otherwise
    the worker blocks in wait_command for the full barrier timeout."""
    monkeypatch.setattr(git_ops, "current_commit", lambda repo: "b" * 40)
    runner = _multi_runner(tmp_path)

    def decide_boom(commit):
        raise git_ops.GitError("boom")

    monkeypatch.setattr(runner.builder, "decide", decide_boom)
    candidate = Candidate(commit="b" * 40, pr_number=None, subject="bad")

    with pytest.raises(git_ops.GitError):
        runner.validate(candidate, 1, tmp_path)

    command = json.loads((tmp_path / "coord" / "round_1" / "command.json").read_text(encoding="utf-8"))
    assert command["action"] == "SKIP"
    assert (tmp_path / "coord" / "round_1" / "verdict.json").exists()


def test_validate_publishes_verdict_when_deploy_fails_after_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A master-side deploy failure AFTER the command was published must still
    publish the round's SKIP verdict so the already-deployed worker leaves
    wait_start promptly instead of dying on its timeout."""
    monkeypatch.setattr(git_ops, "current_commit", lambda repo: "b" * 40)
    runner = _multi_runner(tmp_path)

    def boom(commit, log_file=None):
        raise git_ops.GitError("checkout failed")

    monkeypatch.setattr(runner.builder, "prepare", boom)
    candidate = Candidate(commit="b" * 40, pr_number=None, subject="bad")

    with pytest.raises(git_ops.GitError):
        runner.validate(candidate, 1, tmp_path)

    assert (tmp_path / "coord" / "round_1" / "verdict.json").exists()
