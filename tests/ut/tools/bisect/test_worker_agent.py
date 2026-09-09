import json
import logging
import threading
import time
from pathlib import Path

from tools.bisect import git_ops
from tools.bisect.config import BisectInput, BisectOptions
from tools.bisect.coordinator import Coordinator
from tools.bisect.worker_agent import run_worker


def _wait_for_condition(condition, timeout_s: float = 60.0, desc: str = "condition") -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {desc}")


def _publish_command(coord_dir: Path, rnd: int, commit: str) -> None:
    rdir = coord_dir / f"round_{rnd}"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "command.json").write_text(
        json.dumps(
            {
                "round": rnd,
                "commit": commit,
                "rebuild": False,
                "action": "RUN",
                "version_targets": {},
                "version_checks": [],
            }
        ),
        encoding="utf-8",
    )


def _start_worker(inp: BisectInput, opt: BisectOptions) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            run_worker(inp, opt)
        except BaseException as exc:  # noqa: BLE001 - recorded and asserted below
            errors.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, errors


def _join_worker(thread: threading.Thread, errors: list[BaseException]) -> None:
    thread.join(timeout=120)
    assert not thread.is_alive(), "run_worker did not exit (hung)"
    assert errors == [], f"run_worker crashed: {errors!r}"


def _worker_options(tmp_path: Path, worker_repo: Path, coord_dir: Path) -> BisectOptions:
    return BisectOptions(
        repo_dir=worker_repo,
        work_dir=str(tmp_path / "work"),
        coord_dir=str(coord_dir),
        num_nodes=2,
        node_index=1,
        barrier_timeout_s=30,
        assume_built_head=True,
    )


def test_worker_deploys_ancestor_commit_missing_from_shallow_clone(
    tmp_path: Path,
    repo_with_commits,
    shallow_clone,
    monkeypatch,
):
    """Regression: the CI worker's repo is a depth-1 clone, so the commanded
    bisect commit (an ancestor) is missing locally. The worker must recover it
    from origin instead of dying with GitError (which leaves the master waiting
    out the barrier timeout on every round)."""
    src, shas = repo_with_commits(3)
    mid = shas[1]
    worker_repo = shallow_clone(src, tmp_path / "worker_repo")

    coord_dir = tmp_path / "coord"
    _publish_command(coord_dir, 1, mid)

    inp = BisectInput(scene="multi_node", config_yaml="case.yaml", bad_commit="bad", soc="a3")
    opt = _worker_options(tmp_path, worker_repo, coord_dir)

    launched = []
    monkeypatch.setattr("tools.bisect.worker_agent._launch_pytest", lambda *a, **k: launched.append(a) or 0)
    monkeypatch.setattr("tools.bisect.worker_agent.runner.kill_stray_servers", lambda: None)

    thread, errors = _start_worker(inp, opt)

    # The worker deployed the commanded ancestor and reported it ready.
    ready = coord_dir / "round_1" / "ready_1.json"
    _wait_for_condition(ready.exists, desc=str(ready))
    assert json.loads(ready.read_text(encoding="utf-8"))["head"] == mid
    assert git_ops.current_commit(worker_repo) == mid

    # Master-side flow: release the round, then end the bisect.
    coord = Coordinator(str(coord_dir), num_nodes=2, node_index=0)
    coord.publish_start(1)
    _wait_for_condition(lambda: len(launched) == 1, desc="round-1 pytest launch")
    coord.publish_done()

    _join_worker(thread, errors)
    assert launched, "worker never launched the round's pytest"


def test_worker_survives_command_for_unresolvable_commit(
    tmp_path: Path,
    repo_with_commits,
    shallow_clone,
    monkeypatch,
):
    """A commit that even origin cannot provide must not kill the worker agent:
    it signals the deliberate 'worker-deploy-failed' marker so the master can
    SKIP the round, and keeps serving later rounds."""
    src, shas = repo_with_commits(2)
    tip = shas[1]
    worker_repo = shallow_clone(src, tmp_path / "worker_repo")

    coord_dir = tmp_path / "coord"
    _publish_command(coord_dir, 1, "c" * 40)

    inp = BisectInput(scene="multi_node", config_yaml="case.yaml", bad_commit="bad", soc="a3")
    opt = _worker_options(tmp_path, worker_repo, coord_dir)

    monkeypatch.setattr("tools.bisect.worker_agent._launch_pytest", lambda *a, **k: 0)
    monkeypatch.setattr("tools.bisect.worker_agent.runner.kill_stray_servers", lambda: None)

    thread, errors = _start_worker(inp, opt)

    ready = coord_dir / "round_1" / "ready_1.json"
    _wait_for_condition(ready.exists, desc=str(ready))
    assert json.loads(ready.read_text(encoding="utf-8"))["head"] == "worker-deploy-failed"
    assert git_ops.current_commit(worker_repo) == tip

    # Master aborts the round (SKIP verdict); the worker loops on and exits on DONE.
    coord = Coordinator(str(coord_dir), num_nodes=2, node_index=0)
    coord.publish_verdict(1, "SKIP")
    coord.publish_done()

    _join_worker(thread, errors)


def test_worker_survives_wait_start_timeout(tmp_path: Path, repo_with_commits, monkeypatch, caplog):
    """A missing start/abort decision for a round (master still rebuilding, or
    the round abandoned after a desync) must not kill the worker agent: it
    moves on to the next round instead of dying on the raised TimeoutError."""
    src, shas = repo_with_commits(3)
    worker_repo = src  # a full clone: the deploy is an instant local checkout
    coord_dir = tmp_path / "coord"
    _publish_command(coord_dir, 1, shas[1])

    inp = BisectInput(scene="multi_node", config_yaml="case.yaml", bad_commit="bad", soc="a3")
    opt = _worker_options(tmp_path, worker_repo, coord_dir)
    opt.barrier_timeout_s = 5

    monkeypatch.setattr("tools.bisect.worker_agent._launch_pytest", lambda *a, **k: 0)
    monkeypatch.setattr("tools.bisect.worker_agent.runner.kill_stray_servers", lambda: None)

    thread, errors = _start_worker(inp, opt)

    ready = coord_dir / "round_1" / "ready_1.json"
    _wait_for_condition(ready.exists, desc=str(ready))

    # Publish NO start/verdict for round 1: wait_start must time out and the
    # worker must continue to the next round instead of dying.
    with caplog.at_level(logging.ERROR, logger="bisect.worker"):
        _wait_for_condition(
            lambda: any("no start/abort decision" in r.message for r in caplog.records),
            timeout_s=30,
            desc="wait_start timeout log",
        )

    coord = Coordinator(str(coord_dir), num_nodes=2, node_index=0)
    coord.publish_done()

    _join_worker(thread, errors)
