import json
import subprocess
from pathlib import Path

from tools.bisect import auto_bisect, runner
from tools.bisect.verdict import RunOutcome


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo: Path, subject: str, content: str) -> str:
    (repo / "marker.txt").write_text(content, encoding="utf-8")
    _git(repo, "add", "marker.txt")
    _git(repo, "commit", "-m", subject)
    return _git(repo, "rev-parse", "HEAD")


def test_aop_cli_to_first_bad_report_full_chain(tmp_path: Path, monkeypatch):
    """Exercise the controllable AOP chain from CLI inputs to report.json.

    Git history, good/env tables, candidate selection, endpoint checks, verdict
    evaluation, binary search, state persistence and report generation are real.
    Only build/NPU execution is replaced by a deterministic runner.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Bisect UT")
    _git(repo, "config", "user.email", "bisect-ut@example.invalid")
    good = _commit(repo, "good baseline", "good")
    passing = _commit(repo, "still good (#101)", "passing")
    first_bad = _commit(repo, "introduce regression (#102)", "first bad")
    bad = _commit(repo, "current nightly head (#103)", "bad")

    good_table = tmp_path / "good_table.csv"
    good_table.write_text(
        "name,yaml/path,link,status,vLLM Git information,"
        "vLLM-Ascend Git information,soc,scene,time\n"
        f"aop-case,configs/case.yaml,run-good,success,vllm-good,{good},a2,single_node,"
        "2026-07-30 10:00:00 +08:00\n",
        encoding="utf-8",
    )
    env_table = tmp_path / "env_table.csv"
    env_table.write_text(
        "name,yaml/path,link,status,vLLM Git information,"
        "vLLM-Ascend Git information,CANN Version,torch-npu Version,time\n"
        f"aop-case,configs/case.yaml,run-good,success,vllm-good,{good},9.0.0,2.5.0,"
        "2026-07-30 10:00:00 +08:00\n"
        f"aop-case,configs/case.yaml,run-bad,failure,vllm-bad,{first_bad},9.0.1,2.6.0,"
        "2026-07-31 10:00:00 +08:00\n",
        encoding="utf-8",
    )

    observed = {}

    class FakeBuildManager:
        def __init__(self, opt):
            self.opt = opt

    class DeterministicRunner:
        def __init__(self, opt):
            self.opt = opt
            self.finished = False

        def validate(self, candidate, round_idx, log_dir):
            observed.setdefault("validated", []).append(candidate.commit)
            observed["env_by_commit"] = dict(self.opt.env_by_commit)
            return RunOutcome(exit_code=1 if candidate.commit in {first_bad, bad} else 0)

        def teardown(self):
            observed["teardown_count"] = observed.get("teardown_count", 0) + 1

        def finish(self):
            self.finished = True
            observed["finished"] = True

    monkeypatch.setattr(auto_bisect, "BuildManager", FakeBuildManager)
    monkeypatch.setattr(runner, "build_runner", lambda inp, opt, builder: DeterministicRunner(opt))

    work_dir = tmp_path / "work"
    rc = auto_bisect.main(
        [
            "--scene",
            "single_node",
            "--config-yaml",
            "case.yaml",
            "--config-base-path",
            "configs",
            "--name",
            "aop-case",
            "--soc",
            "a2",
            "--bad-commit",
            bad,
            "--good-table",
            str(good_table),
            "--env-table",
            str(env_table),
            "--repo-dir",
            str(repo),
            "--work-dir",
            str(work_dir),
            "--fail-confirm-retries",
            "0",
        ]
    )

    report_path = work_dir / "single_node__case.yaml" / "report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["first_bad_commit"] == first_bad
    assert payload["first_bad_pr"] == "102"
    assert [trial["commit"] for trial in payload["trials"]] == [bad, good, first_bad, passing]
    assert [trial["verdict"] for trial in payload["trials"]] == ["FAIL", "PASS", "FAIL", "PASS"]
    assert observed["env_by_commit"][good]["vllm_ref"] == "vllm-good"
    assert observed["env_by_commit"][passing]["vllm_ref"] == "vllm-good"
    assert observed["env_by_commit"][first_bad]["vllm_ref"] == "vllm-bad"
    assert observed["env_by_commit"][bad]["vllm_ref"] == "vllm-bad"
    assert observed["teardown_count"] == 4
    assert observed["finished"] is True
