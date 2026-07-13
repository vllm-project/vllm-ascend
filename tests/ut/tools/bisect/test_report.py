import json
from pathlib import Path

from tools.bisect.config import BisectInput, Candidate, TrialResult
from tools.bisect.report import write_report_json


def test_write_report_json_serializes_candidates_and_trials(tmp_path: Path):
    good = Candidate(commit="a" * 40, pr_number="1", subject="good")
    bad = Candidate(commit="b" * 40, pr_number="2", subject="bad")
    first_bad = Candidate(commit="c" * 40, pr_number="3", subject="first bad")
    trial = TrialResult(
        candidate=first_bad,
        verdict="FAIL",
        duration_s=1.24,
        rebuilt=True,
        exit_code=1,
        log_path="/tmp/round.log",
        note="pytest failed",
    )
    inp = BisectInput(scene="single_node", config_yaml="case.yaml", bad_commit=bad.commit, name="case")

    path = write_report_json(
        tmp_path / "report.json",
        inp=inp,
        good=good,
        bad=bad,
        first_bad=first_bad,
        trials=[trial],
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["case_key"] == "single_node::case.yaml"
    assert payload["good"]["commit"] == good.commit
    assert payload["bad"]["pr"] == "2"
    assert payload["first_bad_commit"] == first_bad.commit
    assert payload["trials"] == [
        {
            "commit": first_bad.commit,
            "pr": "3",
            "verdict": "FAIL",
            "rebuilt": True,
            "duration_s": 1.2,
            "exit_code": 1,
            "log": "/tmp/round.log",
            "note": "pytest failed",
        }
    ]
