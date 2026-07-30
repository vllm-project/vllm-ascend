import json
from pathlib import Path

from tools.bisect.state import BisectState


def test_bisect_state_save_and_load_round_trips(tmp_path: Path):
    path = tmp_path / "state.json"
    state = BisectState(
        good="a" * 40,
        bad="b" * 40,
        lo=2,
        hi=5,
        round_idx=3,
        verdicts={"c" * 40: "PASS", "d" * 40: "SKIP"},
    )

    state.save(path)

    assert BisectState.load(path, good="a" * 40, bad="b" * 40) == state


def test_bisect_state_load_ignores_stale_good_bad_range(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "good": "old-good",
                "bad": "old-bad",
                "lo": 1,
                "hi": 2,
                "round_idx": 3,
                "verdicts": {},
            }
        ),
        encoding="utf-8",
    )

    assert BisectState.load(path, good="new-good", bad="new-bad") is None


def test_bisect_state_load_missing_file_returns_none(tmp_path: Path):
    assert BisectState.load(tmp_path / "missing.json", good="good", bad="bad") is None
