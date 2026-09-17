import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

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


def test_save_preserves_previous_state_on_write_failure(tmp_path: Path):
    path = tmp_path / "state.json"
    BisectState(good="a" * 40, bad="b" * 40, lo=2, hi=5).save(path)
    old_bytes = path.read_bytes()
    state = BisectState(good="a" * 40, bad="b" * 40, lo=2, hi=4)
    original_factory = tempfile.NamedTemporaryFile

    @contextmanager
    def failing_temp_file(*args, **kwargs):
        with original_factory(*args, **kwargs) as temp_file:
            with patch.object(temp_file, "write", side_effect=OSError("disk full")):
                yield temp_file

    with (
        patch.object(tempfile, "NamedTemporaryFile", failing_temp_file),
        pytest.raises(OSError),
    ):
        state.save(path)

    assert path.read_bytes() == old_bytes
    assert BisectState.load(path, good="a" * 40, bad="b" * 40) is not None
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_preserves_previous_state_on_replace_failure(tmp_path: Path):
    path = tmp_path / "state.json"
    BisectState(good="a" * 40, bad="b" * 40, lo=2, hi=5).save(path)
    old_bytes = path.read_bytes()
    state = BisectState(good="a" * 40, bad="b" * 40, lo=2, hi=4)

    with (
        patch.object(Path, "replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError),
    ):
        state.save(path)

    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_failure_does_not_publish_initial_state(tmp_path: Path):
    path = tmp_path / "state.json"
    state = BisectState(good="a" * 40, bad="b" * 40)
    original_factory = tempfile.NamedTemporaryFile

    @contextmanager
    def failing_temp_file(*args, **kwargs):
        with original_factory(*args, **kwargs) as temp_file:
            with patch.object(temp_file, "write", side_effect=OSError("disk full")):
                yield temp_file

    with (
        patch.object(tempfile, "NamedTemporaryFile", failing_temp_file),
        pytest.raises(OSError),
    ):
        state.save(path)

    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_leaves_no_temporary_files(tmp_path: Path):
    path = tmp_path / "state.json"
    state = BisectState(good="a" * 40, bad="b" * 40)
    state.save(path)
    state.round_idx = 1
    state.save(path)

    assert list(tmp_path.glob("*.tmp")) == []
    assert BisectState.load(path, good="a" * 40, bad="b" * 40) == state
