import json
import time
from pathlib import Path

import pytest

from tools.bisect.coordinator import Coordinator


def test_publish_command_and_wait_command_round_trip(tmp_path: Path):
    coord = Coordinator(str(tmp_path), num_nodes=2, node_index=1)

    coord.publish_command(1, "abcdef1234567890", rebuild=True)

    command = coord.wait_command(1, timeout_s=0.1)
    assert command == {
        "round": 1,
        "commit": "abcdef1234567890",
        "rebuild": True,
        "action": "RUN",
    }


def test_wait_command_ignores_stale_done_when_since_ts_is_newer(tmp_path: Path):
    coord = Coordinator(str(tmp_path), num_nodes=1, node_index=0)
    coord.publish_done()
    since_ts = time.time() + 1
    coord.publish_command(1, "abcdef1234567890", rebuild=False)

    command = coord.wait_command(1, timeout_s=0.1, since_ts=since_ts)

    assert command is not None
    assert command["commit"] == "abcdef1234567890"


def test_wait_command_returns_none_for_release_file(tmp_path: Path):
    coord = Coordinator(str(tmp_path / "coord"), num_nodes=1, node_index=0)
    release_file = tmp_path / "done"
    release_file.write_text("done", encoding="utf-8")

    assert coord.wait_command(1, timeout_s=0.1, release_file=str(release_file)) is None


def test_wait_all_ready_rejects_split_commit(tmp_path: Path):
    coord = Coordinator(str(tmp_path), num_nodes=1, node_index=0)
    round_dir = tmp_path / "round_1"
    round_dir.mkdir()
    (round_dir / "ready_0.json").write_text(json.dumps({"node": 0, "head": "badbadbadbad"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected abcdef123456"):
        coord.wait_all_ready(1, "abcdef1234567890", timeout_s=0.1)
