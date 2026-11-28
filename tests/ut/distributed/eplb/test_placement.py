# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import json
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.eplb.placement import (
    PlacementExporter,
    validate_placement_snapshot,
    write_placement_snapshot,
)


def _snapshot(with_draft: bool = False) -> dict:
    data: dict = {
        "version": 1,
        "ep_size": 2,
        "target": {
            "num_logical_experts": 4,
            "num_redundant_experts": 2,
            "physical_to_logical_map": [[0, 1, 2, 0, 1, 3]],
        },
    }
    if with_draft:
        data["draft"] = {
            "num_logical_experts": 2,
            "num_redundant_experts": 0,
            "physical_to_logical_map": [[1, 0]],
        }
    return data


def test_snapshot_write_target_and_optional_draft(tmp_path: Path):
    for with_draft in (False, True):
        path = tmp_path / f"placement_{int(with_draft)}.json"
        expected = _snapshot(with_draft=with_draft)
        write_placement_snapshot(path, expected)
        assert json.loads(path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        (lambda s: s.update(version=2), "version"),
        (lambda s: s.update(version=True), "version"),
        (lambda s: s.update(extra=1), "must contain"),
        (lambda s: s["target"].update(model="unexpected"), "must contain exactly"),
        (lambda s: s["target"].update(physical_to_logical_map=[[0, 1, 2, 0, 1, 2]]), "cover every"),
        (lambda s: s["target"].update(physical_to_logical_map=[[0, 0, 1, 2, 3, 1]]), "duplicate logical experts"),
        (lambda s: s["target"].update(physical_to_logical_map=[[False, 1, 2, 0, 1, 3]]), "invalid logical expert"),
        (lambda s: s["target"].update(physical_to_logical_map=[[0, 1, 2, 3]]), "slot count"),
    ],
)
def test_validate_placement_snapshot_rejects_invalid_data(mutator, pattern: str):
    data = _snapshot()
    mutator(data)
    with pytest.raises(ValueError, match=pattern):
        validate_placement_snapshot(data)


def test_atomic_write_preserves_previous_snapshot_on_failure(tmp_path: Path):
    path = tmp_path / "placement.json"
    previous = _snapshot(with_draft=False)
    write_placement_snapshot(path, previous)
    with (
        patch("vllm_ascend.distributed.eplb.placement.os.replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError, match="replace failed"),
    ):
        write_placement_snapshot(path, _snapshot(with_draft=True))
    assert json.loads(path.read_text(encoding="utf-8")) == previous
    assert list(tmp_path.iterdir()) == [path]


def test_pp_listener_survives_probes_write_errors_and_ipv6(tmp_path: Path):
    hosts = ["127.0.0.1"]
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
        hosts.append("::1")
    except OSError:
        pass

    for idx, host in enumerate(hosts):
        out = tmp_path / f"pp_{idx}.json"
        writer = PlacementExporter(ep_size=1, pp_rank=0, pp_size=2, is_stage_leader=True, path=str(out))
        sender = PlacementExporter(ep_size=1, pp_rank=1, pp_size=2, is_stage_leader=True, path=str(out))
        entry = {"num_logical_experts": 2, "num_redundant_experts": 0, "physical_to_logical_map": [[0, 1]]}
        updated = {"num_logical_experts": 2, "num_redundant_experts": 0, "physical_to_logical_map": [[1, 0]]}
        try:
            endpoint = writer.start_pp_listener(host)
            gathered: list[dict] = [
                {"pp_rank": 0, "target": None, "draft": None, "endpoint": endpoint},
                {"pp_rank": 1, "target": entry, "draft": None, "endpoint": None},
            ]
            writer.configure_pp_stages(gathered, same_node_as_writer=True)
            sender.configure_pp_stages(gathered, same_node_as_writer=True)

            # Empty probe + invalid auth payload must not kill the accept loop.
            with socket.create_connection((host, endpoint[1]), timeout=2.0):
                pass
            with socket.create_connection((host, endpoint[1]), timeout=2.0) as bad:
                bad.sendall(b"\x00\x00\x00\x02" + b"x" * 32 + b"{}")
                assert bad.recv(1) == b"\x00"

            assert sender.publish_evaluation(
                target_stats=(1.6, 1), target_entry=entry, draft_stats=None, draft_entry=None
            )
            assert not out.exists()

            sender.mark_committed("target")
            with patch("vllm_ascend.distributed.eplb.placement.write_placement_snapshot", side_effect=OSError("disk")):
                assert not sender.publish_evaluation(
                    target_stats=(1.1, 1), target_entry=updated, draft_stats=None, draft_entry=None
                )
            assert sender.has_unexported_commit
            assert writer.min_peak_to_mean_ratio == pytest.approx(1.6)

            assert sender.publish_evaluation(
                target_stats=(1.2, 1), target_entry=updated, draft_stats=None, draft_entry=None
            )
            assert not sender.has_unexported_commit
            assert json.loads(out.read_text(encoding="utf-8"))["target"]["physical_to_logical_map"] == [[1, 0]]
        finally:
            sender.close()
            writer.close()


def test_asymmetric_role_stats_preserve_pending_commits(tmp_path: Path):
    target_init = {"num_logical_experts": 2, "num_redundant_experts": 0, "physical_to_logical_map": [[0, 1]]}
    target_next = {"num_logical_experts": 2, "num_redundant_experts": 0, "physical_to_logical_map": [[1, 0]]}
    draft_init = {"num_logical_experts": 2, "num_redundant_experts": 0, "physical_to_logical_map": [[0, 1]]}
    draft_next = {"num_logical_experts": 2, "num_redundant_experts": 0, "physical_to_logical_map": [[1, 0]]}

    # 1. Draft commit reported during a prefill-only window (draft_stats=None) is retained for the next decode window.
    out_draft = tmp_path / "draft_retained.json"
    exporter = PlacementExporter(ep_size=1, pp_rank=0, pp_size=1, is_stage_leader=True, path=str(out_draft))
    assert exporter.publish_evaluation(
        target_stats=(1.8, 1), target_entry=target_init, draft_stats=(1.6, 1), draft_entry=draft_init
    )
    exporter.mark_committed("target")
    exporter.mark_committed("draft")
    assert exporter.publish_evaluation(
        target_stats=(1.2, 1), target_entry=target_next, draft_stats=None, draft_entry=draft_next
    )
    saved = json.loads(out_draft.read_text(encoding="utf-8"))
    assert saved["target"]["physical_to_logical_map"] == [[1, 0]]
    assert saved["draft"]["physical_to_logical_map"] == [[0, 1]]

    assert exporter.publish_evaluation(
        target_stats=(1.3, 1), target_entry=target_next, draft_stats=(1.1, 1), draft_entry=draft_next
    )
    saved = json.loads(out_draft.read_text(encoding="utf-8"))
    assert saved["draft"]["physical_to_logical_map"] == [[1, 0]]

    # 2. Target commit reported when target_stats=None is retained until target_stats is observed.
    out_target = tmp_path / "target_retained.json"
    exporter2 = PlacementExporter(ep_size=1, pp_rank=0, pp_size=1, is_stage_leader=True, path=str(out_target))
    assert exporter2.publish_evaluation(
        target_stats=(1.8, 1), target_entry=target_init, draft_stats=(1.6, 1), draft_entry=draft_init
    )
    exporter2.mark_committed("target")
    exporter2.mark_committed("draft")
    assert exporter2.publish_evaluation(
        target_stats=None, target_entry=target_next, draft_stats=(1.1, 1), draft_entry=draft_next
    )
    assert not out_target.exists()

    assert exporter2.publish_evaluation(
        target_stats=(1.2, 1), target_entry=target_next, draft_stats=(1.2, 1), draft_entry=draft_next
    )
    saved2 = json.loads(out_target.read_text(encoding="utf-8"))
    assert saved2["target"]["physical_to_logical_map"] == [[1, 0]]
    assert saved2["draft"]["physical_to_logical_map"] == [[1, 0]]
