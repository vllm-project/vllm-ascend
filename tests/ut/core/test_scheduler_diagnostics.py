from types import SimpleNamespace

import vllm_ascend.patch.platform.patch_scheduler as patch_scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.core.scheduler_diagnostics import print_scheduler_summary


def _request(status: RequestStatus, num_tokens: int):
    return SimpleNamespace(status=status, all_token_ids=list(range(num_tokens)))


def test_print_scheduler_summary_includes_waiting_queues(capsys):
    scheduler = SimpleNamespace(
        running=[_request(RequestStatus.RUNNING, 17)],
        waiting=[_request(RequestStatus.WAITING, 9)],
        skipped_waiting=[_request(RequestStatus.WAITING_FOR_REMOTE_KVS, 5)],
        block_size=8,
    )

    print_scheduler_summary(scheduler)

    output = capsys.readouterr().out
    assert "schedule() | scheduler req num: [1, 2, 1, 1, 0, 0]" in output
    assert "blk num [3, 2, 1]" in output


def test_normal_scheduler_prints_summary(monkeypatch):
    events = []
    scheduler_output = object()
    monkeypatch.setattr(
        patch_scheduler,
        "_ORIGINAL_SCHEDULE",
        lambda scheduler: events.append("schedule") or scheduler_output,
    )
    monkeypatch.setattr(
        patch_scheduler,
        "print_scheduler_summary",
        lambda scheduler: events.append("summary"),
    )

    assert patch_scheduler._schedule_with_summary(object()) is scheduler_output
    assert events == ["schedule", "summary"]
