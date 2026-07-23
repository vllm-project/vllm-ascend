from types import SimpleNamespace

from vllm.v1.request import RequestStatus

from vllm_ascend.core.scheduler_diagnostics import (
    diagnostics_enabled,
    print_scheduler_summary,
)


def _request(request_id: str, status: RequestStatus, num_tokens: int):
    return SimpleNamespace(
        request_id=request_id,
        status=status,
        all_token_ids=list(range(num_tokens)),
    )


def test_print_scheduler_summary_includes_waiting_queues(capsys):
    scheduler = SimpleNamespace(
        running=[_request("running", RequestStatus.RUNNING, 17)],
        waiting=[_request("waiting", RequestStatus.WAITING, 9)],
        skipped_waiting=[
            _request(
                "remote",
                RequestStatus.WAITING_FOR_REMOTE_KVS,
                5,
            )
        ],
        block_size=8,
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"request-1": 1, "request-2": 1})

    print_scheduler_summary(scheduler, scheduler_output)

    output = capsys.readouterr().out
    assert "schedule() | scheduler req num: [2, 2, 1, 1, 0, 0]" in output
    assert "blk num [3, 2, 1]" in output


def test_print_scheduler_summary_distinguishes_lb_pause_from_preemption(capsys):
    scheduler = SimpleNamespace(
        running=[],
        waiting=[
            _request("waiting", RequestStatus.WAITING, 9),
            _request("lb-paused", RequestStatus.PREEMPTED, 17),
            _request("preempted", RequestStatus.PREEMPTED, 25),
        ],
        skipped_waiting=[],
        block_size=8,
        _lb_paused_req_ids={"lb-paused"},
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={})

    print_scheduler_summary(scheduler, scheduler_output)

    output = capsys.readouterr().out
    assert "schedule() | scheduler req num: [0, 3, 2, 0, 0, 1]" in output


def test_diagnostics_enabled_reads_nested_nonbsp_config():
    vllm_config = SimpleNamespace(
        additional_config={
            "scheduler_config": {
                "nonbsp_config": {
                    "enabled": False,
                    "enable_diagnostics": True,
                }
            }
        }
    )

    assert diagnostics_enabled(vllm_config) is True


def test_diagnostics_enabled_is_false_for_missing_or_invalid_config():
    assert diagnostics_enabled(SimpleNamespace(additional_config=None)) is False
    assert diagnostics_enabled(SimpleNamespace(additional_config={})) is False
    assert (
        diagnostics_enabled(
            SimpleNamespace(
                additional_config={
                    "scheduler_config": {
                        "nonbsp_config": {
                            "enable_diagnostics": "true",
                        }
                    }
                }
            )
        )
        is False
    )
