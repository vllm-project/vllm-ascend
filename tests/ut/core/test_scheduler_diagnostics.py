from types import SimpleNamespace

from vllm.v1.request import RequestStatus

from vllm_ascend.core.scheduler_diagnostics import (
    diagnostics_enabled,
    print_scheduler_summary,
)


def _request(status: RequestStatus, num_tokens: int):
    return SimpleNamespace(status=status, all_token_ids=list(range(num_tokens)))


def test_print_scheduler_summary_includes_waiting_queues(capsys):
    scheduler = SimpleNamespace(
        running=[_request(RequestStatus.RUNNING, 17)],
        waiting=[_request(RequestStatus.WAITING, 9)],
        skipped_waiting=[_request(RequestStatus.WAITING_FOR_REMOTE_KVS, 5)],
        block_size=8,
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"request-1": 1, "request-2": 1})

    print_scheduler_summary(scheduler, scheduler_output)

    output = capsys.readouterr().out
    assert "schedule() | scheduler req num: [2, 2, 1, 1, 0, 0]" in output
    assert "blk num [3, 2, 1]" in output


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
