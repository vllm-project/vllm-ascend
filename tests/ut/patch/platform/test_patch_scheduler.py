from types import SimpleNamespace

import vllm_ascend.patch.platform.patch_scheduler as patch_scheduler


def _scheduler(enable_diagnostics: bool):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={
                "scheduler_config": {
                    "nonbsp_config": {
                        "enabled": False,
                        "enable_diagnostics": enable_diagnostics,
                    }
                }
            }
        )
    )


def test_scheduler_diagnostics_are_disabled_by_default(monkeypatch):
    scheduler_output = object()
    summaries = []
    monkeypatch.setattr(
        patch_scheduler,
        "_ORIGINAL_SCHEDULE",
        lambda self, *args, **kwargs: scheduler_output,
    )
    monkeypatch.setattr(
        patch_scheduler,
        "print_scheduler_summary",
        lambda *args: summaries.append(args),
    )

    result = patch_scheduler._scheduler_schedule_with_diagnostics(_scheduler(False))

    assert result is scheduler_output
    assert summaries == []


def test_bl_scheduler_diagnostics_can_be_enabled(monkeypatch):
    scheduler = _scheduler(True)
    scheduler_output = object()
    summaries = []
    monkeypatch.setattr(
        patch_scheduler,
        "_ORIGINAL_SCHEDULE",
        lambda self, *args, **kwargs: scheduler_output,
    )
    monkeypatch.setattr(
        patch_scheduler,
        "print_scheduler_summary",
        lambda *args: summaries.append(args),
    )

    result = patch_scheduler._scheduler_schedule_with_diagnostics(scheduler)

    assert result is scheduler_output
    assert summaries == [(scheduler, scheduler_output)]


def test_bl_async_scheduler_diagnostics_can_be_enabled(monkeypatch):
    scheduler = _scheduler(True)
    scheduler_output = object()
    summaries = []
    monkeypatch.setattr(
        patch_scheduler,
        "_ORIGINAL_ASYNC_SCHEDULE",
        lambda self, *args, **kwargs: scheduler_output,
    )
    monkeypatch.setattr(
        patch_scheduler,
        "print_scheduler_summary",
        lambda *args: summaries.append(args),
    )

    result = patch_scheduler._async_scheduler_schedule_with_diagnostics(scheduler)

    assert result is scheduler_output
    assert summaries == [(scheduler, scheduler_output)]
