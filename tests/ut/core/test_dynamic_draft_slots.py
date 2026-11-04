# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform.patch_dynamic_draft_slots import _patch_scheduler


@pytest.fixture
def scheduler_types():
    class Spec:
        def __init__(self, mode="dspark"):
            self.mode = mode
            self.parallel_drafting = mode in ("dspark", "dflash", "PARD")
            self.num_speculative_tokens = 7

        def use_dflash(self):
            return self.mode == "dflash"

        def uses_draft_model(self):
            return self.mode in ("draft", "PARD")

        @property
        def max_num_new_slots_for_drafting(self):
            return {"dspark": 6, "dflash": 7, "PARD": 7, "draft": 1, "eagle": 0}[self.mode]

    class Scheduler:
        def __init__(self, spec, lookup=None, batch=256):
            self.vllm_config = SimpleNamespace(speculative_config=spec)
            self.dynamic_sd_lookup = lookup
            self.running, self.waiting = [], [None] * batch
            self.hook = None
            self.admission_limit = 256

        def schedule(self):
            self.slots = self.vllm_config.speculative_config.max_num_new_slots_for_drafting
            if self.hook:
                self.hook()
            count = min(len(self.waiting), self.admission_limit, 2048 // (5 + self.slots))
            k = self.dynamic_sd_lookup[count] if count and self.dynamic_sd_lookup is not None else 7
            output = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(range(count), 5), num_spec_tokens_to_schedule=k)
            self._update_after_schedule(output)
            return output

        def _update_after_schedule(self, output):
            self.recorded_k = output.num_spec_tokens_to_schedule

    class AsyncScheduler(Scheduler):
        def _update_after_schedule(self, output):
            super()._update_after_schedule(output)
            self.placeholders = [-1] * output.num_spec_tokens_to_schedule

    _patch_scheduler(Scheduler, Spec)
    return Spec, Scheduler, AsyncScheduler


def test_dynamic_slots_restore_batch_without_changing_startup_config(scheduler_types):
    Spec, Scheduler, _ = scheduler_types
    spec = Spec()
    dynamic = Scheduler(spec, [0] + [4] * 256, batch=400)
    assert len(Scheduler(spec).schedule().num_scheduled_tokens) == 186
    assert len(dynamic.schedule().num_scheduled_tokens) == 256
    assert dynamic.slots == 3
    assert spec.max_num_new_slots_for_drafting == 6
    assert spec.num_speculative_tokens == 7


def test_final_k_is_capped_before_async_bookkeeping(scheduler_types):
    Spec, _, AsyncScheduler = scheduler_types
    scheduler = AsyncScheduler(Spec(), [0, 7] + [1] * 255, batch=2)
    scheduler.admission_limit = 1
    assert scheduler.schedule().num_spec_tokens_to_schedule == 1
    assert scheduler.recorded_k == 1
    assert scheduler.placeholders == [-1]


@pytest.mark.parametrize("mode,slots", [("dflash", 4), ("PARD", 7), ("draft", 1), ("eagle", 0)])
def test_drafting_mode_slot_contract(scheduler_types, mode, slots):
    Spec, Scheduler, _ = scheduler_types
    scheduler = Scheduler(Spec(mode), [0] + [4] * 256, batch=16)
    scheduler.schedule()
    assert scheduler.slots == slots


def test_nested_schedule_and_exception_restore_context(scheduler_types):
    Spec, Scheduler, _ = scheduler_types
    spec = Spec()
    outer = Scheduler(spec, [0] + [4] * 256)
    inner = Scheduler(spec)
    outer.hook = inner.schedule
    outer.schedule()
    assert inner.slots == 6

    def fail():
        raise ValueError("schedule interrupted")

    outer.hook = fail
    with pytest.raises(ValueError, match="interrupted"):
        outer.schedule()
    assert spec.max_num_new_slots_for_drafting == 6


def test_shared_config_is_isolated_across_threads(scheduler_types):
    Spec, Scheduler, _ = scheduler_types
    spec = Spec()
    scheduler = Scheduler(spec, [0] + [4] * 256)
    entered, release = Event(), Event()

    def hold_scope():
        entered.set()
        assert release.wait(timeout=5)

    scheduler.hook = hold_scope
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(scheduler.schedule)
        try:
            assert entered.wait(timeout=5)
            assert spec.max_num_new_slots_for_drafting == 6
            assert len(Scheduler(spec).schedule().num_scheduled_tokens) == 186
        finally:
            release.set()
        assert len(result.result(timeout=5).num_scheduled_tokens) == 256


def test_patch_installation_is_idempotent(scheduler_types):
    Spec, Scheduler, _ = scheduler_types
    original = Scheduler.schedule, Scheduler._update_after_schedule, Spec.max_num_new_slots_for_drafting
    _patch_scheduler(Scheduler, Spec)
    assert original == (Scheduler.schedule, Scheduler._update_after_schedule, Spec.max_num_new_slots_for_drafting)


def test_old_base_without_slot_budget_is_unchanged():
    class Scheduler:
        def schedule(self):
            return "original"

    original = Scheduler.schedule
    _patch_scheduler(Scheduler, type("OldSpec", (), {}))
    assert Scheduler.schedule is original
