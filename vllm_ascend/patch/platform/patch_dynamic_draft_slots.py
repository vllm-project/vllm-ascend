# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Budget native dynamic drafts without copying the upstream schedule loop.

Only calls through Scheduler.schedule are scoped. Independent Ascend schedule
overrides keep their existing budgets. Remove this adapter once the supported
vLLM base reserves slots and caps its output K using the same dynamic plan.
"""

from contextvars import ContextVar
from functools import wraps
from typing import Any, NamedTuple

from vllm.config import SpeculativeConfig
from vllm.v1.core.sched.scheduler import Scheduler


class _DraftPlan(NamedTuple):
    scheduler: Any
    spec: Any
    k: int
    slots: int


def _patch_scheduler(scheduler_cls: type, speculative_cls: type) -> None:
    original_schedule = scheduler_cls.schedule
    if getattr(original_schedule, "_ascend_dynamic_draft_slots", False):
        return
    slot_property = getattr(speculative_cls, "max_num_new_slots_for_drafting", None)
    if not isinstance(slot_property, property) or slot_property.fget is None:
        return  # Older vLLM bases do not have this input-slot budget.
    original_slots = slot_property.fget
    original_after = scheduler_cls._update_after_schedule
    current_plan: ContextVar[_DraftPlan | None] = ContextVar("ascend_dynamic_draft_slots", default=None)

    @wraps(original_slots)
    def draft_slots(spec):
        plan = current_plan.get()
        return plan.slots if plan is not None and plan.spec is spec else original_slots(spec)

    @wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        spec = self.vllm_config.speculative_config
        lookup = getattr(self, "dynamic_sd_lookup", None)
        plan = None
        if spec is not None and lookup is not None:
            dflash = spec.use_dflash()
            if dflash or (spec.parallel_drafting and not spec.uses_draft_model()):
                batch = min(len(self.running) + len(self.waiting), len(lookup) - 1)
                k = lookup[batch]
                slots = k if dflash else max(0, k - 1)
                plan = _DraftPlan(self, spec, k, min(original_slots(spec), slots))
        # Also mask an outer scheduler's plan for nested calls sharing a config.
        token = current_plan.set(plan)
        try:
            return original_schedule(self, *args, **kwargs)
        finally:
            current_plan.reset(token)

    @wraps(original_after)
    def update_after_schedule(self, output):
        plan = current_plan.get()
        if plan is not None and plan.scheduler is self and output.num_scheduled_tokens:
            output.num_spec_tokens_to_schedule = min(output.num_spec_tokens_to_schedule, plan.k)
        return original_after(self, output)

    schedule._ascend_dynamic_draft_slots = True
    speculative_cls.max_num_new_slots_for_drafting = property(
        draft_slots, slot_property.fset, slot_property.fdel, slot_property.__doc__
    )
    scheduler_cls._update_after_schedule = update_after_schedule
    scheduler_cls.schedule = schedule


_patch_scheduler(Scheduler, SpeculativeConfig)
