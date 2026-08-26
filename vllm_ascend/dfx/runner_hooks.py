#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-runner glue helpers that belong to DFX, not Ascend business logic."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


def need_pre_sample_hook(dfx: Any) -> bool:
    """True when wrapping ``compute_logits`` is useful (detector on + gate open)."""
    cfg = getattr(dfx, "dfx_config", None)
    if cfg is None:
        return False
    if not (
        bool(cfg.detector_get("logits_finite", "enabled", False))
        or bool(cfg.detector_get("position_alignment", "enabled", False))
    ):
        return False
    dumper = getattr(dfx, "dumper", None)
    can = getattr(dumper, "can_run_anomaly_detection", None)
    return not (callable(can) and not bool(can()))


def check_before_sample_from_batch(
    dfx: Any,
    logits: Any,
    input_batch: Any,
    *,
    scheduler_output: Any = None,
) -> None:
    """Pack batch fields and call :meth:`DfxProcessor.check_before_sample`."""
    positions = getattr(input_batch, "positions", None)
    total = 0
    if scheduler_output is not None:
        total = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    if total <= 0:
        total = int(getattr(input_batch, "num_tokens", 0) or 0)
    dfx.check_before_sample(
        scheduler_output=scheduler_output,
        logits=logits,
        positions=positions,
        total_scheduled_tokens=total,
        logits_indices=getattr(input_batch, "logits_indices", None),
        input_batch=input_batch,
    )


@contextmanager
def wrap_compute_logits_for_pre_sample(runner: Any, input_batch: Any):
    """Temporarily wrap ``model.compute_logits`` so DFX runs before grammar.

    Parent ``GPUModelRunner.sample`` does ``compute_logits`` then grammar then
    sampler with no mid-hook. Wrapping the bound method inserts
    ``check_before_sample`` on the return path (still before grammar) without
    copying the upstream ``sample()`` body. Restored in ``finally`` so
    prompt-logprobs / draft paths are unaffected outside this call.
    """
    model = runner.model
    # Prefer deleting the instance override so the class method is restored
    # (assigning a bound method back leaves a stale instance attribute).
    had_instance_attr = "compute_logits" in getattr(model, "__dict__", {})
    orig = model.compute_logits
    dfx = runner.dfx
    scheduler_output = getattr(runner, "_dfx_scheduler_output", None)

    def wrapped(hidden_states, *args, **kwargs):
        logits = orig(hidden_states, *args, **kwargs)
        check_before_sample_from_batch(
            dfx,
            logits,
            input_batch,
            scheduler_output=scheduler_output,
        )
        return logits

    model.compute_logits = wrapped
    try:
        yield
    finally:
        if had_instance_attr:
            model.compute_logits = orig
        elif "compute_logits" in getattr(model, "__dict__", {}):
            del model.compute_logits
        else:
            model.compute_logits = orig
