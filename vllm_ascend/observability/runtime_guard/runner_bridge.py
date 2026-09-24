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

"""Model-runner glue: pre-sample wrap + async output after-sample hooks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# Written from ModelRunner.postprocess_sampled; read only by sample-phase hooks.
_POSTPROCESS_SAMPLED_ATTR = "_obs_postprocess_sampled_tokens"
_POSTPROCESS_NUM_ATTR = "_obs_postprocess_num_sampled"


def note_postprocess_sampled(runner: Any, sampled_tokens: Any, num_sampled: Any) -> None:
    """Stash spec-sample stats for the current sample phase (clears with None)."""
    setattr(runner, _POSTPROCESS_SAMPLED_ATTR, sampled_tokens)
    setattr(runner, _POSTPROCESS_NUM_ATTR, num_sampled)


def get_postprocess_sampled(runner: Any) -> tuple[Any, Any]:
    """Return ``(sampled_tokens, num_sampled)`` stashed by :func:`note_postprocess_sampled`."""
    return (
        getattr(runner, _POSTPROCESS_SAMPLED_ATTR, None),
        getattr(runner, _POSTPROCESS_NUM_ATTR, None),
    )


def need_pre_sample_hook(guard: Any) -> bool:
    """True when wrapping ``compute_logits`` is useful (detector on + gate open)."""
    cfg = getattr(guard, "runtime_config", None)
    if cfg is None:
        return False
    if not bool(cfg.detector_get("logits_finite", "enabled", False)):
        return False
    executor = getattr(guard, "action_executor", None)
    can = getattr(executor, "can_run_detection", None)
    return not (callable(can) and not bool(can()))


def check_before_sample_from_batch(
    guard: Any,
    logits: Any,
    input_batch: Any,
) -> None:
    """Pack batch fields and call :meth:`RuntimeGuardProcessor.check_before_sample`."""
    runner = getattr(guard, "runner", None)
    logits_indices = getattr(input_batch, "logits_indices", None)
    if logits_indices is None:
        logits_indices = getattr(runner, "logits_indices", None)
    guard.check_before_sample(
        logits=logits,
        logits_indices=logits_indices,
        input_batch=input_batch,
    )


@contextmanager
def wrap_compute_logits_for_pre_sample(runner: Any, input_batch: Any):
    """Wrap ``model.compute_logits`` so ``check_before_sample`` runs once before grammar.

    Parent ``sample`` has no mid-hook between logits and grammar; wrapping the
    bound method inserts the check without copying upstream ``sample()``.
    Restored in ``finally``. Fires once per context (v2 calls ``compute_logits``
    twice per step).
    """
    model = runner.model
    # Prefer deleting the instance override so the class method is restored.
    had_instance_attr = "compute_logits" in getattr(model, "__dict__", {})
    orig = model.compute_logits
    guard = runner.runtime_guard
    fired = False

    def wrapped(hidden_states, *args, **kwargs):
        nonlocal fired
        logits = orig(hidden_states, *args, **kwargs)
        if not fired:
            fired = True
            check_before_sample_from_batch(guard, logits, input_batch)
        return logits

    model.compute_logits = wrapped
    try:
        yield
    finally:
        # had_instance_attr: restore the saved override. Otherwise the wrapped
        # bound method sits in __dict__ (set unconditionally above) — delete it
        # so the class method is restored.
        if had_instance_attr:
            model.compute_logits = orig
        elif "compute_logits" in getattr(model, "__dict__", {}):
            del model.compute_logits


def is_async_output_rank() -> bool:
    """True on the TP rank that materializes async model-runner output (TP0)."""
    try:
        return get_tp_group().rank_in_group == 0
    except Exception:
        return True


def maybe_wrap_v2_async_output(output: Any, runner: Any) -> Any:
    """Wrap v2 ``AsyncOutput`` so ``check_after_sample`` runs after D2H trim.

    Non-output ranks keep the bare ``AsyncOutput`` (unique_reply_rank already
    forwards only TP0 into ``get_output``).
    """
    if not isinstance(output, AsyncOutput):
        return output
    if not is_async_output_rank():
        return output
    return AscendAsyncOutput(output, runner)


class AscendAsyncGPUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    """v1 async output: run ``check_after_sample`` after D2H in ``get_output``."""

    def __init__(self, *args: Any, runner: Any | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._runner = runner

    def get_output(self) -> ModelRunnerOutput:
        output = super().get_output()
        if self._runner is None:
            return output
        try:
            self._runner.runtime_guard.check_after_sample(
                sampled_token_ids=output.sampled_token_ids,
                req_ids=output.req_ids,
            )
        except Exception:
            logger.exception("[runtime_guard soft-fail] async check_after_sample raised")
        return output


class AscendAsyncOutput(AsyncModelRunnerOutput):
    """v2 async output: run ``check_after_sample`` after inner ``AsyncOutput`` D2H.

    Must be the only after-sample IO append for v2: ``AsyncOutput.sampled_token_ids``
    is padded until ``get_output`` trims with ``num_sampled`` (W2-3 / D-11).
    """

    def __init__(self, inner: AsyncOutput, runner: Any):
        self._inner = inner
        self._runner = runner

    def get_output(self) -> ModelRunnerOutput:
        output = self._inner.get_output()
        try:
            self._runner.runtime_guard.check_after_sample(
                sampled_token_ids=output.sampled_token_ids,
                req_ids=output.req_ids,
            )
        except Exception:
            logger.exception("[runtime_guard soft-fail] async check_after_sample raised")
        return output
