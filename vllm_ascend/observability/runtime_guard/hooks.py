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
#

"""Decorators that wire runtime_guard into the engine loop.

Goal: keep ModelRunner/Worker free of runtime_guard types and ``_rg_*``
protocol methods, and keep every decorator pure observability — deleting
any of them must leave the functional path byte-identical (worker
``execute_dummy_batch`` pattern): the decorated methods keep their
original functional bodies, and the decorators only add wave sync /
sample-phase orchestration around them.

SamplePhaseResult construction, pre-sample logits wrap, and async-output wrap
live only here (and in ``runner_bridge``).

Decorator placement: guard step decorators must sit INSIDE
``@torch.inference_mode()`` so wave sync stays in that context. The worker
idle decorator has no inference-mode wrapper — ``_dummy_run`` owns it.
"""

from __future__ import annotations

import functools
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_guard.processor import SamplePhaseResult
from vllm_ascend.observability.runtime_guard.runner_bridge import (
    get_postprocess_sampled,
    maybe_wrap_v2_async_output,
    need_pre_sample_hook,
    note_postprocess_sampled,
    wrap_compute_logits_for_pre_sample,
)

logger = init_logger_ascend(__name__)

# Stashed by ``runtime_guard_step`` for the same-wave sample phase.
_SCHEDULER_OUTPUT_ATTR = "_pending_scheduler_output"


def runtime_guard_step(execute_model_fn):
    """Wave-level runtime_guard sync for ``execute_model`` (shared by v1/v2).

    - before the step body: ``sync_for_step`` (config bus, wave arm, reap)
    - ``finally``: ``end_of_wave_sync(allow_arm=False)`` on no-sample paths
      (early return / exception / dummy wave) so the lockstep collectives can
      never be skipped.

    Must be listed BELOW ``@torch.inference_mode()``.
    """

    @functools.wraps(execute_model_fn)
    def wrapper(self, scheduler_output, *args, **kwargs):
        guard = getattr(self, "runtime_guard", None)
        setattr(self, _SCHEDULER_OUTPUT_ATTR, scheduler_output)
        if guard is None:
            return execute_model_fn(self, scheduler_output, *args, **kwargs)
        dummy_run = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        allow_arm = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) > 0
        guard.sync_for_step(scheduler_output=scheduler_output, allow_arm=allow_arm)
        try:
            return execute_model_fn(self, scheduler_output, *args, **kwargs)
        finally:
            # Collectives must stay lockstep — do not soft-fail this gate.
            if dummy_run or self.execute_model_state is None:
                guard.end_of_wave_sync(allow_arm=False)

    return wrapper


def runtime_guard_idle_step(dummy_batch_fn):
    """Worker-level wave sync for idle DP ranks (``execute_dummy_batch``).

    Soft-fails (unlike ``runtime_guard_step``): this path must not stall the
    dummy loop. Guard lives on ``self.model_runner``.
    """

    @functools.wraps(dummy_batch_fn)
    def wrapper(self, *args, **kwargs):
        runner = getattr(self, "model_runner", None)
        guard = getattr(runner, "runtime_guard", None)
        if guard is not None:
            try:
                guard.sync_for_step(allow_arm=False)
            except Exception:
                logger.warning(
                    "[runtime_guard soft-fail] execute_dummy_batch sync_for_step failed",
                    exc_info=True,
                )
        return dummy_batch_fn(self, *args, **kwargs)

    return wrapper


def _peek_sample_pre_state(runner: Any) -> tuple[Any, Any]:
    """Peek ephemeral execute_model_state before parent ``sample_tokens`` pops it."""
    state = getattr(runner, "execute_model_state", None)
    input_batch = getattr(state, "input_batch", None) if state is not None else None
    finished_req_ids = getattr(state, "finished_req_ids", None) if state is not None else None
    return input_batch, finished_req_ids


def _build_sample_phase_result(runner: Any, output: Any, input_batch: Any, finished_req_ids: Any) -> SamplePhaseResult:
    """Assemble SamplePhaseResult inside the guard layer (not on the runner)."""
    req_ids = list(getattr(input_batch, "req_ids", None) or [])
    sampled, _num = get_postprocess_sampled(runner)
    # AsyncOutput.sampled_token_ids is padded until get_output() trims; defer
    # after-sample for AsyncModelRunnerOutput (W2-3 / D-11).
    return SamplePhaseResult(
        scheduler_output=getattr(runner, _SCHEDULER_OUTPUT_ATTR, None),
        input_batch=input_batch,
        model_runner_output=output,
        sampler_output=SimpleNamespace(sampled_token_ids=sampled),
        valid_sampled_token_ids=getattr(output, "sampled_token_ids", None),
        req_ids_output_copy=req_ids,
        invalid_req_indices=None,
        finished_req_ids=finished_req_ids,
    )


def runtime_guard_sample_tokens(sample_tokens_fn):
    """Guard orchestration for v2 ``sample_tokens`` — pure observability.

    Worker pattern: the decorated method keeps its original functional body
    (PCP swap, spec-PP draft broadcast), so deleting this decorator restores
    the pre-guard method byte-identically. This wrapper only adds guard
    orchestration:

    - guardless → bare method call (zero guard work);
    - guard → ``run_sample_phase`` around the method, reading the
      ``postprocess_sampled`` stash (``runner_bridge.note_postprocess_sampled``)
      and the pre-pop ``execute_model_state`` peek.
    """

    @functools.wraps(sample_tokens_fn)
    def wrapper(self, grammar_output):
        guard = getattr(self, "runtime_guard", None)
        if guard is None:
            return sample_tokens_fn(self, grammar_output)

        note_postprocess_sampled(self, None, None)  # clear prior-step stash
        # Peek before the method pops execute_model_state (inside the parent
        # sample_tokens). The method's PCP swap rewrites input_batch on
        # non-last-PP ranks only; the detection rank (last-PP TP0) — the only
        # consumer of these fields — is unaffected, so peeking before the
        # method is equivalent to peeking after the swap.
        input_batch, finished_req_ids = _peek_sample_pre_state(self)

        def sample_fn() -> SamplePhaseResult:
            with (
                wrap_compute_logits_for_pre_sample(self, input_batch)
                if need_pre_sample_hook(guard)
                else nullcontext()
            ):
                output = sample_tokens_fn(self, grammar_output)
            return _build_sample_phase_result(self, output, input_batch, finished_req_ids)

        speculative_config = getattr(self, "speculative_config", None)
        result, _ = guard.run_sample_phase(
            sample_fn=sample_fn,
            speculative_config=speculative_config,
            need_accepted_tokens=False,
            use_async=bool(getattr(self, "use_async_scheduling", False)),
            accepted_token_nums_fn=(
                (lambda _result: get_postprocess_sampled(self)[1]) if speculative_config is not None else None
            ),
        )
        output = result.model_runner_output
        if guard.needs_sample_phase_hooks():
            output = maybe_wrap_v2_async_output(output, self)
        return output

    return wrapper
