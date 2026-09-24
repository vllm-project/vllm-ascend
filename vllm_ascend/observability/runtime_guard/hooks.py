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

"""Runner/worker decorators wiring the runtime_guard into the engine loop.

Moving the wave sync / sample-phase orchestration out of the worker files
dedups the v1/v2 step-sync logic and shrinks the model-runner diff surface on
vLLM version bumps. Functional (guard-independent) work stays on the runner
as ``_rg_*`` hook methods; only the guard orchestration lives here.

Decorator placement rule: guard decorators must sit INSIDE
``@torch.inference_mode()`` (i.e. listed below it) so the wave sync keeps
running inside the inference-mode context exactly as before. The worker-level
idle decorator has no inference-mode wrapper to order against —
``_dummy_run`` owns its execution context.
"""

from __future__ import annotations

import functools
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_guard.processor import SamplePhaseResult
from vllm_ascend.observability.runtime_guard.runner_bridge import (
    need_pre_sample_hook,
    wrap_compute_logits_for_pre_sample,
)

logger = init_logger_ascend(__name__)


@dataclass(frozen=True)
class SamplePhasePreState:
    """Runner values peeked from ``execute_model_state`` before the sample phase.

    The peek must happen before the parent ``sample_tokens`` pops the ephemeral
    state; the refs stay valid after the pop.
    """

    input_batch: Any = None
    finished_req_ids: Any = None


def runtime_guard_step(execute_model_fn):
    """Wave-level runtime_guard sync for ``execute_model`` (shared by v1/v2).

    - before the step body: ``sync_for_step`` (config bus, wave arm, reap)
    - ``finally``: ``end_of_wave_sync(allow_arm=False)`` on no-sample paths
      (early return / exception / dummy wave) so the lockstep collectives can
      never be skipped.

    MRV2 dummy waves never reach ``sample_tokens``, so ``dummy_run`` steps flush
    here as well (v1 parity: dummy waves never arm — ``advance(allow_arm=False)``
    is a no-op and ``manual_dump`` is not burned).

    Must be listed BELOW ``@torch.inference_mode()``: the wave sync runs inside
    the inference-mode context.
    """

    @functools.wraps(execute_model_fn)
    def wrapper(self, scheduler_output, *args, **kwargs):
        guard = getattr(self, "runtime_guard", None)
        # The sample phase reads this stash (SamplePhaseResult.scheduler_output).
        self._rg_scheduler_output = scheduler_output
        if guard is None:
            # ``__new__`` UTs omit ``runtime_guard``; keep the bare path.
            return execute_model_fn(self, scheduler_output, *args, **kwargs)
        # MRV2 signature: (scheduler_output, intermediate_tensors, dummy_run, ...).
        dummy_run = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        allow_arm = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) > 0
        guard.sync_for_step(scheduler_output=scheduler_output, allow_arm=allow_arm)
        try:
            return execute_model_fn(self, scheduler_output, *args, **kwargs)
        finally:
            # Collectives must stay lockstep — do not soft-fail this gate.
            # No-sample / early return: do not burn manual_dump.
            if dummy_run or self.execute_model_state is None:
                guard.end_of_wave_sync(allow_arm=False)

    return wrapper


def runtime_guard_idle_step(dummy_batch_fn):
    """Worker-level wave sync for idle DP ranks (``execute_dummy_batch``).

    Idle DP ranks skip ``execute_model`` entirely, so ``@runtime_guard_step``
    never fires there — the worker must issue the same lockstep
    ``sync_for_step`` (``allow_arm=False``: dummy waves never consume
    ``manual_dump``; no ``scheduler_output`` — nothing was scheduled).

    Soft-fails, unlike ``runtime_guard_step``: this path carries no
    end-of-wave collectives, so a guard hiccup on an idle rank must not
    stall the dummy loop. The guard lives on ``self.model_runner``.
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


def runtime_guard_sample_tokens(sample_tokens_fn):
    """Sample-phase runtime_guard orchestration for ``sample_tokens`` (MRV2).

    Runner hooks (defined on the decorated class):

    - ``_rg_before_sample_phase() -> SamplePhasePreState``: functional
      pre-sample work (PCP global-batch restore etc.) plus the ephemeral-state
      peek. Runs on every path — the work is guard-independent.
    - ``_rg_sample_phase_result(output, pre) -> SamplePhaseResult``: builds the
      result handed to ``run_sample_phase`` (reads the ``postprocess_sampled``
      stashes).
    - ``_rg_after_sample_phase(output, guard)``: functional sample tail
      (spec-PP broadcast) plus the guard async-output wrap. Runs on every
      path.

    Keeping the functional work inside the hooks (not only on the guarded
    path) preserves behavior when ``runtime_guard`` is absent (``__new__`` UTs).
    """

    @functools.wraps(sample_tokens_fn)
    def wrapper(self, grammar_output):
        # Functional pre-sample work must run before the parent pops the
        # ephemeral state, guard or not.
        pre: SamplePhasePreState = self._rg_before_sample_phase()
        guard = getattr(self, "runtime_guard", None)
        if guard is None:
            output = sample_tokens_fn(self, grammar_output)
            return self._rg_after_sample_phase(output, None)

        def sample_fn() -> SamplePhaseResult:
            with (
                wrap_compute_logits_for_pre_sample(self, pre.input_batch)
                if need_pre_sample_hook(guard)
                else nullcontext()
            ):
                output = sample_tokens_fn(self, grammar_output)
            return self._rg_sample_phase_result(output, pre)

        speculative_config = getattr(self, "speculative_config", None)
        result, _ = guard.run_sample_phase(
            sample_fn=sample_fn,
            speculative_config=speculative_config,
            need_accepted_tokens=False,
            use_async=bool(getattr(self, "use_async_scheduling", False)),
            accepted_token_nums_fn=(
                (lambda _result: self._rg_spec_num_sampled) if speculative_config is not None else None
            ),
        )
        return self._rg_after_sample_phase(result.model_runner_output, guard)

    return wrapper
