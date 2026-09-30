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
protocol methods where practical, and keep every decorator pure
observability — deleting any of them must leave the functional path
byte-identical (worker ``execute_dummy_batch`` pattern): the decorated
methods keep their original functional bodies, and the decorators only
add wave sync / sample-phase orchestration around them.

v2 uses :func:`runtime_guard_step` on ``execute_model`` and
:func:`runtime_guard_sample_tokens` on ``sample_tokens`` (pre-sample logits
wrap + SamplePhaseResult assembly + async after-sample wrap). ModelRunner
v1 is **not** wired — runtime_guard is v2-only.

Decorator placement: guard step decorators must sit INSIDE
``@torch.inference_mode()`` so wave sync stays in that context. The worker
idle decorator has no inference-mode wrapper — ``_dummy_run`` owns it.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.observability.runtime_guard.processor import SamplePhaseResult

logger = init_logger_ascend(__name__)

# Stashed by ``runtime_guard_step`` for the same-wave sample phase.
_SCHEDULER_OUTPUT_ATTR = "_pending_scheduler_output"


def runtime_guard_step(execute_model_fn):
    """Wave-level runtime_guard sync for ``execute_model`` (v2).

    - before the step body: ``sync_for_step`` (config bus, wave arm, reap)
    - ``finally``: ``end_of_wave_sync(allow_manual_dump=False)`` on no-sample paths
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
        allow_manual_dump = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0) > 0
        guard.sync_for_step(scheduler_output=scheduler_output, allow_manual_dump=allow_manual_dump)
        try:
            return execute_model_fn(self, scheduler_output, *args, **kwargs)
        finally:
            # Collectives must stay lockstep — do not soft-fail this gate.
            if dummy_run or self.execute_model_state is None:
                guard.end_of_wave_sync(allow_manual_dump=False)

    return wrapper


def runtime_guard_idle_step(dummy_batch_fn):
    """Worker-level wave sync for idle DP ranks (``execute_dummy_batch``).

    Same lockstep gate as ``runtime_guard_step`` — do not soft-fail
    ``sync_for_step`` (busy ranks take the same collectives). Guard lives on
    ``self.model_runner`` (v2).
    """

    @functools.wraps(dummy_batch_fn)
    def wrapper(self, *args, **kwargs):
        runner = getattr(self, "model_runner", None)
        guard = getattr(runner, "runtime_guard", None)
        if guard is not None:
            guard.sync_for_step(allow_manual_dump=False)
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
      ``postprocess_sampled`` stash (``note_postprocess_sampled``)
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
            logits_ctx = (
                wrap_compute_logits_for_pre_sample(self, input_batch) if need_pre_sample_hook(guard) else nullcontext()
            )
            with logits_ctx, wrap_postprocess_sampled(self):
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


# ---- runner bind / sample hooks ----

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


def _restore_bound_override(obj: Any, name: str, orig: Any, *, had_instance_attr: bool) -> None:
    """Undo a temporary instance override installed by a wrap contextmanager."""
    if had_instance_attr:
        setattr(obj, name, orig)
    elif name in getattr(obj, "__dict__", {}):
        delattr(obj, name)


def _safe_check_after_sample(runner: Any, output: Any) -> None:
    """Run after-sample detect; never raise into the async copy / output path."""
    try:
        runner.runtime_guard.check_after_sample(
            sampled_token_ids=output.sampled_token_ids,
            req_ids=output.req_ids,
        )
    except Exception:
        logger.exception("[runtime_guard soft-fail] async check_after_sample raised")


@contextmanager
def wrap_compute_logits_for_pre_sample(runner: Any, input_batch: Any) -> Iterator[None]:
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
            batch = input_batch if input_batch is not None else getattr(runner, "input_batch", None)
            check_before_sample_from_batch(guard, logits, batch)
        return logits

    model.compute_logits = wrapped
    try:
        yield
    finally:
        _restore_bound_override(model, "compute_logits", orig, had_instance_attr=had_instance_attr)


@contextmanager
def wrap_postprocess_sampled(runner: Any) -> Iterator[None]:
    """Stash spec-sample stats from parent ``postprocess_sampled`` (v2).

    ``sampled_tokens`` / ``num_sampled`` are local intermediates of the
    parent ``sample_tokens`` chain, unreachable from the decorator. Wrap the
    bound method for the duration of the sample phase so the runner body
    keeps zero guard footprint (worker pattern). Restored in ``finally``.
    """
    orig = getattr(runner, "postprocess_sampled", None)
    if orig is None:
        yield
        return
    # Prefer deleting the instance override so the class method is restored.
    had_instance_attr = "postprocess_sampled" in getattr(runner, "__dict__", {})

    def wrapped(*args: Any, **kwargs: Any):
        # Positional: (idx_mapping, sampled_tokens, num_sampled, ...)
        sampled = args[1] if len(args) > 1 else kwargs.get("sampled_tokens")
        num = args[2] if len(args) > 2 else kwargs.get("num_sampled")
        note_postprocess_sampled(runner, sampled, num)
        return orig(*args, **kwargs)

    runner.postprocess_sampled = wrapped
    try:
        yield
    finally:
        _restore_bound_override(runner, "postprocess_sampled", orig, had_instance_attr=had_instance_attr)


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
        _safe_check_after_sample(self._runner, output)
        return output
