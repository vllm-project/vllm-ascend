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
"""Patches for profiling-based dynamic chunk sizing.

This module patches ``EngineCore`` to:
1. Run profiling at startup (after model_executor is ready).
2. Record execution timing after each model step to refine the
   history-aware chunk prediction model online.

In multiprocessing ``spawn`` mode the child process starts a fresh Python
interpreter, so class-level monkey-patches applied in the parent are lost.
To handle this we additionally wrap ``EngineCoreProc.run_engine_core``
(the subprocess entry-point): when pickle resolves the wrapper it triggers
an import of this module, which re-applies the ``EngineCore.__init__``
patches inside the child process before any ``EngineCore`` is instantiated.
"""

from vllm.logger import logger
from vllm.v1.engine.core import EngineCore, EngineCoreProc

_profiling_patches_applied = False
_original_update_from_output = None
_original_schedule = None


# ---------------------------------------------------------------------------
# Helper: record execution timing
# ---------------------------------------------------------------------------


def _record_execution_timing(scheduler, scheduler_output, model_output):
    """Record execution timing for online model refinement.

    Extracts ``execution_time_ms`` (set dynamically by the NPU model runner)
    from the model output and feeds it back to the
    ``ProfilingChunkManager`` for incremental fitting of the history-aware
    latency model.
    """
    profiling_mgr = getattr(scheduler, "profiling_chunk_manager", None)
    SET_TIME_COUNT = 3
    if profiling_mgr is None or not profiling_mgr.is_ready:
        return

    # Once both the target latency and history model are calibrated,
    # stop collecting timing data and disable the synchronize-and-time
    # calls in the model runner to avoid unnecessary pipeline stalls.
    if profiling_mgr._set_time_done and profiling_mgr.predictor.history_fitted:
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            get_ascend_config().profiling_chunk_config.need_timing = False
        except RuntimeError:
            pass
        # Mark the scheduler so that the next scheduler_output carries
        # a ``disable_profiling_timing`` flag to the worker process,
        # which will set its own process-local need_timing to False.
        scheduler._profiling_timing_done = True
        return

    elapsed_time_ms = getattr(model_output, "execution_time_ms", 0.0)
    if elapsed_time_ms <= 0:
        return
    elapsed_time = elapsed_time_ms / 1000.0

    try:
        total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", 0)
        if total_tokens <= 0:
            return

        num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})
        request_chunks = []

        total_hist_tokens = 0
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
        for req in new_reqs:
            req_id = getattr(req, "request_id", None) or getattr(req, "req_id", None)
            if req_id and req_id in num_scheduled_tokens:
                chunk_size = num_scheduled_tokens[req_id]
                hist_seq_len = getattr(req, "num_computed_tokens", 0)
                total_hist_tokens += hist_seq_len
                if chunk_size > 0:
                    request_chunks.append((chunk_size, hist_seq_len))

        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is not None:
            req_ids = getattr(cached_reqs, "req_ids", [])
            computed_tokens_list = getattr(cached_reqs, "num_computed_tokens", [])
            for i, req_id in enumerate(req_ids):
                if req_id in num_scheduled_tokens:
                    chunk_size = num_scheduled_tokens[req_id]
                    hist_seq_len = computed_tokens_list[i] if i < len(computed_tokens_list) else 0
                    total_hist_tokens += hist_seq_len
                    if chunk_size > 0:
                        request_chunks.append((chunk_size, hist_seq_len))

        # is first chunk processing — collect 3 samples before marking done
        if total_hist_tokens == 0 and not profiling_mgr._set_time_done:
            profiling_mgr.predictor.set_target_latency(0, elapsed_time * 1000)
            profiling_mgr._set_time_count += 1
            if profiling_mgr._set_time_count >= SET_TIME_COUNT:
                profiling_mgr._set_time_done = True

        if not request_chunks:
            # Cannot accurately attribute batch latency to individual
            # requests — skip this sample to avoid polluting the model.
            logger.debug("[ProfilingChunk] Skipping timing sample: unable to extract per-request chunk info")
            return

        if not profiling_mgr.predictor.history_fitted:
            profiling_mgr.record_batch_execution_time(request_chunks, elapsed_time)

    except (AttributeError, TypeError) as e:
        logger.debug("Failed to record execution timing: %s", e)


# ---------------------------------------------------------------------------
# Helper: wrap scheduler.update_from_output for timing
# ---------------------------------------------------------------------------


def _ensure_update_from_output_wrapped(scheduler):
    """Wrap scheduler.update_from_output to record execution timing."""
    global _original_update_from_output
    if _original_update_from_output is not None:
        return
    if not hasattr(scheduler, "profiling_chunk_manager"):
        return

    cls = type(scheduler)
    _original_update_from_output = cls.update_from_output

    def _wrapped_update_from_output(self, scheduler_output, model_output):
        _record_execution_timing(self, scheduler_output, model_output)
        return _original_update_from_output(self, scheduler_output, model_output)

    cls.update_from_output = _wrapped_update_from_output


def _ensure_schedule_wrapped(scheduler):
    """Wrap scheduler.schedule to propagate timing-done signal via scheduler_output.

    When ``_record_execution_timing`` detects that calibration is complete, it
    sets ``scheduler._profiling_timing_done = True``.  This wrapper copies that
    flag onto every subsequent ``SchedulerOutput`` so the worker process can
    read it and disable its own process-local ``need_timing``.
    """
    global _original_schedule
    if _original_schedule is not None:
        return
    if not hasattr(scheduler, "profiling_chunk_manager"):
        return

    cls = type(scheduler)
    _original_schedule = cls.schedule

    def _wrapped_schedule(self):
        output = _original_schedule(self)
        if getattr(self, "_profiling_timing_done", False) and output is not None:
            output.disable_profiling_timing = True
        return output

    cls.schedule = _wrapped_schedule


# ---------------------------------------------------------------------------
# Core: apply EngineCore.__init__ patches (idempotent)
# ---------------------------------------------------------------------------


def _apply_profiling_patches():
    """Patch ``EngineCore.__init__`` to trigger profiling and timing hooks.

    Safe to call multiple times; the guard ``_profiling_patches_applied``
    ensures the patch is applied at most once per process.
    """
    global _profiling_patches_applied
    if _profiling_patches_applied:
        return
    _profiling_patches_applied = True

    original_init = EngineCore.__init__

    def _patched_engine_core_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if hasattr(self.scheduler, "run_profiling_chunk_init"):
            logger.info("[ProfilingChunk] Running profiling initialization...")
            self.scheduler.run_profiling_chunk_init(self.model_executor)

        _ensure_update_from_output_wrapped(self.scheduler)
        _ensure_schedule_wrapped(self.scheduler)

    EngineCore.__init__ = _patched_engine_core_init


# ---------------------------------------------------------------------------
# 1. Apply patches at module level for the InprocClient (in-process) path.
# ---------------------------------------------------------------------------
_apply_profiling_patches()

# ---------------------------------------------------------------------------
# 2. Wrap EngineCoreProc.run_engine_core so that spawned subprocesses
#    re-apply the patches.  When the child unpickles this wrapper it
#    imports this module, which triggers _apply_profiling_patches() above,
#    ensuring EngineCore.__init__ is patched before any instance is created.
# ---------------------------------------------------------------------------
_original_run_engine_core = EngineCoreProc.run_engine_core


def _patched_run_engine_core(*args, **kwargs):
    _apply_profiling_patches()
    return _original_run_engine_core(*args, **kwargs)


EngineCoreProc.run_engine_core = _patched_run_engine_core
