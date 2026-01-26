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
"""

from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCore

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Patch EngineCore.__init__ to trigger profiling after initialization
# ---------------------------------------------------------------------------
_original_engine_core_init = EngineCore.__init__


def _patched_engine_core_init(self, *args, **kwargs):
    _original_engine_core_init(self, *args, **kwargs)

    if hasattr(self.scheduler, "run_profiling_chunk_init"):
        logger.info("[ProfilingChunk] Running profiling initialization...")
        self.scheduler.run_profiling_chunk_init(self.model_executor)


EngineCore.__init__ = _patched_engine_core_init


# ---------------------------------------------------------------------------
# 2. Patch step / step_with_batch_queue to record execution timing
# ---------------------------------------------------------------------------

def _record_execution_timing(scheduler, scheduler_output, model_output):
    """Record execution timing for online model refinement.

    Extracts ``execution_time_ms`` (set dynamically by the NPU model runner)
    from the model output and feeds it back to the
    ``ProfilingChunkManager`` for incremental fitting of the history-aware
    latency model.
    """
    profiling_mgr = getattr(scheduler, "profiling_chunk_manager", None)
    if profiling_mgr is None or not profiling_mgr.is_ready:
        return

    elapsed_time_ms = getattr(model_output, "execution_time_ms", 0.0)
    if elapsed_time_ms <= 0:
        return
    elapsed_time = elapsed_time_ms / 1000.0

    try:
        total_tokens = getattr(
            scheduler_output, "total_num_scheduled_tokens", 0
        )
        if total_tokens <= 0:
            return

        num_scheduled_tokens = getattr(
            scheduler_output, "num_scheduled_tokens", {}
        )
        request_chunks = []

        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
        for req in new_reqs:
            req_id = getattr(req, "request_id", None) or getattr(
                req, "req_id", None
            )
            if req_id and req_id in num_scheduled_tokens:
                chunk_size = num_scheduled_tokens[req_id]
                hist_seq_len = getattr(req, "num_computed_tokens", 0)
                if chunk_size > 0:
                    request_chunks.append((chunk_size, hist_seq_len))

        cached_reqs = getattr(
            scheduler_output, "scheduled_cached_reqs", None
        )
        if cached_reqs is not None:
            req_ids = getattr(cached_reqs, "req_ids", [])
            computed_tokens_list = getattr(
                cached_reqs, "num_computed_tokens", []
            )
            for i, req_id in enumerate(req_ids):
                if req_id in num_scheduled_tokens:
                    chunk_size = num_scheduled_tokens[req_id]
                    hist_seq_len = (
                        computed_tokens_list[i]
                        if i < len(computed_tokens_list)
                        else 0
                    )
                    if chunk_size > 0:
                        request_chunks.append((chunk_size, hist_seq_len))

        if not request_chunks:
            request_chunks = [(total_tokens, 0)]

        if not profiling_mgr.predictor.history_fitted:
            profiling_mgr.record_batch_execution_time(
                request_chunks, elapsed_time
            )

    except (AttributeError, TypeError) as e:
        logger.debug("Failed to record execution timing: %s", e)


# ---------------------------------------------------------------------------
# 3. Wrap scheduler.update_from_output to intercept model_output for timing
# ---------------------------------------------------------------------------
# Both step() and step_with_batch_queue() call
# scheduler.update_from_output(scheduler_output, model_output), so
# wrapping this single method captures timing in all code paths.
_original_update_from_output = None


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
        return _original_update_from_output(
            self, scheduler_output, model_output
        )

    cls.update_from_output = _wrapped_update_from_output


# Chain onto the __init__ patch to apply the wrapper once the scheduler
# instance exists.
_init_before_wrap = EngineCore.__init__


def _patched_engine_core_init_with_wrap(self, *args, **kwargs):
    _init_before_wrap(self, *args, **kwargs)
    _ensure_update_from_output_wrapped(self.scheduler)


EngineCore.__init__ = _patched_engine_core_init_with_wrap
