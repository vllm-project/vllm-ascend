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
"""EngineCore initialization hook for profiling-based chunk sizing.

The scheduler owns chunk sizing and execution-timing feedback. This module
only patches ``EngineCore.__init__`` so startup profiling can run after the
model executor is available.
"""

from vllm.logger import logger
from vllm.v1.engine.core import EngineCore

_profiling_patches_applied = False


# ---------------------------------------------------------------------------
# Core: apply EngineCore.__init__ patches (idempotent)
# ---------------------------------------------------------------------------


def _apply_profiling_patches():
    """Patch ``EngineCore.__init__`` to trigger startup profiling.

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

    EngineCore.__init__ = _patched_engine_core_init


# ---------------------------------------------------------------------------
# 1. Apply patches at module level for the InprocClient (in-process) path.
# ---------------------------------------------------------------------------
_apply_profiling_patches()

# The EngineCoreProc entry-point patch lives in patch_engine_core.py.
