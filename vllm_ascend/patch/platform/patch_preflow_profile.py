# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run PREFLOW cost calibration after EngineCore initialization."""

import functools

from vllm.v1.engine.core import EngineCore, EngineCoreProc

_PATCHED = False


def apply_preflow_profile_patch() -> None:
    """Install the startup hook once in the current process."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    original_init = EngineCore.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        run_profile = getattr(self.scheduler, "run_preflow_startup_profile", None)
        if run_profile is not None:
            run_profile(self)

    EngineCore.__init__ = patched_init


_ORIGINAL_RUN_ENGINE_CORE = EngineCoreProc.run_engine_core


def _run_engine_core_with_preflow_profile(*args, **kwargs):
    # In spawn mode, resolving this function imports this module in the child.
    # Reapply the process-local EngineCore.__init__ hook before construction.
    apply_preflow_profile_patch()
    return _ORIGINAL_RUN_ENGINE_CORE(*args, **kwargs)


def apply_preflow_profile_process_patch() -> None:
    """Install both in-process and spawned-process startup hooks."""
    apply_preflow_profile_patch()
    if EngineCoreProc.run_engine_core is not _run_engine_core_with_preflow_profile:
        EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_with_preflow_profile)


__all__ = ["apply_preflow_profile_process_patch"]
