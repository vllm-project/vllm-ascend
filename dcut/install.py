# SPDX-License-Identifier: Apache-2.0
"""D-Cut patch application + vLLM general-plugin entrypoint."""

from __future__ import annotations

import os

# Module-level print so the user can see WHICH process imports dcut.install.
# Using print(flush=True) because logger.* is silently swallowed in the dcut
# vLLM service process.
print(
    f"[D-Cut] install module imported by pid={os.getpid()} "
    "(GDN PIECEWISE graph switch is evaluated during config creation).",
    flush=True,
)

from .globals import ENV_CONFIG, logger
from .patch_attention import _patch_attention
from .patch_full_graph import _patch_full_decode_multishape
# Importing patch_piecewise triggers its module-level arming of the
# CompilationConfig.set_splitting_ops_for_v1 patch — this is what makes the
# GDN PIECEWISE patch take effect in the EngineCore process (where
# install() is never called).  The explicit _arm_* call in install() below
# is kept as a belt-and-suspenders for the Worker process.
from .patch_piecewise import (
    _arm_gdn_piecewise_splitting_patch,
    _is_enabled as _gdn_piecewise_graph_enabled,
)
from .patch_proposer import _patch_proposer
from .patch_runner import _patch_runner
from .patch_worker import _patch_worker


def _apply_patches_once() -> None:
    """Apply the real monkey patches.  Runs once per process, deferred to the
    first worker construction (see ``install``) so that importing the NPU
    worker/runner/proposer modules is safe."""
    from . import globals as _g

    if _g._PATCHED:
        return
    # Mark done up-front so a failure (e.g. non-Ascend platform) is not retried
    # on every subsequent worker construction and does not spam the log.
    _g._PATCHED = True
    try:
        from .patch_gdn_v023 import _patch_gdn_dcut

        if os.environ.get(ENV_CONFIG) and not _patch_gdn_dcut():
            raise RuntimeError(
                "D-Cut GDN state operators are unavailable; run `bash dcut/kernel/build.sh` first"
            )
        if (
            os.environ.get(ENV_CONFIG)
            and _gdn_piecewise_graph_enabled()
            and not _patch_attention()
        ):
            raise RuntimeError(
                "D-Cut could not preserve the eager full-attention "
                "boundary while capturing GDN"
            )
        _patch_full_decode_multishape()
        _patch_proposer()
        _patch_runner()
        _patch_worker()
        logger.info(
            "D-Cut adaptive-verify patches applied for NPU "
            "(active only if VLLM_DCUT_CONFIG is set + method is dflash/PARD)."
        )
    except Exception as e:  # pragma: no cover - never break vLLM startup
        logger.error("D-Cut patching failed (vLLM continues normally): %s", e)


def install(*args, **kwargs) -> None:
    """vLLM general-plugin entrypoint.  Idempotent; safe to call per process.

    IMPORTANT — deferred by design.  ``install`` runs during *general-plugin
    load*, which happens BEFORE vllm-ascend has finished importing its own
    ``ops/fused_moe`` / ``device`` graph.  Eagerly importing the NPU
    worker/runner/proposer modules here re-enters that partially-initialised
    graph and raises a circular ``ImportError`` that poisons ``sys.modules`` —
    which then breaks vllm-ascend's *own* later imports (e.g.
    ``pre_register_and_update`` -> ``select_experts``), taking down even vanilla
    serving.  So here we only *arm* a deferred trigger on the vLLM-core
    ``WorkerBase`` (safe to import at this point) and apply the real patches on
    the first worker construction, by which time vllm-ascend is fully imported.

    The GDN PIECEWISE splitting_ops patch is an exception: it wraps a
    vLLM-core ``CompilationConfig`` method that runs during config creation
    (before any worker exists), so it must be armed here *before* the deferred
    trigger.  Only vLLM-core symbols are imported — no vllm_ascend.worker —
    so there is no circular-import risk.
    """
    from . import globals as _g

    if _g._INSTALLED:
        return
    _g._INSTALLED = True
    try:
        # Arm the GDN PIECEWISE splitting_ops patch early — must run before
        # vllm-ascend platform creates the compilation config.  (Also armed
        # at module-import time above, but re-call here for the Worker
        # process in case the import was somehow skipped.)
        _arm_gdn_piecewise_splitting_patch()

        from vllm.v1.worker.worker_base import WorkerBase

        if getattr(WorkerBase, "_dcut_defer_armed", False):
            return
        _orig_wb_init = WorkerBase.__init__

        def __init__(self, *a, **k):
            # NPUWorker.__init__ calls super().__init__() (this) early, before it
            # builds the model runner — so patching here lands before any
            # NPUModelRunner / proposer instance exists.
            _apply_patches_once()
            return _orig_wb_init(self, *a, **k)

        WorkerBase.__init__ = __init__
        WorkerBase._dcut_defer_armed = True
        print(
            "[D-Cut] deferred installer armed on WorkerBase "
            "(patches apply on first worker init to avoid a vllm-ascend "
            "circular import).",
            flush=True,
        )
    except Exception as e:  # pragma: no cover - never break vLLM startup
        logger.error("D-Cut install (arm) failed (vLLM continues normally): %s", e)
