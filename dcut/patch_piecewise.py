# SPDX-License-Identifier: Apache-2.0
"""Capture pure-spec GDN, including its recurrent update, in PIECEWISE mode.

When explicitly enabled, the expanded pure-spec GDN core is compiled into the
outer PIECEWISE graph. Native prefill and mixed batches retain the whole-core
boundary and therefore preserve the stock vLLM-Ascend path.
"""

from __future__ import annotations

import os

_WHOLE_GDN_OP = "vllm::qwen_gdn_attention_core"
_RECURRENT_OP = "vllm::dcut_gdn_recurrent"
ENV_DCUT_CONFIG = "VLLM_DCUT_CONFIG"
ENV_GDN_PIECEWISE = "VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE"
LEGACY_ENV_GDN_PIECEWISE = "VLLM_DCUT_GDN_PIECEWISE"


def _ensure_gdn_splitting_ops(ops):
    """Keep native GDN outside while capturing the pure-spec recurrent op."""
    result = [op for op in (ops or ()) if op != _RECURRENT_OP]
    # Keep this boundary for the untouched native prefill/mixed path and for
    # vLLM's PIECEWISE attention validation. It is absent from the expanded
    # pure-spec decode graph when the graphable recurrent route is active.
    if _WHOLE_GDN_OP not in result:
        result.append(_WHOLE_GDN_OP)
    return result


def _env_flag(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _is_enabled() -> bool:
    """Return whether GDN may be captured by PIECEWISE ACLGraph.

    The registered vllm-ascend variable is authoritative. The earlier
    D-Cut-only spelling remains accepted so existing launch scripts do not
    silently lose the optimization.
    """
    if not os.environ.get(ENV_DCUT_CONFIG):
        return False

    legacy = os.environ.get(LEGACY_ENV_GDN_PIECEWISE)
    if legacy is not None:
        return _env_flag(legacy)

    try:
        from vllm_ascend import envs

        return bool(envs.VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE)
    except (AttributeError, ImportError, ValueError):
        return _env_flag(os.environ.get(ENV_GDN_PIECEWISE, "0"))


def _arm_gdn_piecewise_splitting_patch():
    """Configure native fallback and recurrent capture in every process.
    Must be called **before** vllm-ascend platform code invokes
    ``set_splitting_ops_for_v1`` (i.e. during ``install()``, not during the
    deferred ``WorkerBase.__init__`` trigger).  Only vLLM-core symbols are
    imported here — no vllm_ascend.worker / model_runner — so there is no
    circular-import risk.

    Using ``print(flush=True)`` for all diagnostics because ``logger.*`` calls
    are silently swallowed in the dcut vLLM service process.
    """
    if not _is_enabled():
        print(
            "[D-Cut] GDN PIECEWISE boundary patch SKIPPED "
            f"(requires {ENV_DCUT_CONFIG} and "
            f"{ENV_GDN_PIECEWISE}=1).",
            flush=True,
        )
        return

    try:
        from vllm.config import CUDAGraphMode, CompilationConfig
    except Exception as exc:  # pragma: no cover - vLLM not installed
        print(
            f"[D-Cut] cannot import CompilationConfig, "
            f"GDN PIECEWISE graph patch NOT armed: {exc}",
            flush=True,
        )
        return

    if getattr(
        CompilationConfig.set_splitting_ops_for_v1,
        "_dcut_piecewise_patched",
        False,
    ):
        print(
            "[D-Cut] GDN PIECEWISE graph patch already armed (skip).",
            flush=True,
        )
        return

    _ORIG_SET_SPLITTING_OPS = CompilationConfig.set_splitting_ops_for_v1

    def _patched_set_splitting_ops_for_v1(self, *args, **kwargs):
        _ORIG_SET_SPLITTING_OPS(self, *args, **kwargs)

        if self.cudagraph_mode != CUDAGraphMode.PIECEWISE:
            print(
                "[D-Cut] GDN graph split configuration skipped for "
                f"cudagraph_mode={self.cudagraph_mode}.",
                flush=True,
            )
            return

        before = list(self.splitting_ops or ())
        after = _ensure_gdn_splitting_ops(self.splitting_ops)
        self.splitting_ops = after

        if before != after:
            print(
                "[D-Cut] configured GDN PIECEWISE splitting boundaries "
                f"(before={len(before)} ops, after={len(after)} ops).",
                flush=True,
            )
            print(f"[D-Cut] splitting_ops before={before}", flush=True)
            print(f"[D-Cut] splitting_ops after ={after}", flush=True)
        print(
            "[D-Cut] GDN PIECEWISE strategy: expanded pure-spec core with "
            "recurrent update inside the graph "
            f"(recurrent_is_boundary={_RECURRENT_OP in after}).",
            flush=True,
        )

    _patched_set_splitting_ops_for_v1._dcut_piecewise_patched = True  # type: ignore[attr-defined]
    CompilationConfig.set_splitting_ops_for_v1 = (  # type: ignore[assignment]
        _patched_set_splitting_ops_for_v1
    )
    print(
        "[D-Cut] GDN PIECEWISE graph selection patch armed.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Module-level arming — runs at import time so the patch is applied in EVERY
# process that imports the dcut package (EngineCore + Worker), not just where
# ``install()`` is called.  The vLLM general-plugin ``install()`` entrypoint
# is only invoked in Worker processes; the EngineCore process (where
# ``set_splitting_ops_for_v1`` actually runs during config creation) never
# calls ``install()``.  Arming at import time closes that gap.
# ---------------------------------------------------------------------------
print("[D-Cut] patch_piecewise module imported — arming patch.", flush=True)
try:
    _arm_gdn_piecewise_splitting_patch()
except Exception as _e:  # pragma: no cover - never break import
    print(f"[D-Cut] patch_piecewise module-level arm failed: {_e}", flush=True)
