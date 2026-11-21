"""Fix zero expert handling on Ascend NPU for LongCat-Flash with EP (vllm >= 0.23).

Problem 1: vllm-ascend's native zero-expert path never runs
------------------------------------------------------------
vllm 0.23 moved the zero-expert config onto ``ZeroExpertRouter``, so
``AscendUnquantizedFusedMoEMethod.apply``'s gate
``getattr(layer, "zero_expert_num", 0) > 0`` is always False.  Without it,
top-k ids in [N, N+Z) (zero experts) reach the dispatch kernel -> aicore crash.

Problem 2: MC2 MoE comm is incompatible with zero-expert weight zeroing
-----------------------------------------------------------------------
``npu_moe_distribute_dispatch_v2`` drops zero-weight slots, so
``MoeDistributeCombineV2``'s shape check (expandX.dim0 >= tokens*topk)
fails, the op never launches, and later collectives hang (AllGather
timeout).  The ALLGATHER comm method computes MoE locally after a gather,
where zero-weight slots are harmless (same semantics as the GPU path).

Problem 3: the native path adds the zero-expert result at the WRONG point
-------------------------------------------------------------------------
``AscendUnquantizedFusedMoEMethod.apply`` adds ``zero_expert_result`` onto
the fused-experts output *before* ``finalize`` and before the runner's
final TP/EP all-reduce.  With EP the MoE input is replicated across all
EP ranks, so every rank computes the SAME full identity contribution —
the downstream all-reduce then sums it ``world_size`` times (×64 on a
TP=EP=64 deployment), drowning the real output → garbled text (乱码).
Upstream adds the zero-expert output at the very END of
``MoERunner.forward`` (``_maybe_add_zero_expert_output``), AFTER
``_maybe_reduce_final_output``'s all-reduce — exactly once.

Fix
---
0b. Mirror ``ZeroExpertRouter`` config onto ``AscendFusedMoE`` so the native
    zero-expert path in ``apply`` runs (id sanitization is still needed);
    teach ``FusedExpertsResult`` ``+=`` for the native add.
0b2. Wrap ``zero_experts_compute`` in ``fused_moe.py``: stash the real
    identity contribution and return zeros instead, so the premature
    ``final_hidden_states += zero_expert_result`` in ``apply`` becomes a
    no-op (Problem 3).
0c. Optionally force the ALLGATHER MoE comm method
    (``EASYINFER_MOE_COMM=allgather``) to bypass MC2.
3.  In ``_maybe_add_zero_expert_output``, feed the stashed identity
    contribution to the runner so it is added once, at the end, after the
    final all-reduce (same semantics as upstream GPU).  Falls back to a
    scalar zero no-op when no stashed value is available (also satisfies
    the runner's ``assert zero_expert_output is not None``).

Version compatibility note
---------------------------
Patch 0b targets ``vllm_ascend.ops.fused_moe.fused_moe_0_23_0``, a
module whose name encodes the vllm_ascend version, and Patch 0c targets
``vllm_ascend.ascend_forward_context.select_moe_comm_method``.  If
vllm_ascend changes its internal naming/layout (e.g. ``fused_moe_0_24_0``),
these patches will be silently skipped by ``apply_all_patches``
(ImportError/AttributeError).  When upgrading vllm_ascend, verify that the
zero-expert EP path still works and update the target module names if
needed.

Cross-patch coordination via module-level state
------------------------------------------------
Patches 0b2 and 3 coordinate through a module-level global
``_pending_zero_expert_output``:

1. ``zero_experts_compute`` (patched by Patch 0b2) computes the real
   identity contribution but returns zeros to ``apply``, stashing the
   real result in the global.
2. ``_maybe_add_zero_expert_output`` (patched by Patch 3) retrieves the
   stash and feeds it to the runner for a single addition after the
   final all-reduce — the correct point in the computation graph.

This 1:1 producer-consumer pattern is safe because:
- Each MoE layer processes exactly one forward call per inference step.
- vLLM uses single-threaded CUDA stream execution (no concurrent MoE
  dispatches within one process).
- The stash is cleared (set to None) immediately after consumption.
"""

from __future__ import annotations

import os
import torch
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)

import logging

logger = logging.getLogger(__name__)

# ===========================================================================
# Patch 0b: enable vllm-ascend's NATIVE zero-expert handling (>= 0.23)
# ===========================================================================
# vllm 0.23 stores zero-expert config on the ZeroExpertRouter, so
# ``AscendUnquantizedFusedMoEMethod.apply``'s gate
# ``getattr(layer, "zero_expert_num", 0) > 0`` is always False and the
# native path (id sanitization + zero_expert_result add) never runs.
# Re-enable it by mirroring the router's config onto the layer.  The id
# sanitization is required; the premature add is neutralized by Patch 0b2.
#
# NOTE: The target module name ``fused_moe_0_23_0`` is version-encoded.
# If vllm_ascend changes its naming convention in future releases, this
# patch will be silently skipped.  See the module docstring for details.


def patch_enable_native_zero_expert(module: object) -> None:
    # Guard against repeated patching (e.g. module reload or multiple imports).
    if getattr(module.AscendFusedMoE, "_ez_patched", False):
        return
    module.AscendFusedMoE._ez_patched = True  # type: ignore[attr-defined]

    _orig_init = module.AscendFusedMoE.__init__

    # ``apply`` adds the zero-expert result onto the value returned by
    # ``fused_experts``; in that version the return value is a
    # ``FusedExpertsResult`` dataclass, not a tensor.  Give it an
    # ``__iadd__`` so ``result += zero_expert_result`` works.
    #
    # IMPORTANT: only inject ``__iadd__`` if the class doesn't already
    # define one (including inherited).  ``hasattr`` catches both own and
    # inherited definitions; if a parent class provides ``__iadd__`` we
    # must not silently replace it with our tensor-only version.
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

    if not hasattr(FusedExpertsResult, "__iadd__"):
        def _fused_experts_result_iadd(self, other):
            # ``other`` is always a plain tensor in the current ``apply``
            # code path (the return value of ``zero_experts_compute``).
            # Guard against an unexpected non-tensor value to fail loudly
            # rather than silently corrupting state.
            if not isinstance(other, torch.Tensor):
                raise TypeError(
                    "[fix_ep_zero_expert] FusedExpertsResult.__iadd__ "
                    "expected a Tensor, got %s" % type(other).__name__
                )
            # dataclasses.replace preserves every other field (expert_tokens,
            # group_list_type, swiglu_limit, ...); a manual reconstruction
            # would silently reset them to their defaults, breaking the
            # dynamic-EPLB bookkeeping that reads them after ``apply``.
            import dataclasses

            return dataclasses.replace(self, routed_out=self.routed_out + other)

        FusedExpertsResult.__iadd__ = _fused_experts_result_iadd
        logger.info(
            "[fix_ep_zero_expert] Injected FusedExpertsResult.__iadd__"
        )

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # Mirror ZeroExpertRouter config onto AscendFusedMoE so the native
        # zero-expert path in ``apply`` can read it.
        router = self.router
        if isinstance(router, ZeroExpertRouter) and router.zero_expert_type is not None:
            # Derive zero_expert_num from the bias shape when available
            # (bias covers real + zero experts), otherwise from the router's
            # own expert counts.  NOTE: ZeroExpertRouter has no
            # ``n_zero_experts`` attribute — do not probe for it.
            bias = getattr(router, "e_score_correction_bias", None)
            if bias is not None:
                n_zero = bias.shape[0] - self.global_num_experts
            else:
                n_zero = getattr(router, "global_num_experts", 0) - getattr(
                    router, "num_logical_experts", 0
                )
            if n_zero > 0:
                self.zero_expert_num = n_zero
                self.zero_expert_type = router.zero_expert_type
                # Flag consumed by Patch 3 to decide whether to redirect.
                router._ez_native_handled = True  # type: ignore[attr-defined]
                logger.info(
                    "[fix_ep_zero_expert] Enabled native zero-expert path: "
                    "num={}, type={}",
                    self.zero_expert_num,
                    self.zero_expert_type,
                )
            else:
                # zero_expert_type is set but n_zero derived as 0 —
                # inconsistent config or bias layout change.  Fail fast
                # because zero-expert IDs will NOT be sanitized and will
                # reach the dispatch kernel → aicore crash.
                raise RuntimeError(
                    "[fix_ep_zero_expert] zero_expert_type=%s is set but "
                    "derived n_zero=0 "
                    "(bias.shape=%s, global_num_experts=%s).  "
                    "Cannot enable native zero-expert path — the model "
                    "will crash without ID sanitization.  "
                    "The vllm-ascend version may be incompatible."
                    % (router.zero_expert_type,
                    tuple(bias.shape) if bias is not None else "N/A",
                    self.global_num_experts,
                )

    module.AscendFusedMoE.__init__ = _init


# ===========================================================================
# Patch 0b2: wrap zero_experts_compute — stash identity contribution
# ===========================================================================
# ``AscendUnquantizedFusedMoEMethod.apply`` computes the zero-expert
# identity contribution *inside* the fused-MoE code path by calling
# ``zero_experts_compute``, then adds it to the fused-experts output
# BEFORE ``finalize`` — before the runner's final TP/EP all-reduce.
#
# The wrapper below keeps the native call (its id/weight sanitization is
# required to keep zero-expert ids out of the dispatch kernel) but stashes
# the real identity contribution and returns zeros, making the premature
# ``final_hidden_states += zero_expert_result`` a no-op.  Patch 3 then
# hands the stashed value to the runner, which adds it once at the very
# end of ``MoERunner.forward`` — the same point upstream uses on GPU.

# Stash for the identity contribution computed inside ``apply``.  Written
# by the ``zero_experts_compute`` wrapper (Patch 0b2), consumed and cleared
# by the ``_maybe_add_zero_expert_output`` wrapper (Patch 3) — a strict
# 1:1 producer-consumer sequence per MoE layer per forward pass.
#
# Thread-safety: vLLM executes MoE layers sequentially on a single CUDA
# stream within each process, so a single slot is sufficient.  Torch
# multiprocessing uses ``spawn``, giving each rank its own memory space.
_pending_zero_expert_output: torch.Tensor | None = None


def patch_relocate_zero_expert_add(module: object) -> None:
    # Guard against repeated patching (consistent with the other patches).
    if getattr(module, "_ez_reloc_patched", False):
        return
    module._ez_reloc_patched = True  # type: ignore[attr-defined]

    _orig_zec = module.zero_experts_compute

    def _zero_experts_compute_stashing(*args, **kwargs):
        global _pending_zero_expert_output
        expert_indices, expert_scales, result = _orig_zec(*args, **kwargs)
        _pending_zero_expert_output = result
        # Zeros, not the real result: ``apply`` unconditionally adds this
        # to the fused-experts output pre-finalize, where it would be
        # all-reduced world_size times (Problem 3).
        return expert_indices, expert_scales, torch.zeros_like(result)

    module.zero_experts_compute = _zero_experts_compute_stashing
    logger.info(
        "[fix_ep_zero_expert] Wrapped zero_experts_compute: identity "
        "contribution relocated to the runner (post all-reduce)"
    )


# ===========================================================================
# Patch 0c: force ALLGATHER MoE comm (EASYINFER_MOE_COMM=allgather)
# ===========================================================================
# MC2 dispatch drops zero-weight (clamped zero-expert) slots, so the
# combine kernel's shape check ``expandX.dim0 >= tokens*topk`` fails and
# the op never launches, corrupting the stream and hanging later
# collectives (AllGather timeout).  This patch optionally forces the
# ALLGATHER comm method when the env var is set, bypassing MC2 entirely.
#
# NOTE: ``select_moe_comm_method`` is defined in
# ``vllm_ascend.ascend_forward_context`` (NOT in the version-encoded
# ``fused_moe_0_23_0`` module — an earlier revision of this patch targeted
# that module and was silently skipped with an AttributeError, leaving MC2
# active despite ``EASYINFER_MOE_COMM=allgather``).  Some callers
# (``vllm_ascend.platform``, ``vllm_ascend.worker.v2.model_runner``) hold
# from-import bindings, so after patching the defining module we also swap
# any stale references in already-imported modules.


def patch_force_allgather_comm(module: object) -> None:
    if os.environ.get("EASYINFER_MOE_COMM", "").lower() != "allgather":
        return

    # Guard against repeated patching (consistent with Patch 0b and Patch 3).
    if getattr(module, "_ez_ag_patched", False):
        return
    module._ez_ag_patched = True  # type: ignore[attr-defined]

    _orig = module.select_moe_comm_method
    _logged = False

    def _select(num_tokens, vllm_config, is_draft_model=False):
        nonlocal _logged
        selected = _orig(num_tokens, vllm_config, is_draft_model)
        if selected is not None:
            if not _logged:
                _logged = True
                logger.info(
                    "[fix_ep_zero_expert] MoE comm method overridden: "
                    "%s -> ALLGATHER",
                    selected,
                )
            return module.MoECommType.ALLGATHER
        return selected

    module.select_moe_comm_method = _select

    # Swap stale from-import bindings in already-imported caller modules
    # (same from-import stale-reference problem as fix_dual_attention).
    # Inspect ``mod.__dict__`` directly instead of ``getattr``: attribute
    # access on lazy modules (e.g. transformers' _LazyModule) fires their
    # ``__getattr__`` hook, logging an alias warning per module and
    # potentially triggering unwanted imports.
    import sys

    for mod in list(sys.modules.values()):
        if mod is None or mod is module:
            continue
        mod_dict = getattr(mod, "__dict__", None)
        if not isinstance(mod_dict, dict):
            continue
        if mod_dict.get("select_moe_comm_method") is _orig:
            mod.select_moe_comm_method = _select
            logger.info(
                "[fix_ep_zero_expert] Rebound select_moe_comm_method in %s",
                mod.__name__,
            )


# ===========================================================================
# Patch 3: MoERunner._maybe_add_zero_expert_output — add once, at the end
# ===========================================================================
# Upstream calls this at the very END of ``MoERunner.forward``, AFTER
# ``_maybe_reduce_final_output``'s TP/EP all-reduce.  That is the only
# correct place to add the zero-expert identity contribution (Problem 3).
# Here we hand the runner the real contribution stashed by Patch 0b2.
#
# This is the **consumer** side of the cross-patch coordination described
# in the module docstring.


def _ep_group_size_and_rank() -> tuple[int, int]:
    """Return ``(world_size, rank)`` of vllm's expert-parallel group.

    Falls back to ``(1, 0)`` when the group is unavailable (EP disabled or
    distributed state not initialised) — in that case no cross-rank token
    gather happens and the stash already matches the local token count.
    """
    try:
        from vllm.distributed.parallel_state import get_ep_group

        group = get_ep_group()
        return group.world_size, group.rank_in_group
    except Exception:
        return 1, 0


_slice_layout_warned = False


def _slice_zero_expert_output(
    stashed: torch.Tensor, ref: torch.Tensor
) -> torch.Tensor:
    """Extract this rank's zero-expert contribution from the gathered stash.

    With ALLGATHER MoE comm the stash is computed on the *gathered* tokens.
    ``all_gather`` concatenates equal-sized per-rank blocks in rank order,
    so rank r's tokens live at ``[r * block, r * block + n_local)`` with
    ``block = stashed.rows // ep_size`` — this stays correct even when
    prepare padded every rank to a uniform token count first (``block`` is
    the padded size then, not ``n_local``).

    The slice is cast to ``ref.dtype`` so the runner's final add does not
    upcast the hidden states to float32 (the stash is float32 because
    ``zero_experts_compute`` runs before apply casts topk weights).
    """
    global _slice_layout_warned
    n_local = ref.shape[0]
    total = stashed.shape[0]
    if total == n_local:
        return stashed.to(ref.dtype)
    if n_local == 0:
        # No local tokens: a scalar zero broadcasts to a no-op add.
        return torch.tensor(0.0, device=ref.device, dtype=ref.dtype)
    ep_size, ep_rank = _ep_group_size_and_rank()
    if ep_size > 1 and total % ep_size == 0:
        block = total // ep_size
        offset = ep_rank * block
        if offset + n_local <= total:
            return stashed[offset : offset + n_local].contiguous().to(ref.dtype)
    # Layout not understood (non-standard gather or uneven blocks).  Adding
    # the wrong rows would silently corrupt output; add zeros instead and
    # warn once — the identity contribution is lost for this call, which is
    # bounded noise, whereas a wrong slice is arbitrary garbage.
    if not _slice_layout_warned:
        _slice_layout_warned = True
        logger.warning(
            "[fix_ep_zero_expert] Cannot map gathered zero-expert stash "
            "(rows={}) to local output (rows={}, ep_size={}); adding zeros "
            "for this and later calls",
            total,
            n_local,
            ep_size,
        )
    return torch.zeros_like(ref)


def patch_moe_runner_zero_expert(module: object) -> None:
    MoERunner = module.MoERunner

    # Guard against repeated patching.
    if getattr(MoERunner, "_ez_maybe_patched", False):
        return
    MoERunner._ez_maybe_patched = True  # type: ignore[attr-defined]

    _orig_maybe = MoERunner._maybe_add_zero_expert_output

    def _maybe(self, result):
        global _pending_zero_expert_output
        if (
            isinstance(self.router, ZeroExpertRouter)
            and self.router.zero_expert_type is not None
        ):
            # Only redirect the runner addition when the native path in
            # ``apply`` actually handled zero experts (flag set by
            # ``patch_enable_native_zero_expert``).  Otherwise preserve
            # the original ``_zero_expert_output`` so the runner can add
            # it normally (future versions where the native path is off).
            if getattr(self.router, "_ez_native_handled", False):
                # ``result`` may be a plain Tensor (current vllm-ascend)
                # or a FusedExpertsResult (future versions).
                if isinstance(result, torch.Tensor):
                    ref = result
                elif hasattr(result, "routed_out"):
                    ref = result.routed_out
                else:
                    raise TypeError(
                        "[fix_ep_zero_expert] Unexpected result type: "
                        "%s.  Expected Tensor or object with 'routed_out'."
                        % type(result).__name__
                    )

                stashed = _pending_zero_expert_output
                _pending_zero_expert_output = None
                if stashed is not None:
                    self.router._zero_expert_output = _slice_zero_expert_output(
                        stashed, ref
                    )
                else:
                    # No stashed value (native path did not run this
                    # forward, e.g. non-MoE call).  Inject a scalar zero
                    # as a no-op add so the runner's
                    # ``assert zero_expert_output is not None`` holds.
                    self.router._zero_expert_output = torch.tensor(
                        0.0, device=ref.device, dtype=ref.dtype
                    )
        return _orig_maybe(self, result)

    MoERunner._maybe_add_zero_expert_output = _maybe
    logger.info(
        "[fix_ep_zero_expert] Patched MoERunner._maybe_add_zero_expert_output"
    )

def patch() -> None:
    """Apply all 4 EP zero-expert patches (call with explicit module targets)."""
    import vllm_ascend.ops.fused_moe.fused_moe_0_23_0 as _m0
    import vllm_ascend.ops.fused_moe.fused_moe as _fm
    import vllm_ascend.ascend_forward_context as _afc
    import vllm.model_executor.layers.fused_moe.runner.moe_runner as _mr

    logger.info("[fix_ep_zero_expert] Applying 4 EP patches...")
    patch_enable_native_zero_expert(_m0)
    patch_relocate_zero_expert_add(_fm)
    patch_force_allgather_comm(_afc)
    patch_moe_runner_zero_expert(_mr)
    logger.info("[fix_ep_zero_expert] All 4 patches applied")
