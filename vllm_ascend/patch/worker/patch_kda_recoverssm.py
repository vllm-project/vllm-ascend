# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire the upstream deferred RecoverSSM onto Ascend's GLM KDA path.

The upstream deferred RecoverSSM (PR #66030) lives in
``vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm`` and is normally
installed by the Kimi-K3 *specialized decode* runtime, which Ascend's GLM
model does not use.  This patch:

1.  Replaces the stock commit context with the Ascend one (immediate path) —
    the existing behaviour.
2.  Patches ``KDADeferredCommitContext`` so its ``create`` skips the CuTe
    probe and its ``commit`` / ``flush`` / ``_fold`` use the Ascend Triton
    kernels (which already support ``record_rows`` indexing from tasks 1-3).
3.  Hooks ``init_asecnd_model_state`` to call the upstream
    ``install_model_state`` (swaps ``RecoverSSMState`` →
    ``DeferredRecoverSSMState``, wraps ``add_request`` / ``prepare_attn``)
    and marks each KDA layer ``deferred_recoverssm = True``.

The KDA layer's ``forward`` (in ``glm5next/kda.py``) already checks
``attn_metadata_narrowed.recoverssm_deferred`` to flush pending replays and
pass ``record_indices`` to ``kda_recoverssm_verify``.
"""

from __future__ import annotations

from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm

# Set True once the deferred machinery is installed.
_deferred_installed = False


def patch_kda_recoverssm(vllm_config):
    if not use_glm_recoverssm(vllm_config):
        return

    # 1. Immediate path: swap the stock commit context for the Ascend one.
    import vllm.models.kimi_k3.nvidia.ops.recoverssm as recoverssm

    from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext

    recoverssm.KDARecoverSSMCommitContext = KDARecoverSSMCommitContext

    # 2. Deferred path: patch KDADeferredCommitContext to use Ascend kernels.
    _install_deferred_commit_context()

    # 3. Hook model-state creation to install the deferred state + layers.
    _hook_model_state_init()

    global _deferred_installed
    _deferred_installed = True


# ────────────────────── DeferredCommitContext patch ───────────────────────


def _install_deferred_commit_context() -> None:
    """Patch ``KDADeferredCommitContext`` so it works on Ascend.

    The upstream class inherits from the *stock* ``KDARecoverSSMCommitContext``
    (bound at import time, before our patch runs), and its ``_fold`` calls the
    CuTe ``sdec_commit.commit_f32``.  We replace ``create``, ``commit``,
    ``flush`` and ``_fold`` with methods that delegate to the Ascend
    ``KDARecoverSSMCommitContext`` (which already has full deferred support
    from tasks 1-3, including ``record_rows`` indexing and the Ascend-specific
    conv/commit kernels).
    """
    from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.context import (
        FP32_DEFERRED,
        KDADeferredCommitContext,
    )

    from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext

    @classmethod
    def create(cls, layers, *, spec_query_len, max_num_reqs, backend=FP32_DEFERRED):
        """Ascend: skip the CuTe probe — always use the Ascend Triton commit.

        The upstream ``create`` probes ``sdec_commit.supported`` (CuTe DSL),
        which is not available on NPU.  We build via the Ascend
        ``KDARecoverSSMCommitContext.create`` (which reads ``kv_cache[2:3]``
        for records instead of ``recoverssm_records``) and tag the backend
        ``FP32_DEFERRED`` so the deferred commit path activates.
        """
        context = KDARecoverSSMCommitContext.create(
            layers,
            spec_query_len=spec_query_len,
            max_num_reqs=max_num_reqs,
            backend="triton",
        )
        # Re-class the instance so ``isinstance(ctx, KDADeferredCommitContext)``
        # holds (the metadata builder's ``build`` checks this).
        context.__class__ = cls
        context.backend = FP32_DEFERRED  # type: ignore[attr-defined]
        return context

    # Delegate commit / flush / _fold to the Ascend methods.  The Ascend
    # ``commit`` already handles the ``pending`` / ``eager`` / ``record_rows``
    # deferred branch; ``_fold`` uses the Ascend conv/commit kernels.
    KDADeferredCommitContext.create = create  # type: ignore[assignment]
    KDADeferredCommitContext.commit = KDARecoverSSMCommitContext.commit  # type: ignore[assignment]
    KDADeferredCommitContext.flush = KDARecoverSSMCommitContext.flush  # type: ignore[assignment]
    KDADeferredCommitContext._fold = KDARecoverSSMCommitContext._fold  # type: ignore[assignment]


# ────────────────────── Model-state hook ──────────────────────────────────


def _hook_model_state_init() -> None:
    """Wrap ``init_asecnd_model_state`` to install the deferred state + layers.

    After the Ascend model state is created (which sets ``self.recoverssm =
    RecoverSSMState()`` via the upstream ``MambaHybridModelState.__init__``),
    we call the upstream ``install_model_state`` to swap it for a
    ``DeferredRecoverSSMState`` and wrap ``add_request`` / ``prepare_attn``.

    We also mark each KDA decoder layer ``deferred_recoverssm = True`` so the
    metadata builder's ``get_context`` knows to create a
    ``KDADeferredCommitContext``.  Unlike upstream, Ascend does not reallocate
    records (they live in ``kv_cache[2:3]``), so this is just a flag flip.
    """
    import vllm_ascend.worker.v2.model_states as model_states_mod

    from vllm_ascend.models.glm5next.recoverssm_config import is_glm_recoverssm_model

    base_init = model_states_mod.init_asecnd_model_state

    def init_asecnd_model_state(vllm_config, model, encoder_cache, device):
        state = base_init(vllm_config, model, encoder_cache, device)

        if not is_glm_recoverssm_model(vllm_config):
            return state
        if not use_glm_recoverssm(vllm_config):
            return state
        if getattr(state, "recoverssm", None) is None:
            return state

        # Install the deferred RecoverSSM on the model runner state.
        from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.state import (
            install_model_state,
        )

        install_model_state(state)

        # Mark each KDA layer as deferred.  Ascend keeps records in
        # ``kv_cache[2:3]`` (no reallocation), so we only set the flag that
        # the metadata builder's ``get_context`` checks via
        # ``layers[0].recoverssm_backend``.
        language_model = model
        if hasattr(model, "get_language_model"):
            language_model = model.get_language_model()
        layers = getattr(language_model, "layers", None)
        start = getattr(language_model, "start_layer", 0)
        end = getattr(language_model, "end_layer", None)
        if layers is not None:
            decoder_layers = layers[start:end] if end is not None else layers
            for decoder in decoder_layers:
                attn = getattr(decoder, "self_attn", None)
                if attn is None:
                    continue
                if not getattr(attn, "use_recoverssm", False):
                    continue
                if getattr(attn, "recoverssm_backend", "triton") != "fp32_deferred":
                    continue
                attn.deferred_recoverssm = True

        return state

    model_states_mod.init_asecnd_model_state = init_asecnd_model_state