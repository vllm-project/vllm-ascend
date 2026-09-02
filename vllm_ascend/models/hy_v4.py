# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 model — Ascend NPU implementation.

The architecture (HYV4ForCausalLM + HYV4MTP) is identical to the NVIDIA
implementation, so we reuse the NVIDIA modules directly. The only NPU-side
adaptations are:

* The ``compute_logits`` autocast uses the current platform's device type
  instead of hard-coding ``"cuda"``.
* The CUDA-only FlashMLA-sink backend (``HYV4FlashMLASparseBackend``) is
  imported lazily inside the model; if it is not available on the NPU the
  ``learnable_sink`` is silently disabled by the existing sink-resolution
  logic in ``HYV4MLAAttention``. No code change is required to opt in.
* The sparse MLA backend is selected at runtime by
  ``vllm.v1.attention.selector.get_attn_backend``, so the NPU's DSA / SFA
  backend is used automatically.
"""

import torch
from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer
from vllm.models.hy_v4.nvidia.model import (
    HYV4DecoderLayer,
    HYV4ForCausalLM,
    HYV4Model,
    _normalize_hyv4_config,
)
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward, HYV4MoEFused
from vllm.models.hy_v4.nvidia.mtp import HYV4MTP
from vllm.platforms import current_platform


class _HyV4FusedQkvAProj(torch.nn.Module):
    """Virtual fused QKV-A projection for HYV4.

    HYV4 stores ``q_a_proj`` and ``kv_a_proj_with_mqa`` as two separate
    linear layers, but the NPU's MLA preprocessing expects a single
    ``fused_qkv_a_proj`` whose output is the concatenation of
    ``[q_lora_rank, kv_lora_rank + qk_rope_head_dim]``. This module
    reuses the two existing linear weights (no extra parameters) and
    concatenates their outputs on the fly.
    """

    def __init__(self, q_a_proj, kv_a_proj_with_mqa):
        super().__init__()
        # Hold references to the underlying linears so the fused
        # module is a thin alias and weight loading into the
        # originals flows through unchanged.
        self.q_a_proj = q_a_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        # Expose a ``.weight`` attribute that points to ``q_a_proj``'s
        # weight so any caller that introspects the fused module's
        # weight (e.g. weight prefetch) still finds something useful.
        self.weight = q_a_proj.weight

    def forward(self, x):
        q = self.q_a_proj(x)[0]
        kv = self.kv_a_proj_with_mqa(x)[0]
        return torch.cat([q, kv], dim=-1), None


def _patch_compute_logits_for_npu():
    """Replace the hard-coded ``device_type="cuda"`` autocast in
    ``HYV4ForCausalLM.compute_logits`` with the current platform's device
    type so the FP32 LM-head path works on NPU."""
    device_type = current_platform.device_type
    if device_type == "cuda":
        return  # nothing to do on CUDA hosts

    import torch
    from vllm.sequence import IntermediateTensors

    def compute_logits(self, hidden_states):  # type: ignore[no-redef]
        if self.enable_lm_head_fp32:
            with torch.autocast(device_type=device_type, enabled=False):
                logits = self.logits_processor(self.lm_head, hidden_states.to(torch.float32))
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)

        if getattr(self.config, "soft_logits_capping", False):
            soft_cap = self.config.soft_logits_capping_logits
            logits = soft_cap * torch.nn.functional.tanh(logits / soft_cap)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    HYV4ForCausalLM.compute_logits = compute_logits
    HYV4ForCausalLM.forward = forward


def _patch_mla_attention_init_and_forward_for_npu():
    """Replace ``HYV4MLAAttention.__init__`` and ``forward`` on NPU.

    The HYV4 attention constructs a vllm-core ``MLAAttention`` instance
    (``self.mla_attn``) and calls it with already-projected ``q``,
    ``kv_c_normed`` and ``k_pe`` tensors. Ascend's MLA implementations
    (``AscendMultiHeadLatentAttention`` / ``AscendSFAImpl``) deliberately
    raise ``NotImplementedError`` from the ``forward_mha`` /
    ``forward_mqa`` entry points and only support the unified
    ``mla_forward`` op, which expects the *raw* ``hidden_states`` and a
    layer wrapper registered in ``forward_context.no_compile_layers``.

    The cleanest port is to mirror ``vllm-ascend/patch/worker/
    patch_deepseek_v2.py``: after the HYV4 attention finishes building
    its MLA modules, swap ``self.mla_attn`` for a
    ``MultiHeadLatentAttentionWrapper`` that calls the NPU's MLA
    forward, and replace the per-layer ``forward`` so the wrapper
    runs the MLA preprocessing (fused-QKV-A → RoPE → MLA) and the
    o_proj itself. The HYV4-specific ``gated_mla`` post-processing
    still runs after the wrapper returns.
    """
    from vllm.model_executor.layers.mla import (
        MLAModules,
        MultiHeadLatentAttentionWrapper,
    )
    from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention

    original_init = HYV4MLAAttention.__init__

    def _npu_init(self, *args, **kwargs):  # type: ignore[no-redef]
        # Resolve cache/quant configs before delegating so the wrapper
        # has everything it needs. ``quant_config`` may legitimately be
        # ``None`` (e.g. when the model is served in bf16), so we only
        # treat the cache_config as required.
        cache_config = kwargs.get("cache_config")
        quant_config = kwargs.get("quant_config")
        if cache_config is None:
            # kwargs may use ``cache_config``; check positional too.
            for name, value in zip(
                (
                    "vllm_config",
                    "config",
                    "hidden_size",
                    "num_heads",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                    "v_head_dim",
                    "q_lora_rank",
                    "kv_lora_rank",
                    "max_position_embeddings",
                    "cache_config",
                    "quant_config",
                ),
                args,
            ):
                if name == "cache_config":
                    cache_config = value
                elif name == "quant_config" and quant_config is None:
                    quant_config = value
        if cache_config is None:
            vllm_config = kwargs.get("vllm_config")
            if vllm_config is None and len(args) >= 1:
                vllm_config = args[0]
            if vllm_config is not None:
                cache_config = vllm_config.cache_config
                if quant_config is None:
                    quant_config = vllm_config.quant_config
        if cache_config is None:
            raise RuntimeError(
                f"NPU HYV4 attention could not recover cache_config "
                f"(cache_config={cache_config}, quant_config={quant_config}, "
                f"args={args}, kwargs_keys={list(kwargs.keys())}) for "
                f"MultiHeadLatentAttentionWrapper construction."
            )
        # Stash for later use if the original constructor overwrites
        # ``self.quant_config`` (it does, so this is only a backup).
        kwargs["cache_config"] = cache_config
        kwargs["quant_config"] = quant_config
        # The NPU's ``AscendMultiHeadLatentAttention`` bakes
        # ``o_proj`` into its impl, so the wrapper returns the
        # post-projection tensor of shape ``[N, hidden_size]``.
        # HYV4's ``gated_mla`` post-processing expects the
        # pre-projection tensor ``[N, num_heads * v_head_dim]``,
        # so it is applied INSIDE the SFA impl (see
        # ``gated_mla_linear_gate`` binding below) rather than
        # after the wrapper.
        #
        # The GPU ``HYV4MLAAttention.__init__`` is the only place
        # that builds ``self.linear_gate`` and
        # ``self.learnable_sink_param``; we delegate to it but
        # mask two NPU-incompatible code paths:
        #   * The sparse-backend probe at the top of the init
        #     (``get_attn_backend(..., use_sparse=True, ...)``)
        #     raises ``RuntimeError`` when no NPU sparse backend
        #     is registered. We temporarily clear the module-level
        #     ``_SPARSE_LAYER_TYPES`` set so every layer is
        #     treated as dense; the SFA impl does its own sparse
        #     dispatch afterwards.
        #   * ``learnable_sink`` is not touched — its probe is
        #     already exception-safe.
        from vllm.models.hy_v4.nvidia import attention as _gpu_attn_mod

        config = kwargs.get("config")
        if config is None and len(args) >= 2:
            config = args[1]

        # The GPU ``HYV4MLAAttention.__init__`` only creates the
        # ``Indexer`` (and sets ``self.is_sparse = True``) when
        # ``layer_types[layer_id] in _SPARSE_LAYER_TYPES`` AND the
        # probe ``get_attn_backend(..., use_sparse=True, ...)`` returns
        # a valid backend. On NPU the latter raises (no NPU
        # ``use_sparse=True`` selector key), so the original code
        # short-circuited by clearing ``_SPARSE_LAYER_TYPES`` to
        # ``()`` — that left every layer in dense mode and the
        # ``Indexer`` was never built, so the NPU had no sparse
        # attention at all (P0-1 in the precision bug list).
        #
        # We now (a) keep the original ``_SPARSE_LAYER_TYPES`` so
        # ``requested_sparse``/``create_indexer`` are computed
        # correctly, and (b) monkey-patch ``get_attn_backend`` in the
        # GPU module to return a sentinel class for the ``use_sparse
        # =True`` probe so the ``if self.is_sparse: get_attn_backend
        # (...)`` sanity check does not raise on NPU. The sentinel
        # class is never instantiated — the NPU path replaces
        # ``self.mla_attn`` with a ``MultiHeadLatentAttentionWrapper``
        # anyway.
        class _SparseBackendSentinel:
            """Stand-in returned by ``get_attn_backend`` during NPU
            init so the GPU ``HYV4MLAAttention.__init__`` sparse
            probe succeeds. Not used at runtime — the NPU replaces
            ``self.mla_attn`` with its own wrapper.

            The GPU ``MLAAttention.__init__`` calls a few class-level
            helpers on the backend (``is_mla``, ``get_name``,
            ``get_impl_cls``); we satisfy those probes and force
            ``supports_sink()`` to True so the GPU sink-resolution
            path short-circuits and uses the sentinel directly."""

            @staticmethod
            def get_name() -> str:
                return "ASCEND_SFA_SENTINEL"

            @staticmethod
            def is_mla() -> bool:
                return True

            @staticmethod
            def supports_sink() -> bool:
                return True

            @staticmethod
            def get_impl_cls():
                return None

        def _patched_get_attn_backend(*_a, **_kw):
            return _SparseBackendSentinel

        def _patched_resolve_sink_backend(self, kv_cache_dtype):
            # On NPU the GPU sink-resolution path is meaningless; the
            # NPU's SFA impl applies the per-head sink via its own
            # ``_apply_learnable_sink_rescale`` after the fact.
            # Returning None keeps the GPU init code path valid
            # (``enable_sink=False`` -> the sink param is still
            # created as a buffer, just not fed into the GPU
            # ``MLAAttention`` impl) so we can read it back in the
            # NPU wrapper and attach it to the SFA impl.
            return None

        saved_sparse_layer_types = _gpu_attn_mod._SPARSE_LAYER_TYPES
        # (a) keep the layer types so requested_sparse evaluates
        # correctly.
        # (b) patch get_attn_backend to swallow the probe.
        # (c) patch _resolve_sink_backend so the GPU sink backend
        # selection does not try to use the sentinel as a real
        # backend.
        saved_get_attn_backend = _gpu_attn_mod.get_attn_backend
        saved_resolve_sink = HYV4MLAAttention._resolve_sink_backend
        _gpu_attn_mod.get_attn_backend = _patched_get_attn_backend
        HYV4MLAAttention._resolve_sink_backend = _patched_resolve_sink_backend
        try:
            original_init(self, *args, **kwargs)
        finally:
            _gpu_attn_mod.get_attn_backend = saved_get_attn_backend
            _gpu_attn_mod._SPARSE_LAYER_TYPES = saved_sparse_layer_types
            HYV4MLAAttention._resolve_sink_backend = saved_resolve_sink

        # The GPU ``HYV4MLAAttention.__init__`` created a vllm-core
        # ``MLAAttention`` under ``{prefix}.attn`` and registered it in
        # ``static_forward_context``. That module is discarded below when
        # ``self.mla_attn`` is replaced with the NPU wrapper, but its
        # registration persists, which would leave TWO AttentionLayerBase
        # entries per decoder layer (legacy ``.attn`` + live
        # ``.npu_attn.attn``). On main this duplicates the KV cache
        # allocation and corrupts the Mooncake PD transfer metadata
        # (``_build_kv_group2layeridx`` maps both names to the same layer
        # index). Remove the stale registration so only the live NPU
        # wrapper's inner MLAAttention stays registered.
        from vllm.config import get_current_vllm_config

        try:
            _compilation_config = get_current_vllm_config().compilation_config
        except AssertionError:
            _compilation_config = None
        if _compilation_config is not None:
            _legacy_layer_name = f"{self.prefix}.attn"
            _compilation_config.static_forward_context.pop(_legacy_layer_name, None)

        # ``f8E4M3FN``); the wrapper accepts ``is_sparse=False`` and
        # ``skip_topk=True`` and will simply skip the indexer branch.
        # NOTE: re-enabled sparse path on NPU. The vllm indexer's
        # ``per_token_group_quant_fp8`` kernel is bypassed by the
        # ``IndexerWrapper`` (no-op forward); the NPU's SFA impl
        # performs the actual top-k selection in
        # ``indexer_select_post_process`` using the indexer's weights
        # (``wq_b``, ``wk_weights_proj``, ``k_norm``) and a float32
        # fallback. The ``is_sparse`` / ``skip_topk`` flags computed
        # by ``original_init`` are preserved so the SFA impl matches
        # the GPU's "full" / "shared" indexer layout.
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            # The NPU's MLA preprocessing needs a single
            # ``fused_qkv_a_proj`` whose output is the concatenation
            # of ``[q_lora_rank, kv_lora_rank + qk_rope_head_dim]``.
            # HYV4 keeps these as two separate linears; wrap them
            # in a virtual fused module so the NPU can consume
            # them without duplicating weights.
            fused_qkv_a_proj=_HyV4FusedQkvAProj(self.q_a_proj, self.kv_a_proj_with_mqa)
            if self.q_lora_rank is not None and self.q_a_proj is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            # Pass through the vllm indexer for sparse layers; the
            # ``AscendMultiHeadLatentAttention.__init__`` wraps it in
            # an ``IndexerWrapper`` so ``indexer(...)`` becomes a
            # no-op and only the weights (used by the SFA impl's
            # ``indexer_select_post_process``) survive.
            indexer=self.indexer if getattr(self, "is_sparse", False) else None,
            indexer_rotary_emb=self.indexer_rope_emb if getattr(self, "is_sparse", False) else None,
            is_sparse=bool(getattr(self, "is_sparse", False)),
            # The vllm ``Indexer`` stores the topk sharing buffer on
            # itself. Re-expose it on ``MLAModules`` so the SFA impl
            # can read/write the buffer for ``skip_topk`` layers
            # (full indexer layers expose it via ``self.indexer``;
            # shared indexer layers do not build an indexer so we
            # fall back to the buffer that ``HYV4Model`` created
            # and threaded through every decoder layer's
            # ``__init__``).
            topk_indices_buffer=(
                self.indexer.topk_indices_buffer
                if (getattr(self, "is_sparse", False) and self.indexer is not None)
                else kwargs.get("topk_indices_buffer")
                if getattr(self, "is_sparse", False)
                else None
            ),
        )
        # ``prefix`` is the layer's registered name; the wrapper needs
        # a unique prefix (not the same as the original ``MLAAttention``)
        # so vllm's module registry does not raise a duplicate-name
        # error. We append ``.npu_attn`` so the wrapper's own
        # submodules live under ``model.layers.N.self_attn.npu_attn``
        # while the legacy ``MLAAttention`` (whose forward is now
        # bypassed) keeps its original ``.attn`` namespace.
        attn_prefix = f"{self.prefix}.npu_attn"
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            attn_prefix,
            # Respect the GPU-computed skip_topk flag so "shared"
            # indexer layers reuse the top-k buffer instead of
            # recomputing. Sparse "full" layers keep
            # ``skip_topk=False``; dense layers also have
            # ``skip_topk=False`` (their indexer is None).
            skip_topk=bool(getattr(self, "skip_topk", False)),
        )
        # Bind the per-head learnable attention sink (if any) to the NPU
        # SFA impl so it can rescale the op's output to match the GPU
        # ``flash_mla_sparse_fwd`` kernel. The original GPU
        # ``HYV4MLAAttention.__init__`` created ``learnable_sink_param``
        # and passed it as ``sinks=...`` to its MLA impl; the NPU wrapper
        # doesn't accept that kwarg, so we attach it directly to the impl
        # that the wrapper just instantiated.
        sink = getattr(self, "learnable_sink_param", None)
        if sink is not None:
            inner_impl = getattr(self.mla_attn, "mla_attn", None)
            inner_impl_impl = getattr(inner_impl, "impl", None) if inner_impl is not None else None
            if inner_impl_impl is not None:
                inner_impl_impl.learnable_sink_param = sink
                # One-shot log so we can confirm the binding actually took
                # effect at startup. Use the standard vllm logger so the
                # message lands in the worker log alongside the existing
                # sink-unavailable warning.
                from vllm.logger import logger as _vllm_logger

                _vllm_logger.info_once(
                    "HYV4 NPU sink bound: dtype=%s shape=%s impl=%s",
                    sink.dtype,
                    tuple(sink.shape),
                    type(inner_impl_impl).__name__,
                )
            else:
                from vllm.logger import logger as _vllm_logger

                _vllm_logger.warning_once(
                    "HYV4 NPU sink present but inner SFA impl not found; learnable_sink will be ignored. layer=%s",
                    self.prefix,
                )

        # Bind the gated_mla linear_gate to the SFA impl so it can apply
        # ``sigmoid(linear_gate(hidden_states))`` to the pre-o_proj
        # attention output (matching the GPU
        # ``HYV4MLAAttention._indexer_and_attn`` post-processing). The
        # gate is constructed in the GPU ``__init__`` (when
        # ``config.gated_mla=True``) but the NPU wrapper discards the
        # kwarg, so we attach it directly here.
        gate = getattr(self, "linear_gate", None)
        if gate is not None:
            inner_impl = getattr(self.mla_attn, "mla_attn", None)
            inner_impl_impl = getattr(inner_impl, "impl", None) if inner_impl is not None else None
            if inner_impl_impl is not None:
                inner_impl_impl.gated_mla_linear_gate = gate
                inner_impl_impl.gated_mla_gating_type = getattr(self.config, "gating_type", "headwise")
                from vllm.logger import logger as _vllm_logger

                _vllm_logger.info_once(
                    "HYV4 NPU gated_mla bound: gating_type=%s impl=%s",
                    inner_impl_impl.gated_mla_gating_type,
                    type(inner_impl_impl).__name__,
                )

    def _npu_forward(self, positions, hidden_states, llama_4_scaling=None):  # type: ignore[no-redef]
        # Delegate MLA + o_proj to the NPU's
        # ``AscendMultiHeadLatentAttention`` (instantiated via the
        # ``MultiHeadLatentAttentionWrapper`` PluggableLayer). The
        # wrapper's impl bakes ``o_proj`` into its forward and
        # returns the post-projection tensor of shape
        # ``[N, hidden_size]``. ``gated_mla`` and ``learnable_sink``
        # are both applied INSIDE the SFA impl between attention
        # and o_proj, so this wrapper-level forward stays a
        # straight pass-through.
        return self.mla_attn(positions, hidden_states, llama_4_scaling)

    HYV4MLAAttention.__init__ = _npu_init
    HYV4MLAAttention.forward = _npu_forward


def _patch_mla_attention_forward_for_npu():
    """DEPRECATED: superseded by
    ``_patch_mla_attention_init_and_forward_for_npu`` which replaces both
    ``HYV4MLAAttention.__init__`` and ``HYV4MLAAttention.forward``. Kept
    here as a no-op so older code that imports it still works.
    """
    return


_patch_compute_logits_for_npu()
_patch_mla_attention_init_and_forward_for_npu()


def _disable_hyv4_indexer_for_npu():
    """DEPRECATED: kept as a no-op for backward compatibility.

    The NPU's SFA impl now runs the top-k selection in
    ``indexer_select_post_process`` (using the vllm ``Indexer``'s
    weights as a parameter holder and a float32 fallback for the
    FP8 quant kernel), and ``_npu_init`` preserves the
    ``is_sparse`` / ``skip_topk`` flags computed by
    ``original_init``. The NVIDIA indexer's ``forward`` is wrapped
    by ``IndexerWrapper`` (no-op), so no FP8 Triton kernel is
    actually invoked on the NPU path. This function used to force
    every layer into dense attention; that is no longer needed.
    """
    return


_disable_hyv4_indexer_for_npu()


AscendHYV4ForCausalLM = HYV4ForCausalLM
AscendHYV4MTP = HYV4MTP

__all__ = [
    "AscendHYV4ForCausalLM",
    "AscendHYV4MTP",
    "HYV4ForCausalLM",
    "HYV4DecoderLayer",
    "HYV4Model",
    "HYV4MTP",
    "HYV4MLAAttention",
    "HYV4HCLayer",
    "HYV4HCHeadLayer",
    "HYV4FeedForward",
    "HYV4MoEFused",
    "_normalize_hyv4_config",
]
