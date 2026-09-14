# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate."""

import torch
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs

# FusedRMSNormGated is a CustomOp, so the Ascend implementation is picked up
# through the OOT registration in `vllm_ascend.utils` rather than by import.
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.models.glm5next.config import Glm5NextConfig
from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.models.glm5next.ops.kda import KDA_MAX_RECURRENT_TOKENS, chunk_kda, recurrent_kda
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend


class _Glm5NextMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with multiple replicated output shards.

    Extends K3's ``_KimiGDNMergedColumnParallelLinear`` to support two
    replicated shards (f_a, g_a) instead of one. Pre-multiplies each
    replicated entry's output_size by tp_size so the per-rank shard
    divides back to the full size, and forces tp_rank=0 during weight
    loading for replicated shards.
    """

    # Owned by the base class; declared here so the temporary override in the
    # weight loaders below does not read the attribute before its type is known.
    tp_rank: int

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_ids: tuple[int, ...],
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = set(replicated_shard_ids)
        output_sizes = output_sizes.copy()
        for sid in self.replicated_shard_ids:
            output_sizes[sid] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    head_dim: int
    num_heads: int
    conv_size: int

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # conv_state width must include num_spec so the spec-decode conv update
        # (AscendC causal-conv with num_accepted_tokens) can
        # slide the window across the draft-verify tokens without reading past
        # the allocated width. Matches qwen_gdn_linear_attn.get_state_shape.
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: Glm5NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # KDA projections remain BF16 because fp8 checkpoints omit their scales.
        saved_quant_config = vllm_config.quant_config
        try:
            vllm_config.quant_config = None
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config

        if config.linear_head_dim != 128 or vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("GLM AscendC KDA requires BF16 activations and head_dim=128.")
        num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        if num_spec + 1 > KDA_MAX_RECURRENT_TOKENS:
            raise ValueError("GLM AscendC KDA supports at most seven speculative tokens.")
        if not 2 <= config.linear_conv_kernel_dim <= 4:
            raise ValueError("GLM AscendC causal-conv requires a kernel width in [2, 4].")
        if num_spec and config.linear_conv_kernel_dim != 4:
            raise ValueError("GLM AscendC causal-conv requires kernel width=4 for MTP.")
        if not -5 <= config.linear_lower_bound < 0:
            raise ValueError("GLM AscendC KDA requires linear_lower_bound in [-5, 0).")
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.conv_size = config.linear_conv_kernel_dim
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.tp_size)

        # Merge q, k, v, b, f_a, g_a projections into one GEMM (6→1 launches).
        # Order matches checkpoint's fused_qkvbfg_a_proj convention.
        # Shards 4 (f_a) and 5 (g_a) are replicated across TP ranks.
        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                projection_size,  # q (shard 0)
                projection_size,  # k (shard 1)
                projection_size,  # v (shard 2)
                self.num_heads,  # b (shard 3)
                self.head_dim,  # f_a (shard 4, replicated)
                self.head_dim,  # g_a (shard 5, replicated)
            ],
            replicated_shard_ids=(4, 5),
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvbfg_a",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32))

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)
        # Lazily-built merged q|k|v conv weight (built on first forward, after
        # weights are loaded). See _forward.
        self._merged_conv_weight: torch.Tensor | None = None

        self.A_log = nn.Parameter(torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32))
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Checkpoints store A_log as 1-D; the model parameter is 4-D.
        def _a_log_weight_loader(param, loaded_weight):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        self.A_log.weight_loader = _a_log_weight_loader

        # GLM-5.3-Flash uses a bounded sigmoid gate instead of the default
        # unbounded softplus gate.
        self.kda_lower_bound = config.linear_lower_bound
        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        self._conv_state_dim_first = is_conv_state_dim_first()

    def get_attn_backend(self):
        return AscendGDNAttentionBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 separate GEMMs).
        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )

        # Beta stays raw (bf16) here: the recurrent kernel sigmoids it in fp32
        # at load (SIGMOID_BETA), and only the chunked prefill path needs the
        # pre-computed fp32 sigmoid — computed lazily in _forward. Pure decode
        # / spec-verify steps then skip the separate sigmoid and its fp32
        # intermediate entirely.
        beta = beta_raw.unsqueeze(0)
        g1 = self.f_b_proj(f_a)[0]
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        g_proj_states = self.g_b_proj(g_a)[0]
        # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
        g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Keep the layer's dispatch outside piecewise graph compilation.
        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(1), -1)
        return self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            # Profile/warmup dummy runs may omit mamba-family metadata.
            core_attn_out.zero_()
            return
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata_narrowed.non_spec_state_indices_tensor  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        # Spec-decode metadata (all None when speculative decoding is disabled).
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = spec_sequence_masks is not None and num_spec_decodes > 0
        # Safe-gate checkpoints use the bounded sigmoid variant.
        lower_bound = self.kda_lower_bound
        constant_caches = self.kv_cache

        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # AscendC consumes [cache, state_len, dim]. Preserve the original storage.
        # Layout is process-global and resolved once at init (see __init__).
        if self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)

        # One merged short-conv over q|k|v instead of three separate calls. The
        # 1D conv is independent per channel, so concatenating q/k/v along the
        # channel dim preserves the independent q/k/v convolutions.
        # The merged weight is q|k|v conv weights concatenated;
        # built once and cached (params are fixed after load). conv_state is
        # already stored as the merged q|k|v state, so it is used directly.
        if self._merged_conv_weight is None:

            def _w(m):
                return m.weight.view(m.weight.size(0), m.weight.size(2))

            self._merged_conv_weight = (
                torch.cat(
                    [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                    dim=0,
                )
                .transpose(0, 1)
                .to(dtype=qkv_proj_states.dtype)
                .contiguous()
            )
        conv_weights = self._merged_conv_weight

        # Split projections / gating into spec (draft-verify) and non-spec token
        # groups when speculative decoding is active. Spec tokens carry
        # num_spec+1 recurrent-state columns each and are advanced with
        # num_accepted_tokens for rejection-sampling rollback; non-spec tokens
        # are one-per-request. Mirrors olmo_gdn_linear_attn.py. Projections are
        # [n, *] (token dim 0); g1/beta are [1, n, h, d] (token dim 1).
        if use_spec:
            # In a pure spec-verify step (no non-spec tokens) the metadata
            # builder sets spec_token_indx = arange(num_actual_tokens), making
            # the index_select calls below identity copies. Skip them on this
            # steady-state decode hot path. The outputs alias the inputs here;
            # the downstream conv/recurrent kernels read them without mutating
            # in place, so the aliasing is safe.
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0:
                qkv_spec = qkv_proj_states
                g1_spec = g1
                beta_spec = beta
            else:
                qkv_spec = qkv_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
            if non_spec_token_indx is not None and non_spec_token_indx.numel() > 0:
                qkv_ns = qkv_proj_states.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
            else:
                qkv_ns = g1_ns = beta_ns = None
        else:
            qkv_spec = g1_spec = beta_spec = None
            qkv_ns, g1_ns, beta_ns = qkv_proj_states, g1, beta

        # --- causal conv1d: spec (draft-verify) path ---
        if use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            conv_meta = attn_metadata_narrowed.spec_decode_metadata.spec_causal_conv1d
            qkv_spec = causal_conv1d(
                qkv_spec,
                conv_weights,
                conv_state,
                conv_meta.query_start_loc,
                conv_meta.cache_indices,
                run_mode=1,
                num_accepted_tokens=conv_meta.num_accepted_tokens,
            )
            q_spec, k_spec, v_spec = qkv_spec.split(self.local_projection_size, dim=-1)

        # --- causal conv1d: non-spec path (prefill or plain decode) ---
        q_ns = k_ns = v_ns = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert qkv_ns is not None
            conv_meta = attn_metadata_narrowed.non_spec_prefill_metadata.causal_conv1d
            qkv_ns = causal_conv1d(
                qkv_ns,
                conv_weights,
                conv_state,
                conv_meta.query_start_loc,
                conv_meta.cache_indices,
                run_mode=0,
                initial_state_mode=conv_meta.initial_state_mode,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_state_indices_tensor is not None
            conv_meta = attn_metadata_narrowed.non_spec_decode_metadata.causal_conv1d
            qkv_ns = causal_conv1d(
                qkv_ns,
                conv_weights,
                conv_state,
                conv_meta.query_start_loc,
                conv_meta.cache_indices,
                run_mode=1,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)

        def rearrange(x):
            return x.reshape(1, -1, self.local_num_heads, self.head_dim)

        core_attn_out.zero_()
        if use_spec:
            spec_output = recurrent_kda(
                rearrange(q_spec),
                rearrange(k_spec),
                rearrange(v_spec),
                g1_spec,
                beta_spec,
                recurrent_state,
                spec_query_start_loc[: num_spec_decodes + 1],
                spec_state_indices_tensor,
                self.A_log,
                self.dt_bias,
                lower_bound,
                num_accepted_tokens,
            )
            core_attn_out[0].index_copy_(0, spec_token_indx, spec_output[0])

        if q_ns is None:
            return
        q_ns, k_ns, v_ns = rearrange(q_ns), rearrange(k_ns), rearrange(v_ns)
        metadata = attn_metadata_narrowed
        decode_tokens = metadata.num_decode_tokens if metadata.num_prefills > 0 else q_ns.shape[1]
        output = None
        if metadata.num_decodes > 0:
            output = recurrent_kda(
                q_ns[:, :decode_tokens],
                k_ns[:, :decode_tokens],
                v_ns[:, :decode_tokens],
                g1_ns[:, :decode_tokens],
                beta_ns[:, :decode_tokens],
                recurrent_state,
                non_spec_query_start_loc[: metadata.num_decodes + 1],
                non_spec_state_indices_tensor,
                self.A_log,
                self.dt_bias,
                lower_bound,
            )
        if metadata.num_prefills > 0:
            prefill_output = chunk_kda(
                q_ns[:, decode_tokens:],
                k_ns[:, decode_tokens:],
                v_ns[:, decode_tokens:],
                g1_ns[:, decode_tokens:],
                beta_ns[:, decode_tokens:],
                recurrent_state,
                metadata.prefill_state_indices,
                metadata.prefill_has_initial_state,
                metadata.non_spec_prefill_metadata.chunk,
                self.A_log,
                self.dt_bias,
                lower_bound,
            )
            output = prefill_output if output is None else torch.cat((output, prefill_output), dim=1)
        assert output is not None
        if use_spec:
            core_attn_out[0].index_copy_(0, non_spec_token_indx, output[0])
        else:
            core_attn_out[0, : output.shape[1]].copy_(output[0])
