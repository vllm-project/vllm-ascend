# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate."""

import torch
from torch import nn
from transformers import Glm5NextTextConfig
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import divide, get_pcp_group
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
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.models.glm5next.ops.kda import KDA_MAX_RECURRENT_TOKENS, chunk_kda, recurrent_kda
from vllm_ascend.models.glm5next.recoverssm_config import use_glm_recoverssm
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBackend
from vllm_ascend.ops.triton.kda.recoverssm import kda_recoverssm_verify


class Glm5NextGDNHostMetadataBackend(AscendGDNHostMetadataBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        # The GLM KDA layer gathers PCP prefill tokens before its head-sharded
        # recurrent update and restores the PCP-local output afterward.
        return True


def get_kda_parallel_size(parallel_config: ParallelConfig) -> int:
    """Use PCP head sharding only when tensor parallelism is disabled."""
    # TODO: Support joint TP * PCP head partitioning for projections and state.
    if parallel_config.tensor_parallel_size == 1:
        return parallel_config.prefill_context_parallel_size
    return parallel_config.tensor_parallel_size


class PCPColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel projection using PCP ranks and collectives."""

    def __init__(self, input_size: int, output_size: int, **kwargs) -> None:
        pcp_group = get_pcp_group()
        super().__init__(
            input_size,
            output_size,
            tp_rank=pcp_group.rank_in_group,
            tp_size=pcp_group.world_size,
            **kwargs,
        )
        self.tp_group = pcp_group

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output = self.quant_method.apply(self, input_, bias)
        if self.gather_output and self.tp_size > 1:
            output = self.tp_group.all_gather(output, dim=-1)
        if not self.return_bias:
            return output
        return output, self.bias if self.skip_bias_add else None


class PCPRowParallelLinear(RowParallelLinear):
    """Row-parallel projection reducing head contributions over PCP."""

    def __init__(self, input_size: int, output_size: int, **kwargs) -> None:
        pcp_group = get_pcp_group()
        # The pinned RowParallelLinear has no rank/size overrides. Allocate
        # the local input shard, then bind its inherited weight loaders to PCP.
        super().__init__(divide(input_size, pcp_group.world_size), output_size, disable_tp=True, **kwargs)
        self.input_size = input_size
        self.disable_tp = False
        self.tp_size = pcp_group.world_size
        self.tp_rank = pcp_group.rank_in_group
        self.tp_group = pcp_group
        self.update_param_tp_status()

    def forward(self, input_):
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = input_.chunk(self.tp_size, dim=-1)[self.tp_rank].contiguous()
        bias = None if self.tp_rank > 0 or self.skip_bias_add else self.bias
        output = self.quant_method.apply(self, input_parallel, bias)
        if self.reduce_results and self.tp_size > 1:
            output = self.tp_group.all_reduce(output)
        if not self.return_bias:
            return output
        return output, self.bias if self.skip_bias_add else None


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


class PCPGlm5NextMergedColumnParallelLinear(_Glm5NextMergedColumnParallelLinear, PCPColumnParallelLinear):
    """PCP-sharded Q/K/V/b with replicated low-rank f_a/g_a shards."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_ids: tuple[int, ...],
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = set(replicated_shard_ids)
        self.output_sizes = output_sizes.copy()
        for sid in self.replicated_shard_ids:
            self.output_sizes[sid] *= tp_size
        # MergedColumnParallelLinear's constructor hardcodes the TP group.
        # Retain its shard loaders while allocating through the PCP column layer.
        PCPColumnParallelLinear.__init__(self, input_size, sum(self.output_sizes), **kwargs)


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    use_contiguous_conv_indices = True
    use_shared_conv_output = True

    tp_size: int
    head_dim: int
    num_heads: int
    conv_size: int

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        dtypes = MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )
        if self.use_recoverssm:
            return MambaStateDtypeCalculator.append_kda_recoverssm_record(dtypes, self.model_config.dtype)
        return dtypes

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # conv_state width must include num_spec so the spec-decode conv update
        # (AscendC causal-conv with num_accepted_tokens) can
        # slide the window across the draft-verify tokens without reading past
        # the allocated width. Matches qwen_gdn_linear_attn.get_state_shape.
        shapes = MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )
        if self.use_recoverssm:
            return MambaStateShapeCalculator.append_kda_recoverssm_record(
                shapes, self.num_heads, self.head_dim, self.tp_size, self.num_spec + 1
            )
        return shapes

    def __init__(
        self,
        config: Glm5NextTextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # KDA projections remain BF16 because fp8 checkpoints omit their scales.
        self.use_recoverssm = use_glm_recoverssm(vllm_config)
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
        self.pcp_group = get_pcp_group()
        self.pcp_size = self.pcp_group.world_size
        self.pcp_rank = self.pcp_group.rank_in_group
        self.use_pcp_head_sharding = self.tp_size == 1 and self.pcp_size > 1
        column_linear = ColumnParallelLinear
        merged_linear = _Glm5NextMergedColumnParallelLinear
        row_linear = RowParallelLinear
        if self.use_pcp_head_sharding:
            self.tp_size = get_kda_parallel_size(vllm_config.parallel_config)
            self.tp_rank = self.pcp_rank
            column_linear = PCPColumnParallelLinear
            merged_linear = PCPGlm5NextMergedColumnParallelLinear
            row_linear = PCPRowParallelLinear

        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.conv_size = config.linear_conv_kernel_dim
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.tp_size)

        # Merge q, k, v, b, f_a, g_a projections into one GEMM (6→1 launches).
        # Order matches checkpoint's fused_qkvbfg_a_proj convention.
        # Shards 4 (f_a) and 5 (g_a) are replicated across head-sharding ranks.
        self.in_proj_qkvbfg_a = merged_linear(
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

        self.f_b_proj = column_linear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32))

        set_weight_attrs(self.dt_bias, {"weight_loader": self._dt_bias_weight_loader})

        self.q_conv1d = column_linear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = column_linear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = column_linear(
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
        set_weight_attrs(self.A_log, {"weight_loader": self._a_log_weight_loader})

        self.g_b_proj = column_linear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        # FLA imports can initialize an NPU context through capability queries.
        # Resolve and register the Ascend subclass after worker device selection.
        # TODO: Restore eager import/registration once supported vLLM FLA or
        # Triton Ascend versions make these queries safe before device selection.
        from vllm_ascend.utils import register_ascend_fla_customop

        register_ascend_fla_customop()
        from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = row_linear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=saved_quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # GLM-5.3-Flash uses a bounded sigmoid gate instead of the default
        # unbounded softplus gate.
        self.kda_lower_bound = config.linear_lower_bound
        self.gate_lower_bound = self.kda_lower_bound
        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        self._conv_state_dim_first = is_conv_state_dim_first()

    def _dt_bias_weight_loader(self, param, loaded_weight):
        shard_size = param.shape[0]
        default_weight_loader(param, loaded_weight.narrow(0, self.tp_rank * shard_size, shard_size))

    def _a_log_weight_loader(self, param, loaded_weight):
        # Checkpoints store A_log as 1-D; the model parameter is 4-D.
        if loaded_weight.dim() == 1:
            loaded_weight = loaded_weight.view(1, 1, -1, 1)
        shard_size = param.shape[2]
        default_weight_loader(param, loaded_weight.narrow(2, self.tp_rank * shard_size, shard_size))

    def get_attn_backend(self):
        return Glm5NextGDNHostMetadataBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata_raw = get_forward_context().attn_metadata
        attn_metadata = attn_metadata_raw.get(self.prefix) if isinstance(attn_metadata_raw, dict) else None
        pcp_context = getattr(attn_metadata, "pcp_context", None)
        has_pcp_prefill = pcp_context is not None and bool(pcp_context.global_batch.is_prefilling_np.any())
        local_num_tokens = hidden_states.size(0)
        if has_pcp_prefill:
            assert pcp_context is not None
            assert pcp_context.padded_gather_idx is not None
            # PCP assigns the first and last chunks of a prefill to the same
            # rank. Reconstruct causal order before advancing the head-sharded
            # recurrent state. With TP=1 each PCP rank owns different heads.
            hidden_states = self.pcp_group.all_gather(hidden_states, dim=0)[pcp_context.hidden_restore_idx]
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
        output = self.o_proj(core_attn_out)[0]
        if has_pcp_prefill:
            assert pcp_context is not None
            if pcp_context.padded_gather_idx.numel() != self.pcp_size * local_num_tokens:
                raise RuntimeError("KDA PCP token mapping does not match the rank-local model input.")
            local_start = self.pcp_rank * local_num_tokens
            local_indices = pcp_context.padded_gather_idx[local_start : local_start + local_num_tokens]
            output = output[local_indices]
        return output

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
        wait_for_kv_layer_from_connector(self.prefix)
        record_attention_compute_start()
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata_narrowed.non_spec_state_indices_tensor  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        # Spec-decode metadata (all None when speculative decoding is disabled).
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = num_spec_decodes > 0 and spec_state_indices_tensor is not None
        # Safe-gate checkpoints use the bounded sigmoid variant.
        lower_bound = self.kda_lower_bound
        constant_caches = self.kv_cache

        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state, *records = constant_caches
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
                output_buffer=conv_meta.output_buffer,
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
                output_buffer=conv_meta.output_buffer,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)

        def rearrange(x):
            return x.reshape(1, -1, self.local_num_heads, self.head_dim)

        # Pure decode writes the whole captured buffer, including padding.
        direct_output = (
            not use_spec and attn_metadata_narrowed.num_prefills == 0 and attn_metadata_narrowed.num_decodes > 0
        )
        # Prefill copies its output and speculative dispatch scatters both token
        # groups. Recurrent writeback handles decode padding without a pre-clear.
        if use_spec:
            if self.use_recoverssm:
                correction_cache, kg_cache = records
                # One paged checkpoint per request; verification only records
                # compact updates. MRV2 commits after rejection sampling.
                spec_output = kda_recoverssm_verify(
                    rearrange(q_spec),
                    rearrange(k_spec),
                    rearrange(v_spec),
                    g1_spec,
                    beta_spec,
                    self.A_log.reshape(-1),
                    self.dt_bias,
                    lower_bound,
                    recurrent_state,
                    correction_cache,
                    kg_cache,
                    spec_query_start_loc,
                    spec_state_indices_tensor[:, 0],
                    self.num_spec + 1,
                )
            else:
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
            if spec_token_indx is None:
                core_attn_out[:, : spec_output.shape[1]].copy_(spec_output)
            else:
                core_attn_out[0].index_copy_(0, spec_token_indx, spec_output[0])

        if q_ns is None:
            maybe_save_kv_layer_to_connector("", [])
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
                output_buffer=core_attn_out if direct_output else None,
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
        if direct_output:
            maybe_save_kv_layer_to_connector("", [])
            return
        if use_spec:
            core_attn_out[0].index_copy_(0, non_spec_token_indx, output[0])
        else:
            core_attn_out[0, : output.shape[1]].copy_(output[0])
        maybe_save_kv_layer_to_connector("", [])
