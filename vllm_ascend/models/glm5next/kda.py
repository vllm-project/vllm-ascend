# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate."""

import os

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.attention.utils import (
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.distributed.parallel_state import get_otp_group
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter
from vllm_ascend.models.glm5next.config import Glm5NextConfig
from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.models.glm5next.ops.kda import KDA_MAX_RECURRENT_TOKENS, chunk_kda, recurrent_kda
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBackend
from vllm_ascend.utils import get_potential_max_tokens, oproj_tp_enable


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
        # Resolve FLA only when building the model, after device initialization.
        from vllm_ascend.utils import register_ascend_fla_customop

        register_ascend_fla_customop()
        from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Fused o_proj MatMul+ReduceScatter state. The decoder layer arms this
        # via maybe_enable_kda_oproj_mm_rs; unarmed keeps the stock path.
        self._oproj_mm_rs_hcom: str | None = None
        self._oproj_mm_rs_world_size = 0
        self._oproj_mm_rs_min_tokens = 0
        # Fused in_proj AllGather+MatMul state, armed independently of the
        # o_proj fusion via maybe_enable_kda_inproj_ag_mm. Once armed the
        # decoder layer stops issuing sp_all_gather and hands this layer its
        # own token shard instead of the full sequence.
        self._inproj_ag_mm_hcom: str | None = None
        self._inproj_ag_mm_world_size = 0
        self._inproj_ag_mm_min_tokens = 0

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
        return AscendGDNHostMetadataBackend

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._inproj_ag_mm_hcom is not None:
            # Armed: the decoder layer skipped its sp_all_gather, so
            # hidden_states is this rank's token shard and the full token
            # count has to come from positions rather than from dim 0.
            num_tokens = positions.shape[0]
            projected = self._inproj_ag_mm_apply(hidden_states, num_tokens)
        else:
            num_tokens = hidden_states.size(0)
            # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 GEMMs).
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
        return self._finalize_o_proj(core_attn_out)

    def _inproj_ag_mm_can_fuse(
        self,
        shard: torch.Tensor,
        weight: torch.Tensor,
        num_tokens: int,
    ) -> bool:
        """Per-batch gate for the in_proj AllGather+MatMul fusion.

        Mirrors _oproj_mm_rs_can_fuse. num_tokens is the full sequence length
        and is TP-identical, so every rank takes the same branch; the fused
        kernel's fixed cost is what the lower bound guards against.
        """
        if not _is_prefill_only_batch():
            return False
        world_size = self._inproj_ag_mm_world_size
        if shard.ndim != 2 or weight.ndim != 2:
            return False
        if shard.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            return False
        if shard.shape[1] != weight.shape[1]:
            return False
        if not _INPROJ_AG_MM_MIN_K <= shard.shape[1] < _INPROJ_AG_MM_MAX_K:
            return False
        if not (weight.is_contiguous() or weight.t().is_contiguous()):
            return False
        shard_rows = shard.shape[0]
        if shard_rows == 0 or weight.shape[0] == 0 or num_tokens == 0:
            return False
        # The gathered result must cover every real token; otherwise slicing
        # the GEMM output back to num_tokens would silently drop rows.
        if num_tokens > shard_rows * world_size:
            return False
        if num_tokens < self._inproj_ag_mm_min_tokens:
            return False
        return shard_rows * world_size * shard.shape[1] * shard.element_size() < _INPROJ_AG_MM_MAX_COMM_BYTES

    def _inproj_ag_mm_apply(self, shard: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """AllGather this rank's shard into the merged in_proj GEMM.

        The unfused fallback reproduces exactly what the decoder layer used to
        do before this layer was armed: gather, slice to the real token count,
        then run the stock merged projection.
        """
        weight = self.in_proj_qkvbfg_a.weight
        if not self._inproj_ag_mm_can_fuse(shard, weight, num_tokens):
            return self.in_proj_qkvbfg_a(sp_all_gather(shard)[:num_tokens])[0]
        output, _ = torch_npu.npu_all_gather_base_mm(
            shard.contiguous(),
            weight.t(),
            self._inproj_ag_mm_hcom,
            self._inproj_ag_mm_world_size,
            # Nothing in this layer consumes the gathered hidden states, so
            # the kernel is told not to materialise them.
            gather_output=False,
            # comm_mode intentionally not passed, as in _oproj_mm_rs_apply:
            # the accepted values differ per platform and the kernel's own
            # default is the right one.
        )
        # AllGather covers the TP-padded row count; drop the padding rows.
        return output[:num_tokens]

    def _oproj_mm_rs_can_fuse(self, attn_output: torch.Tensor, weight: torch.Tensor) -> bool:
        """Per-batch gate, branching only on TP-consistent metadata.

        Decode, spec-verify and mixed batches stay on the unfused pair: the win
        is a prefill-sized GEMM hiding the collective, and a one-token GEMM
        hides nothing.
        """
        if not _is_prefill_only_batch():
            return False
        if attn_output.ndim != 2 or weight.ndim != 2:
            return False
        if attn_output.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            return False
        if attn_output.shape[1] != weight.shape[1]:
            return False
        if not _OPROJ_MM_RS_MIN_K <= attn_output.shape[1] < _OPROJ_MM_RS_MAX_K:
            return False
        if not (weight.is_contiguous() or weight.t().is_contiguous()):
            return False
        num_tokens = attn_output.shape[0]
        if num_tokens == 0 or weight.shape[0] == 0:
            return False
        # Median us/call on TP4: fused ~= 82 + 0.0603*N, the unfused
        # MatMul + reduce_scatter pair ~= 86 + 0.0863*N, so fusion wins across
        # every prefill chunk actually measured (1420/4352/15232 tokens, -21%
        # to -28%). The floor still dominates far below that: at 4 tokens the
        # unfused pair costs 27.8 us against the fused kernel's ~82 us
        # intercept, and that regime is unmeasured, so keep a lower bound.
        # num_tokens is the full gathered chunk and is TP-identical, so every
        # rank takes the same branch.
        if num_tokens < self._oproj_mm_rs_min_tokens:
            return False
        padded = num_tokens + (-num_tokens) % self._oproj_mm_rs_world_size
        return padded * weight.shape[0] * weight.element_size() < _OPROJ_MM_RS_MAX_COMM_BYTES

    def _oproj_mm_rs_apply(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Return this rank's reduce-scattered o_proj shard.

        Both branches return the same token shard the decoder layer's
        sp_reduce_scatter produced before, zero padding rows included, so the
        caller may skip its own collective unconditionally.
        """
        weight = self.o_proj.weight
        if not self._oproj_mm_rs_can_fuse(attn_output, weight):
            return sp_reduce_scatter(self.o_proj(attn_output)[0])
        world_size = self._oproj_mm_rs_world_size
        # sp_reduce_scatter zero-pads the GEMM result up to the TP multiple.
        # The projection is bias-free, so padding the input rows instead yields
        # the same zero rows out of the fused kernel.
        sp_pad = (-attn_output.shape[0]) % world_size
        if sp_pad:
            attn_output = torch.nn.functional.pad(attn_output, (0, 0, 0, sp_pad))
        output, _ = torch_npu.npu_quant_mm_reduce_scatter(
            attn_output.contiguous(),
            weight.t(),
            self._oproj_mm_rs_hcom,
            world_size,
            reduce_op="sum",
            # comm_mode intentionally not passed: Atlas A3/A2 only accepts
            # "aiv" while Ascend 950DT only accepts "ai_cpu"/"ccu". Omitting it
            # lets the kernel pick the right per-platform default.
        )
        return output

    def _finalize_o_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        if not oproj_tp_enable():
            if self._oproj_mm_rs_hcom is not None:
                return self._oproj_mm_rs_apply(attn_output)
            return self.o_proj(attn_output)[0]
        group = get_otp_group()
        tp_size = group.world_size
        chunk_size = self.o_proj.input_size_per_partition
        num_tokens = attn_output.shape[0]
        capacity = get_potential_max_tokens()
        profile_tokens = 0
        if num_tokens > capacity:
            if get_forward_context().attn_metadata is not None:
                raise ValueError(f"KDA OProj TP token count exceeds decode capacity: {num_tokens} > {capacity}.")
            # Profiling keeps a full dummy batch, but still participates in OTP.
            profile_tokens = num_tokens
            attn_output = attn_output[:0]
            num_tokens = 0
        # DP ranks can replay different buckets; all four HCCL operands must stay fixed.
        if not hasattr(self, "_oproj_send_buf"):
            self._oproj_send_buf = attn_output.new_empty((tp_size, capacity, chunk_size))
            self._oproj_recv_buf = torch.empty_like(self._oproj_send_buf)
            self._oproj_reduce_buf = attn_output.new_empty((tp_size * capacity, self.o_proj.output_size))
            self._oproj_output_buf = attn_output.new_empty((capacity, self.o_proj.output_size))
        self._oproj_send_buf.zero_()
        self._oproj_send_buf[:, :num_tokens].copy_(attn_output.reshape(num_tokens, tp_size, chunk_size).transpose(0, 1))
        dist.all_to_all_single(self._oproj_recv_buf.view(-1), self._oproj_send_buf.view(-1), group=group.device_group)
        bias = None if group.rank_in_group > 0 or self.o_proj.skip_bias_add else self.o_proj.bias
        assert self.o_proj.quant_method is not None
        self._oproj_reduce_buf.copy_(
            self.o_proj.quant_method.apply(
                self.o_proj, self._oproj_recv_buf.view(tp_size * capacity, chunk_size), bias=bias
            )
        )
        dist.reduce_scatter_tensor(self._oproj_output_buf, self._oproj_reduce_buf, group=group.device_group)
        if profile_tokens:
            return self._oproj_output_buf.new_zeros((profile_tokens, self.o_proj.output_size))
        return self._oproj_output_buf[:num_tokens]

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

        # Pure decode writes the whole captured buffer, including padding.
        direct_output = (
            not use_spec and attn_metadata_narrowed.num_prefills == 0 and attn_metadata_narrowed.num_decodes > 0
        )
        if not direct_output:
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


# --- KDA o_proj MatMul+ReduceScatter fusion -------------------------------
# Folds the decoder layer's sp_reduce_scatter into the o_proj GEMM via
# torch_npu.npu_quant_mm_reduce_scatter, so the collective pipelines behind the
# GEMM instead of starting only after it retires. Modelled on
# vllm_ascend/ops/linear_op.py KimiOProjMMReduceScatterOp.
_OPROJ_MM_RS_ENV = "VLLM_ASCEND_GLM_KDA_OPROJ_MM_RS"
_OPROJ_MM_RS_WORLD_SIZES = (2, 4, 8)
_OPROJ_MM_RS_MIN_K = 256
_OPROJ_MM_RS_MAX_K = 65535
_OPROJ_MM_RS_MAX_COMM_BYTES = 16 * 256 * 1024 * 1024
_OPROJ_MM_RS_TRUTHY = ("1", "true", "yes", "on")
_OPROJ_MM_RS_MIN_TOKENS_ENV = "VLLM_ASCEND_GLM_KDA_OPROJ_MM_RS_MIN_TOKENS"
_OPROJ_MM_RS_MIN_TOKENS_DEFAULT = 1024
_oproj_mm_rs_announced = False

# --- KDA in_proj AllGather+MatMul fusion ----------------------------------
# Folds the decoder layer's sp_all_gather into the merged in_proj GEMM via
# torch_npu.npu_all_gather_base_mm, so the collective pipelines behind the
# GEMM instead of fully retiring before it starts. Unlike the o_proj fusion
# this does not reassociate any sum: AllGather moves whole rows along M and
# the K-axis reduction is untouched.
_INPROJ_AG_MM_ENV = "VLLM_ASCEND_GLM_KDA_INPROJ_AG_MM"
_INPROJ_AG_MM_MIN_TOKENS_ENV = "VLLM_ASCEND_GLM_KDA_INPROJ_AG_MM_MIN_TOKENS"
# Measured on Atlas A3, TP4, 34 KDA layers, medians over 544-2720 calls per
# point (prof_p_11061929 unfused vs prof_p_11062136 fused, identical chunk mix):
#
#     chunk    unfused AG+MM        fused      delta/layer
#     15232    454.1 + 2791.7       3071.9       -173.9 us
#      4352    158.4 +  842.4       1024.0        + 23.2 us
#      1420     78.8 +  347.6        583.9        +157.5 us
#
# Least-squares over those three points:
#     fused   = 283.3 + 0.1823 * N us
#     unfused = 124.3 + 0.2047 * N us   =>  break-even at N ~= 7100
#
# The fused kernel's fixed cost (283 us) is 3.5x the o_proj side's (82 us)
# because MC2 splits the GEMM into world_size sub-GEMMs of shard_rows each:
# at chunk 1420 that is four M=355 GEMMs in place of one M=1420 GEMM, well
# below the cube's efficient M. The o_proj fusion does not pay this because
# its GEMM keeps the full M and only the reduce_scatter is chunked. Hence a
# much higher default here than the o_proj side's 1024.
_INPROJ_AG_MM_MIN_TOKENS_DEFAULT = 1024
_INPROJ_AG_MM_WORLD_SIZES = (2, 4, 8)
_INPROJ_AG_MM_MIN_K = 256
_INPROJ_AG_MM_MAX_K = 65535
_INPROJ_AG_MM_MAX_COMM_BYTES = 16 * 256 * 1024 * 1024
_inproj_ag_mm_announced = False

logger = init_logger(__name__)


def glm_kda_oproj_mm_rs_enabled() -> bool:
    """Read the fusion switch; unset keeps the stock GEMM + reduce_scatter."""
    return os.getenv(_OPROJ_MM_RS_ENV, "0").strip().lower() in _OPROJ_MM_RS_TRUTHY


def _is_prefill_only_batch() -> bool:
    """True for a batch that is pure prefill, on every TP rank alike.

    Both comm-matmul fusions share this predicate: their win is a
    prefill-sized GEMM hiding a collective, and a one-token decode GEMM hides
    nothing while still paying the fused kernel's fixed cost.
    """
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, dict):
        metadata = next(
            (m for m in metadata.values() if hasattr(m, "num_prefills") and hasattr(m, "num_decodes")),
            None,
        )
    return not (
        metadata is None
        or getattr(metadata, "num_prefills", 0) == 0
        or getattr(metadata, "num_decodes", 0) != 0
        or getattr(metadata, "num_spec_decodes", 0) != 0
    )


def _min_tokens_from_env(env_name: str, default: int) -> int:
    """Read a token lower bound; 0 disables the threshold entirely.

    Both fusions measured faster at every prefill chunk from 1420 tokens up,
    so the defaults only guard the unmeasured short-prefill regime where the
    fused kernel's fixed cost cannot be amortised. A malformed value falls
    back to the default rather than failing model build.
    """
    raw = os.getenv(env_name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d.", env_name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%d is negative; using %d.", env_name, value, default)
        return default
    return value


def _oproj_mm_rs_min_tokens() -> int:
    return _min_tokens_from_env(_OPROJ_MM_RS_MIN_TOKENS_ENV, _OPROJ_MM_RS_MIN_TOKENS_DEFAULT)


def _inproj_ag_mm_min_tokens() -> int:
    return _min_tokens_from_env(_INPROJ_AG_MM_MIN_TOKENS_ENV, _INPROJ_AG_MM_MIN_TOKENS_DEFAULT)


def _oproj_mm_rs_unsupported_reason(attn: nn.Module) -> str | None:
    """Static reasons this o_proj cannot use npu_quant_mm_reduce_scatter.

    The fused kernel needs a bias-free unquantized BF16 projection on a 2/4/8
    rank TP group with a local K inside its supported range; a nonzero bias is
    explicitly unsupported by the current interface.
    """
    layer = attn.o_proj
    if not callable(getattr(torch_npu, "npu_quant_mm_reduce_scatter", None)):
        return "torch_npu.npu_quant_mm_reduce_scatter is unavailable"
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        return "the fused kernel needs an unquantized o_proj"
    weight = getattr(layer, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
        return "the fused kernel needs BF16 o_proj weights"
    if weight.ndim != 2 or weight.shape[0] == 0:
        return "the fused kernel needs nonempty 2D o_proj weights"
    if layer.bias is not None:
        return "the fused kernel does not support a nonzero bias"
    if not _OPROJ_MM_RS_MIN_K <= weight.shape[1] < _OPROJ_MM_RS_MAX_K:
        return f"local K {weight.shape[1]} is outside [256, 65535)"
    world_size = get_tp_group().world_size
    if world_size not in _OPROJ_MM_RS_WORLD_SIZES:
        return f"TP size {world_size} is not 2, 4 or 8"
    return None


def maybe_enable_kda_oproj_mm_rs(attn: nn.Module) -> bool:
    """Arm the fused o_proj path for one KDA layer.

    Returns True when the caller must stop issuing its own sp_reduce_scatter,
    because _finalize_o_proj now returns the reduce-scattered token shard.
    """
    global _oproj_mm_rs_announced
    if not isinstance(attn, Glm5NextLinearAttention):
        return False
    if not glm_kda_oproj_mm_rs_enabled():
        return False
    reason = _oproj_mm_rs_unsupported_reason(attn)
    if reason is not None:
        if not _oproj_mm_rs_announced:
            _oproj_mm_rs_announced = True
            logger.warning("GLM KDA o_proj MM+ReduceScatter disabled: %s.", reason)
        return False
    tp_group = get_tp_group()
    try:
        backend = tp_group.device_group._get_backend(torch.device("npu"))
        hcom = backend.get_hccl_comm_name(tp_group.rank_in_group)
    except Exception as exc:  # noqa: BLE001 - never fail model build over HCCL
        if not _oproj_mm_rs_announced:
            _oproj_mm_rs_announced = True
            logger.warning("GLM KDA o_proj MM+ReduceScatter disabled: no HCCL handle (%s).", exc)
        return False
    min_tokens = _oproj_mm_rs_min_tokens()
    attn._oproj_mm_rs_hcom = hcom
    attn._oproj_mm_rs_world_size = tp_group.world_size
    attn._oproj_mm_rs_min_tokens = min_tokens
    if not _oproj_mm_rs_announced:
        _oproj_mm_rs_announced = True
        # warning, not info: the deployment's log config drops info from the
        # worker log files, which once made an armed fusion look inactive.
        logger.warning(
            "GLM KDA o_proj MM+ReduceScatter enabled (world_size=%d, min_tokens=%d).",
            tp_group.world_size,
            min_tokens,
        )
    return True


def glm_kda_inproj_ag_mm_enabled() -> bool:
    """Read the in_proj fusion switch; unset keeps sp_all_gather + GEMM."""
    return os.getenv(_INPROJ_AG_MM_ENV, "0").strip().lower() in _OPROJ_MM_RS_TRUTHY


def _inproj_ag_mm_unsupported_reason(attn: nn.Module) -> str | None:
    """Static reasons this in_proj cannot use npu_all_gather_base_mm.

    The fused kernel needs a bias-free unquantized BF16 projection on a 2/4/8
    rank TP group with K inside its supported range; the current interface
    explicitly rejects a nonzero bias.
    """
    layer = attn.in_proj_qkvbfg_a
    if not callable(getattr(torch_npu, "npu_all_gather_base_mm", None)):
        return "torch_npu.npu_all_gather_base_mm is unavailable"
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        return "the fused kernel needs an unquantized in_proj"
    weight = getattr(layer, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
        return "the fused kernel needs BF16 in_proj weights"
    if weight.ndim != 2 or weight.shape[0] == 0:
        return "the fused kernel needs nonempty 2D in_proj weights"
    if layer.bias is not None:
        return "the fused kernel does not support a nonzero bias"
    if not _INPROJ_AG_MM_MIN_K <= weight.shape[1] < _INPROJ_AG_MM_MAX_K:
        return f"K {weight.shape[1]} is outside [256, 65535)"
    world_size = get_tp_group().world_size
    if world_size not in _INPROJ_AG_MM_WORLD_SIZES:
        return f"TP size {world_size} is not 2, 4 or 8"
    return None


def maybe_enable_kda_inproj_ag_mm(attn: nn.Module) -> bool:
    """Arm the fused in_proj path for one KDA layer.

    Returns True when the caller must stop issuing its own sp_all_gather and
    pass this layer the SP token shard, because forward now gathers itself.
    """
    global _inproj_ag_mm_announced
    if not isinstance(attn, Glm5NextLinearAttention):
        return False
    if not glm_kda_inproj_ag_mm_enabled():
        return False
    reason = _inproj_ag_mm_unsupported_reason(attn)
    if reason is not None:
        if not _inproj_ag_mm_announced:
            _inproj_ag_mm_announced = True
            logger.warning("GLM KDA in_proj AllGather+MatMul disabled: %s.", reason)
        return False
    tp_group = get_tp_group()
    try:
        backend = tp_group.device_group._get_backend(torch.device("npu"))
        hcom = backend.get_hccl_comm_name(tp_group.rank_in_group)
    except Exception as exc:  # noqa: BLE001 - never fail model build over HCCL
        if not _inproj_ag_mm_announced:
            _inproj_ag_mm_announced = True
            logger.warning("GLM KDA in_proj AllGather+MatMul disabled: no HCCL handle (%s).", exc)
        return False
    min_tokens = _inproj_ag_mm_min_tokens()
    attn._inproj_ag_mm_hcom = hcom
    attn._inproj_ag_mm_world_size = tp_group.world_size
    attn._inproj_ag_mm_min_tokens = min_tokens
    if not _inproj_ag_mm_announced:
        _inproj_ag_mm_announced = True
        logger.warning(
            "GLM KDA in_proj AllGather+MatMul enabled (world_size=%d, min_tokens=%d).",
            tp_group.world_size,
            min_tokens,
        )
    return True
