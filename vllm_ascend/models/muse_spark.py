# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark Open on Ascend: correctness-first bring-up over the upstream ROCm stack.

vLLM owns the Muse Spark configuration, weight mapping and decoder structure
(``vllm.models.muse_spark``). This module reuses the ROCm text stack, which is
the smaller of the two vendor stacks, and replaces each piece that only runs
on CUDA or ROCm with its PyTorch reference:

* RMSNorm: the upstream module calls a CUDA-tuned Triton kernel for every
  contiguous input, so every Muse Spark norm is switched to ``forward_native``.
* QK-norm and RoPE: upstream rotates through ``RotaryEmbedding.forward_cuda``
  and can insert K/V from a ROCm kernel. Here QK-norm and RoPE run in fp32
  through ``forward_native`` and round once, as the upstream kernel does, and
  the Ascend attention backend writes the KV cache.
* Attention output gate: the weightless RMSNorm with a sigmoid gate runs
  through the Ascend ``rms_norm_gated`` kernel, as in GLM-5.3-Flash KDA.
* MoE: Muse Spark normalizes every expert's output before the weighted top-k
  sum. The Ascend MoE combine has no hook for that norm, so the routed experts
  run as a dense PyTorch reference. It is meant for reduced bring-up
  configurations, not for serving the full checkpoint.

Quantized checkpoints, expert parallelism, sequence-parallel MoE and the
DSpark draft are not supported yet.
"""

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.models.muse_spark.amd.model import (
    MuseSpark1xAttention,
    MuseSpark1xDecoderLayer,
    MuseSpark1xForCausalLM,
    MuseSpark1xMoE,
    MuseSparkTorchLinearMethod,
)
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.models.muse_spark.common.moe import MuseSpark1xRouter
from vllm.models.muse_spark.common.omni import (
    OMNI_WEIGHTS_MAPPER,
    MuseSpark1xOmniMixin,
    register_omni_processor,
)

from vllm_ascend.ops.triton.kda.kda import rms_norm_gated

ROUTER_DTYPE = torch.float32
ROUTER_SCORE_NORMALIZATION = "sigmoid"
ROUTER_SCORE_DIVISION_EPS = 1e-15
ATTENTION_OUTPUT_GATE_ACTIVATION = "sigmoid"


class _NativeRMSNormMixin:
    """Always take the PyTorch reference path of a Muse Spark RMSNorm."""

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual, out_dtype)  # type: ignore[attr-defined]

    def supports_fused_kernel(self, x: torch.Tensor) -> bool:
        # Also read by the fused residual boundaries, which then fall back too.
        return False


class AscendMuseSpark1xRMSNorm(_NativeRMSNormMixin, MuseSpark1xRMSNorm):
    pass


class AscendMuseSpark1xRMSNormWithInputScale(_NativeRMSNormMixin, MuseSpark1xRMSNormWithInputScale):
    pass


class AscendMuseSpark1xAttention(MuseSpark1xAttention):
    """Muse Spark attention with PyTorch QK-norm/RoPE and the Ascend gated norm."""

    def _qk_norm_rope_kv_insert(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.use_o_gate:
            q, k, v, og = qkv.split([self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            og = None

        # Like the upstream kernel, norm and rotate in fp32 and round once at
        # the end; rounding the norm output before RoPE would round twice.
        if self.qk_norm is not None:
            q = self.qk_norm(q.reshape(-1, self.head_dim), out_dtype=torch.float32).reshape(-1, self.q_size)
            k = self.qk_norm(k.reshape(-1, self.head_dim), out_dtype=torch.float32).reshape(-1, self.kv_size)

        if self.use_attn_temperature_tuning:
            q = q * self._get_attn_temperature_scale(positions)

        if self.rotary_emb is not None:
            # fp32 inputs also keep the shared fp32 cos/sin cache in fp32.
            q, k = self.rotary_emb.forward_native(positions, q.float(), k.float())
        return q.to(v.dtype), k.to(v.dtype), v, og

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.fuse_o_norm_gate:
            # The unfused output norm and gate upstream are plain PyTorch.
            return super().forward(positions, hidden_states)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, og = self._qk_norm_rope_kv_insert(qkv, positions)
        assert og is not None
        attn_output = self.attn(q, k, v)

        # Same contract as upstream's FLA call: weightless RMSNorm over each
        # head, then multiplied by sigmoid(og).
        attn_output = rms_norm_gated(
            attn_output.reshape(-1, self.num_heads, self.head_dim),
            og.reshape(-1, self.num_heads, self.head_dim).contiguous(),
            None,
            None,
            ATTENTION_OUTPUT_GATE_ACTIVATION,
            eps=self.rms_norm_eps,
        )
        output, _ = self.o_proj(attn_output.reshape(-1, self.num_heads * self.head_dim))
        return output


# Upstream builds these modules inside its own constructors; the Ascend model
# switches each instance to the Ascend subclass once construction finishes.
# The subclasses add no state, so the switch keeps every parameter and buffer.
ASCEND_MODULE_CLASSES: dict[type[nn.Module], type[nn.Module]] = {
    MuseSpark1xRMSNorm: AscendMuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale: AscendMuseSpark1xRMSNormWithInputScale,
    MuseSpark1xAttention: AscendMuseSpark1xAttention,
}


def adopt_ascend_modules(model: nn.Module) -> None:
    """Move every upstream Muse Spark norm and attention onto its Ascend path."""
    for module in model.modules():
        ascend_cls = ASCEND_MODULE_CLASSES.get(type(module))
        if ascend_cls is None:
            continue
        module.__class__ = ascend_cls
        if isinstance(module, AscendMuseSpark1xAttention):
            # The Ascend attention backend owns the KV cache write.
            module.fused_qk_norm_rope_kv_insert = False
            module.attn.kv_cache_update_bypass = False


class MuseSparkReferenceExperts(nn.Module):
    """Routed experts as a dense PyTorch reference with the per-expert norm.

    Parameter names and shapes follow the consolidated checkpoint tensors,
    ``w13_weight [E, 2I, H]`` (gate rows, then up rows) and
    ``w2_weight [E, H, I]``, so the default weight loader copies them as is.
    Every token runs through every expert, so compute grows with the expert
    count; the shapes stay static and no device-to-host sync is needed.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w13_weight = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, dtype=params_dtype),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        post_expert_norm: MuseSpark1xRMSNorm,
    ) -> torch.Tensor:
        # hidden_states [T, D], topk_weights/topk_ids [T, K]; D is the expert width.
        gate_up = torch.einsum("td,eod->teo", hidden_states, self.w13_weight)  # [T, E, 2I]
        gate, up = gate_up.chunk(2, dim=-1)
        activated = (F.silu(gate.float()) * up.float()).to(hidden_states.dtype)  # [T, E, I]
        expert_rows = torch.einsum("tei,edi->ted", activated, self.w2_weight)  # [T, E, D]

        route_index = topk_ids.long().unsqueeze(-1).expand(-1, -1, self.hidden_size)
        route_rows = expert_rows.gather(1, route_index)  # [T, K, D]
        # Normalize each route's output before the weighted sum.
        normalized = post_expert_norm(route_rows, out_dtype=torch.float32)
        combined = (normalized * topk_weights.float().unsqueeze(-1)).sum(dim=1)  # [T, D]
        return combined.to(hidden_states.dtype)


class AscendMuseSpark1xMoE(MuseSpark1xMoE):
    """Muse Spark latent MoE whose routed experts run as a PyTorch reference.

    The upstream constructor is not called: it builds ``FusedMoEFactory`` with
    a CUDA/ROCm runner. Router, pooler, unpooler and the two expert norms are
    built here with the same names, so checkpoints map onto the same keys.
    """

    def __init__(self, vllm_config: VllmConfig, layer_idx: int, prefix: str = "") -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_text_config
        self.ep_size = get_ep_group().device_group.size()
        if self.ep_size > 1:
            raise NotImplementedError("Muse Spark on Ascend does not support expert parallelism yet.")
        self.is_sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe
        if self.is_sequence_parallel:
            raise NotImplementedError("Muse Spark on Ascend does not support sequence-parallel MoE yet.")

        self.layer_idx = layer_idx
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.n_routed_experts = config.num_local_experts
        self.n_logical_experts = config.num_local_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts
        self.n_shared_experts = config.num_shared_experts

        self.router_dtype = ROUTER_DTYPE
        self.router = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            out_dtype=self.router_dtype,
            params_dtype=self.router_dtype,
            prefix=f"{prefix}.router",
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
        )
        self.route_scale = config.routed_scaling_factor
        self.router_output_multiplier = config.output_multiplier
        self._router_multiplier_folded = False
        self.router.weight.weight_loader = self._load_router_weight
        self.moe_router = MuseSpark1xRouter(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_local_experts,
            expert_bias=self.expert_bias,
            route_scale=self.route_scale,
            router_output_multiplier=self.router_output_multiplier,
            router_score_normalization=ROUTER_SCORE_NORMALIZATION,
            router_score_division_eps=ROUTER_SCORE_DIVISION_EPS,
        )

        moe_hidden_size = config.moe_hidden_size
        self.pooler = ReplicatedLinear(
            config.hidden_size,
            moe_hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.pooler",
        )
        self.pooler.quant_method = MuseSparkTorchLinearMethod()
        self.unpooler = ReplicatedLinear(
            moe_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.unpooler",
        )
        self.unpooler.quant_method = MuseSparkTorchLinearMethod()
        self.pre_expert_norm = AscendMuseSpark1xRMSNorm(
            hidden_size=moe_hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            is_post_norm=False,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
            output_norm_gain_center_type="zero",
        )
        self.post_expert_norm = AscendMuseSpark1xRMSNormWithInputScale(
            hidden_size=moe_hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=None,
            output_norm_gain_center_type="zero",
        )

        self.intermediate_size = config.moe_intermediate_size
        self.experts = MuseSparkReferenceExperts(
            num_experts=config.num_local_experts,
            hidden_size=moe_hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=vllm_config.model_config.dtype,
        )

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.pooler(hidden_states)
        return self.pre_expert_norm(pooled)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self._router_logits(hidden_states)  # [T, E] fp32
        topk_weights, topk_ids = self.moe_router._compute_routing_native(router_logits, indices_type=None)
        pooled = self._pool(hidden_states)  # [T, moe_hidden]
        experts_out = self.experts(pooled, topk_weights, topk_ids, self.post_expert_norm)
        output, _ = self.unpooler(experts_out)
        return output


class AscendMuseSpark1xDecoderLayer(MuseSpark1xDecoderLayer):
    moe_cls = AscendMuseSpark1xMoE


class AscendMuseSpark1xForCausalLM(MuseSpark1xForCausalLM):
    decoder_layer_cls = AscendMuseSpark1xDecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[MuseSpark1xDecoderLayer] | None = None,
    ) -> None:
        if vllm_config.quant_config is not None:
            raise NotImplementedError(
                "Muse Spark on Ascend supports unquantized (BF16) checkpoints only; "
                f"got quantization {vllm_config.quant_config.get_name()!r}."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        adopt_ascend_modules(self)


@register_omni_processor
class AscendMuseSparkForConditionalGeneration(MuseSpark1xOmniMixin, AscendMuseSpark1xForCausalLM, SupportsMultiModal):
    """Muse Spark Omni: the upstream vision and audio towers over the Ascend text stack."""

    hf_to_vllm_mapper = MuseSpark1xForCausalLM.hf_to_vllm_mapper | OMNI_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # The towers are built after the text stack; cover their norms too.
        adopt_ascend_modules(self)


__all__ = [
    "AscendMuseSpark1xForCausalLM",
    "AscendMuseSparkForConditionalGeneration",
]
