# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""V4.1-selected MegaMoE execution, using upstream factory/loader hooks.

The factory owns configuration and checkpoint names; the dedicated runner owns
local routing and the entire fused EP operation. No generic MoE communication
object or W4A8 execution branch participates in this path.
"""

import math
import weakref

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.layer import MoERunner

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.fused_moe.deepgemm_v41 import DeepGemmV41Experts
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.quant_type import QuantType


class DeepGemmV41WeightMethod(AscendFusedMoEMethod):
    """Reuse checkpoint allocation/loading, with backend-owned post-load packing."""

    def process_weights_after_loading(self, layer):
        if getattr(layer, "mega_experts", None) is None:
            layer.mega_experts = DeepGemmV41Experts(layer)

    def apply(self, *args, **kwargs):
        raise RuntimeError("V4.1 MegaMoE must execute through DeepGemmV41MoE")


class DeepGemmV41RoutedExperts(AscendRoutedExperts):
    """Keep existing parameter names, EP shard loaders and loader traversal."""

    def _get_quant_method(self, prefix, quant_config, moe_config):
        method = super()._get_quant_method(prefix, quant_config, moe_config)
        if not isinstance(method, AscendFusedMoEMethod) or method.quant_method.quant_type != QuantType.W4A8MXFP:
            raise ValueError("V4.1 DeepGEMM requires ModelSlim W4A8 MXFP4 routed weights")
        return DeepGemmV41WeightMethod(method.quant_method, moe_config, tid2eid=self.tid2eid)


class DeepGemmV41MoE(MoERunner):
    """Complete local-token MegaMoE module behind vLLM's graph op boundary.

    Inherit only the upstream registration/checkpoint interface. Forward deliberately
    skips generic prepare, dispatch, shared overlap, combine and TP reductions.
    """

    def __init__(
        self,
        layer_name,
        moe_config,
        router,
        routed_experts,
        enable_dbo=False,
        gate=None,
        shared_experts=None,
        shared_expert_gate=None,
        routed_input_transform=None,
        routed_output_transform=None,
        routed_scaling_factor=1.0,
    ):
        if enable_dbo or any(
            value is not None for value in (shared_expert_gate, routed_input_transform, routed_output_transform)
        ):
            raise ValueError("V4.1 MegaMoE requires serialized calls and identical shared/routed inputs")
        if not isinstance(routed_experts.quant_method, DeepGemmV41WeightMethod):
            raise ValueError("V4.1 MegaMoE requires its dedicated weight lifecycle")
        # The model retains registered shared parameters under checkpoint names.
        # Weak references avoid duplicate owners; no separate shared forward runs.
        if shared_experts is not None:
            routed_experts._deepgemm_v41_shared_experts = weakref.ref(shared_experts)
            shared_experts.gate_up_proj._deepgemm_v41_shared_linear = True
            shared_experts.down_proj._deepgemm_v41_shared_linear = True
        routed_experts._deepgemm_v41_routed_scaling_factor = routed_scaling_factor
        super().__init__(layer_name, moe_config, router, routed_experts)

    @property
    def is_internal_router(self):
        # The V4.1 model computes FP32 router logits with its existing gate.
        return False

    def forward(self, hidden_states, router_logits, input_ids=None, shared_experts_input=None):
        if shared_experts_input is not None:
            raise ValueError("MegaMoE shared experts consume the same local hidden states")
        return self._forward_entry(hidden_states, router_logits, None, input_ids, self._encode_layer_name(), 0)

    def _forward_impl(self, hidden_states, router_logits, shared_experts_input, input_ids=None):
        experts = getattr(self.routed_experts, "mega_experts", None)
        if experts is None:
            raise RuntimeError("V4.1 MegaMoE weights must be processed before forward")
        weights, indices = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=torch.int32,
            input_ids=input_ids,
        )
        return experts.forward(self.routed_experts, hidden_states, weights, indices)


def create_deepgemm_v41_moe(**kwargs):
    """Select once in V4.1, preserving the factory's public extension points."""
    config = get_current_vllm_config()
    parallel = config.parallel_config
    ascend = get_ascend_config()
    eplb = getattr(ascend, "eplb_config", None)
    if config.model_config.hf_text_config.model_type != "deepseek_v41":
        raise ValueError("deepgemm_mega_moe supports DeepSeek V4.1 only")
    if not get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE_MXFP):
        raise ValueError("V4.1 DeepGEMM requires Ascend 950")
    if get_tensor_model_parallel_world_size() != 1 or get_ep_group().world_size != 8:
        raise ValueError("V4.1 DeepGEMM requires TP1/EP8")
    if (
        ascend.enable_fused_mc2
        or getattr(ascend, "mix_placement", False)
        or parallel.enable_eplb
        or getattr(eplb, "dynamic_eplb", False)
        or getattr(eplb, "expert_map_path", None) is not None
        or kwargs.get("num_redundant_experts", 0)
        or getattr(parallel, "expert_placement_strategy", "linear") != "linear"
        or getattr(parallel, "enable_dbo", False)
        or config.lora_config is not None
    ):
        raise ValueError("V4.1 DeepGEMM requires linear EP without MC2, mixed placement, EPLB, DBO or LoRA")
    factor = kwargs.get("routed_scaling_factor", 1.0)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("V4.1 MegaMoE requires a positive finite routed scaling factor")
    router = AscendFusedTopKRouter(
        top_k=kwargs["top_k"],
        global_num_experts=kwargs["num_experts"],
        renormalize=kwargs["renormalize"],
        scoring_func=kwargs["scoring_func"],
        routed_scaling_factor=1.0,
        e_score_correction_bias=kwargs.get("e_score_correction_bias"),
        tid2eid=kwargs.get("hash_indices_table"),
        bias_vl=kwargs.get("bias_vl"),
        image_sentinel_lo=kwargs["image_sentinel_lo"],
        local_input_ids=True,
    )
    return FusedMoEFactory(
        **kwargs,
        router=router,
        apply_routed_scale_to_output=True,
        runner_cls=DeepGemmV41MoE,
        routed_experts_cls=DeepGemmV41RoutedExperts,
    )
