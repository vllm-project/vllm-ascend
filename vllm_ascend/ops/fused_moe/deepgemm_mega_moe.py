# SPDX-License-Identifier: Apache-2.0
"""Optional DeepGEMM-Ascend routed FP8 x FP4 MegaMoE backend.

Shared experts remain in vLLM's shared-expert path. The symmetric allocation
belongs to the cached MoE communicator and is reused sequentially across layers.
"""

from functools import cache
from importlib import import_module
from types import ModuleType

import torch
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.distributed.parallel_state import get_mc2_group

_INSTALL_HINT = (
    "Install DeepGEMM-Ascend revision 3a35326 with the HCCL/CANN compatibility patch "
    "documented under enable_deepgemm_mega_moe."
)


@cache
def load_deepgemm_mega_moe_ops() -> ModuleType:
    """Import and cache the validated Ascend backend in the NPU worker."""
    try:
        ops = import_module("deep_gemm")
    except (ImportError, OSError, RuntimeError, AssertionError) as exc:
        raise RuntimeError(f"Could not import deep_gemm for Ascend MegaMoE. {_INSTALL_HINT}") from exc

    # CUDA DeepGEMM exports the same MegaMoE names. get_npu_arch identifies Ascend.
    required = ("get_npu_arch", "transform_weights_for_mega_moe", "SymmBuffer", "fp8_fp4_mega_moe")
    missing = [name for name in required if not callable(getattr(ops, name, None))]
    if missing:
        source = getattr(ops, "__file__", "deep_gemm")
        raise RuntimeError(
            f"Incompatible deep_gemm module at {source}. Missing required APIs ({', '.join(missing)}). {_INSTALL_HINT}"
        )
    return ops


def prepare_weights(w13, w13_scale, w2, w2_scale):
    ops = load_deepgemm_mega_moe_ops()

    return ops.transform_weights_for_mega_moe(
        (
            w13.detach().contiguous().view(torch.int8),
            w13_scale.detach().contiguous().view(torch.uint8).view(torch.int16),
        ),
        (w2.detach().contiguous().view(torch.int8), w2_scale.detach().contiguous().view(torch.uint8).view(torch.int16)),
    )


class DeepGemmMegaMoE:
    def __init__(self, moe_config, max_tokens):
        if moe_config.activation is not MoEActivation.SILU:
            raise NotImplementedError("DeepGEMM-Ascend MegaMoE requires SwiGLU activation")
        self.ops = load_deepgemm_mega_moe_ops()
        self.config = moe_config
        self.max_tokens = max_tokens
        self.buffer = None

    def apply(self, inputs):
        x = inputs.hidden_states
        layer = inputs.layer
        count, hidden = x.shape
        if count > self.max_tokens:
            raise ValueError(f"DeepGEMM MegaMoE token capacity {self.max_tokens} exceeded by {count}")
        if x.dtype != torch.bfloat16:
            raise ValueError("DeepGEMM MegaMoE requires BF16 hidden states")
        if inputs.routing.apply_router_weight_on_input or inputs.dynamic_eplb:
            raise ValueError("DeepGEMM MegaMoE requires output router weighting and static experts")
        if not hasattr(layer, "deepgemm_l1"):
            raise ValueError("DeepGEMM MegaMoE requires converted original FP4 expert weights")
        if self.buffer is None:
            # L1 layout: [local experts, intermediate / granularity, BLOCK_N, hidden/2].
            intermediate = layer.deepgemm_l1[0].shape[1] * layer.deepgemm_l1[0].shape[2] // 2
            self.buffer = self.ops.SymmBuffer(
                get_mc2_group().device_group,
                self.config.num_experts,
                self.max_tokens,
                self.config.experts_per_token,
                hidden,
                intermediate,
                num_shared_experts=0,
            )
            logger.info(
                "DeepGEMM MegaMoE active: experts=%d, hidden=%d, intermediate=%d, capacity=%d",
                self.config.num_experts,
                hidden,
                intermediate,
                self.max_tokens,
            )
        buffer = self.buffer
        if count:
            quantized, scales = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            buffer.x[:count].copy_(quantized)
            buffer.x_sf[:count].copy_(scales.contiguous().view(torch.uint8).view(torch.int16).reshape(count, -1))
            buffer.topk_idx[:count].copy_(inputs.topk_ids)
            buffer.topk_weights[:count].copy_(inputs.topk_weights)
        output = torch.empty_like(x)
        # Empty ranks must still join dispatch/combine and process remote tokens.
        clamp = getattr(layer, "swiglu_limit", self.config.swiglu_limit)
        self.ops.fp8_fp4_mega_moe(
            output,
            layer.deepgemm_l1,
            layer.deepgemm_l2,
            buffer,
            activation_clamp=clamp if clamp and clamp > 0 else None,
        )
        return output
