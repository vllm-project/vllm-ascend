# SPDX-License-Identifier: Apache-2.0
"""Optional DeepGEMM-Ascend routed FP8 x FP4 MegaMoE backend.

Shared experts remain in vLLM's shared-expert path. The symmetric allocation
belongs to the cached MoE communicator and is reused sequentially across layers.
"""

import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend.distributed.parallel_state import get_mc2_group


def prepare_weights(w13, w13_scale, w2, w2_scale):
    # Optional native extension is loaded only in the worker, after device setup.
    import deep_gemm

    return deep_gemm.transform_weights_for_mega_moe(
        (
            w13.detach().contiguous().view(torch.int8),
            w13_scale.detach().contiguous().view(torch.uint8).view(torch.int16),
        ),
        (w2.detach().contiguous().view(torch.int8), w2_scale.detach().contiguous().view(torch.uint8).view(torch.int16)),
    )


class DeepGemmMegaMoE:
    def __init__(self, moe_config, max_tokens):
        import deep_gemm

        required = ("fp8_fp4_mega_moe", "transform_weights_for_mega_moe", "SymmBuffer")
        if any(not callable(getattr(deep_gemm, name, None)) for name in required):
            raise RuntimeError("DeepGEMM-Ascend must expose the native MegaMoE API")
        self.ops = deep_gemm
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
