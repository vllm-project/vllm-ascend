# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.hy_v4_flash.moe import HYV4FlashMoE

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

from .attention import get_flash_attention_metadata
from .pd_role import uses_prefill_ops

ROUTER_TILE_ROWS = 64


class _RouterWeightPreparation:
    def process_weights_after_loading(self, layer):
        weight_fp32 = layer.weight.detach()
        weight_bf16 = weight_fp32.to(torch.bfloat16).contiguous()
        # Keep the checkpoint's precision when it contains genuine FP32 weights.
        if torch.equal(weight_bf16.float(), weight_fp32):
            layer.prefill_router_weight = weight_bf16
        else:
            layer.prefill_router_weight = None
        super().process_weights_after_loading(layer)


class _FlashRouterLinearMethod(_RouterWeightPreparation, UnquantizedLinearMethod):
    pass


class _AscendFlashRouterLinearMethod(_RouterWeightPreparation, AscendUnquantizedLinearMethod):
    pass


class FlashGateLinear(GateLinear):
    def enable_prefill_router(self):
        method = type(self.quant_method)
        if method is UnquantizedLinearMethod:
            self.quant_method = _FlashRouterLinearMethod()
        elif method is AscendUnquantizedLinearMethod:
            self.quant_method = _AscendFlashRouterLinearMethod()
        else:
            return
        self.register_buffer("prefill_router_weight", None, persistent=False)

    def forward(self, x: torch.Tensor):
        batch = get_flash_attention_metadata()
        weight = getattr(self, "prefill_router_weight", None)
        if (
            weight is not None
            and batch is not None
            and batch.prefill_reqs
            and all(batch.prefill_reqs)
            and x.ndim == 2
            and x.shape[0] >= 128
            and x.dtype == torch.bfloat16
            and x.device.type == "npu"
            and weight.device == x.device
            and self.bias is None
            and self.out_dtype == torch.float32
        ):
            return torch.mm(x, weight.t(), out_dtype=torch.float32), None
        if batch is None or len(batch.precise_decode_reqs) < 2 or not all(batch.precise_decode_reqs):
            return super().forward(x)

        # Keep the FP32 router GEMM shape fixed across ordinary decode and
        # speculative verification. Strip padded logits before expert routing.
        logits = []
        for tile in x.split(ROUTER_TILE_ROWS, dim=0):
            rows = tile.shape[0]
            padded = torch.nn.functional.pad(tile, (0, 0, 0, ROUTER_TILE_ROWS - rows))
            scores, bias = super().forward(padded)
            logits.append(scores[:rows])
        return (logits[0] if len(logits) == 1 else torch.cat(logits, dim=0)), bias


class AscendFlashMoE(HYV4FlashMoE):
    gate_cls = FlashGateLinear

    def __init__(self, config, quant_config, prefix):
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        super().__init__(
            config,
            quant_config,
            prefix,
            is_sequence_parallel=parallel_config.use_sequence_parallel_moe,
        )
        if (
            uses_prefill_ops(vllm_config)
            and vllm_config.model_config.enforce_eager
            and is_950()
            and not envs.VLLM_BATCH_INVARIANT
        ):
            self.gate.enable_prefill_router()
