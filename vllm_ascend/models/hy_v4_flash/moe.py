# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.models.hy_v4_flash.moe import HYV4FlashMoE

from .attention import get_flash_attention_metadata

ROUTER_TILE_ROWS = 64


class FlashGateLinear(GateLinear):
    def forward(self, x: torch.Tensor):
        batch = get_flash_attention_metadata()
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
        parallel_config = get_current_vllm_config().parallel_config
        super().__init__(
            config,
            quant_config,
            prefix,
            is_sequence_parallel=parallel_config.use_sequence_parallel_moe,
        )
