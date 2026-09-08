# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MTP draft model for Ascend."""

import copy

from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MTP as UpstreamKimiK3MTP,
)
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MultiTokenPredictor as UpstreamKimiK3MultiTokenPredictor,
)
from vllm.models.kimi_k3.amd.mtp import (
    KimiK3MultiTokenPredictorLayer as UpstreamKimiK3MultiTokenPredictorLayer,
)
from vllm.models.kimi_k3.amd.mtp import SharedHead

from vllm_ascend.models.kimi_k3 import AscendKimiDecoderLayer


class AscendKimiK3MultiTokenPredictorLayer(
    UpstreamKimiK3MultiTokenPredictorLayer,
):
    def __init__(self, config, vllm_config: VllmConfig, prefix: str) -> None:
        # The upstream constructor hard-codes the AMD decoder layer.  Build the
        # same container with the Ascend decoder and inherit its forward path.
        nn.Module.__init__(self)
        self.config = config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
        )
        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            quant_config=vllm_config.quant_config,
        )
        block_config = copy.copy(config)
        block_config.attn_res_block_size = None
        self.mtp_block = AscendKimiDecoderLayer(
            block_config,
            vllm_config,
            prefix=prefix,
        )


class AscendKimiK3MultiTokenPredictor(UpstreamKimiK3MultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # The upstream constructor hard-codes its predictor-layer class.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): AscendKimiK3MultiTokenPredictorLayer(
                    config,
                    vllm_config,
                    f"{prefix}.layers.{idx}",
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)


class AscendKimiK3MTP(UpstreamKimiK3MTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config
        self.model = AscendKimiK3MultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
