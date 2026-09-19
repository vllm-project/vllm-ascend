# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
    SharedHead,
)
from vllm.model_executor.models.utils import maybe_prefix

from .model import Dots3NoteDecoderLayer


class Dots3NoteMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(config=config, prefix=prefix, quant_config=vllm_config.quant_config)
        self.mtp_block = Dots3NoteDecoderLayer(vllm_config=vllm_config, config=config, prefix=prefix)


class Dots3NoteMultiTokenPredictor(DeepSeekMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = nn.ModuleDict(
            {
                str(idx): Dots3NoteMultiTokenPredictorLayer(vllm_config, f"{prefix}.layers.{idx}")
                for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "embed_tokens")
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)


@support_torch_compile
class Dots3NoteMTP(DeepSeekMTP):
    has_own_embed_tokens = True
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Dots3NoteMultiTokenPredictor(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.set_moe_parameters()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def mtp_weights():
            for name, weight in weights:
                if name.startswith("model.mtp.embed_tokens."):
                    name = name.replace(
                        "model.mtp.embed_tokens.",
                        f"model.layers.{self.config.num_hidden_layers}.embed_tokens.",
                        1,
                    )
                yield name, weight

        return super().load_weights(mtp_weights())
