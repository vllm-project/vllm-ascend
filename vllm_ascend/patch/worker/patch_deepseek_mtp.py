from typing import Optional

import torch
import torch.nn as nn
import vllm
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP, DeepSeekMultiTokenPredictorLayer)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
from vllm.model_executor.models.utils import maybe_prefix


def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    inputs_embeds: Optional[torch.Tensor] = None,
    spec_step_index: int = 0,
) -> torch.Tensor:
    assert inputs_embeds is not None
    # masking inputs at position 0, as not needed by MTP
    # Patch this for aclgraph support, as the original operation introduced d2h sync,
    # which breaks aclgraph
    inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    inputs_embeds = self.enorm(inputs_embeds)
    previous_hidden_states = self.hnorm(previous_hidden_states)

    hidden_states = self.eh_proj(
        torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

    hidden_states, residual = self.mtp_block(positions=positions,
                                             hidden_states=hidden_states,
                                             residual=None)
    hidden_states = residual + hidden_states
    return hidden_states


# Patch this only for aclgraph support, as this is not support in vLLM 0.11.0
@support_torch_compile
class AscendDeepSeekMTP(DeepSeekMTP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)


class SharedHead(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


def predictor_init(self, vllm_config: VllmConfig, prefix: str) -> None:
    nn.Module.__init__(self)
    config = vllm_config.model_config.hf_config
    quant_config = vllm_config.quant_config

    self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.eh_proj = nn.Linear(config.hidden_size * 2,
                             config.hidden_size,
                             bias=False)

    # We don't need topk_indices_buffer in Ascend
    topk_indices_buffer = None
    self.shared_head = SharedHead(config=config,
                                  prefix=prefix,
                                  quant_config=quant_config)
    self.mtp_block = DeepseekV2DecoderLayer(vllm_config, prefix,
                                            topk_indices_buffer)


DeepSeekMultiTokenPredictorLayer.__init__ = predictor_init
vllm.model_executor.models.deepseek_mtp.DeepSeekMultiTokenPredictorLayer.forward = forward
