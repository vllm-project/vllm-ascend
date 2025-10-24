import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.deepseek_mtp import \
    DeepSeekMultiTokenPredictorLayer
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
from vllm.model_executor.models.utils import maybe_prefix


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
