import torch.nn as nn
from transformers import PretrainedConfig

from vllm.model_executor.models.glm4_moe_mtp import (SharedHead, 
                                                     Glm4MoeMultiTokenPredictorLayer)
from vllm.model_executor.models.glm4_moe import Glm4MoeDecoderLayer
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import maybe_prefix



class AscendSharedHead(SharedHead):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head") if config.max_position_embeddings == 131072 else "lm_head",
        )


class AscendGlm4MoeMultiTokenPredictorLayer(Glm4MoeMultiTokenPredictorLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
    ) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = SharedHead(
            config=config, prefix=maybe_prefix(prefix, "shared_head"), quant_config=quant_config
        )
        self.enable_eplb = parallel_config.enable_eplb
        self.mtp_block = Glm4MoeDecoderLayer(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            enable_eplb=self.enable_eplb,
        )