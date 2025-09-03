from typing import Optional

from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import VocabParallelEmbeddingWithLoRA

from vllm_ascend.ops.vocab_parallel_embedding import \
    AscendVocabParallelEmbedding


class AscendVocabParallelEmbeddingWithLoRA(VocabParallelEmbeddingWithLoRA):

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is AscendVocabParallelEmbedding
