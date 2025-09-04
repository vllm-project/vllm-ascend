from typing import Optional

import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import VocabParallelEmbeddingWithLoRA
from vllm.lora.utils import _all_lora_classes

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


# Patch for lora register_model issue after overriding VocabParallelEmbedding class (#2515)
_all_lora_classes.add(AscendVocabParallelEmbeddingWithLoRA)
vllm.lora.utils._all_lora_classes = _all_lora_classes
