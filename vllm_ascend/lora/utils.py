from typing import Optional

import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              RowParallelLinearWithLoRA,
                              VocabParallelEmbeddingWithLoRA)

from vllm_ascend.ops.linear import (AscendColumnParallelLinear,
                                    AscendMergedColumnParallelLinear,
                                    AscendRowParallelLinear)
from vllm_ascend.ops.vocab_parallel_embedding import \
    AscendVocabParallelEmbedding


class AscendColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is AscendColumnParallelLinear


class AscendMergedColumnParallelLinearWithLoRA(
        MergedColumnParallelLinearWithLoRA):

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is AscendMergedColumnParallelLinear


class AscendRowParallelLinearWithLoRA(RowParallelLinearWithLoRA):

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is AscendRowParallelLinear


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


def refresh_all_lora_classes():
    vllm.lora.utils._all_lora_classes.add(AscendColumnParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(
        AscendMergedColumnParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendRowParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendVocabParallelEmbeddingWithLoRA)
