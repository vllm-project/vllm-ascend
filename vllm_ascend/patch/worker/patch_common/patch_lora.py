import vllm
from vllm.lora.utils import _all_lora_classes

from vllm_ascend.patch.worker.patch_common.patch_lora_embedding import \
    AscendVocabParallelEmbeddingWithLoRA
from vllm_ascend.patch.worker.patch_common.patch_lora_linear import (
    AscendColumnParallelLinearWithLoRA,
    AscendMergedColumnParallelLinearWithLoRA, AscendRowParallelLinearWithLoRA)

_all_lora_classes.add(AscendRowParallelLinearWithLoRA)
_all_lora_classes.add(AscendColumnParallelLinearWithLoRA)
_all_lora_classes.add(AscendMergedColumnParallelLinearWithLoRA)
_all_lora_classes.add(AscendVocabParallelEmbeddingWithLoRA)

vllm.lora.utils._all_lora_classes = _all_lora_classes
