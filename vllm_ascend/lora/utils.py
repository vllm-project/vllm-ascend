from vllm.lora import utils as lora_utils

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)


def refresh_all_lora_classes():
    ascend_classes = (
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Prepend the Ascend classes in a deterministic order. PunicaWrapperNPU
    # may be constructed more than once in a process, so remove old entries
    # first to keep the registry idempotent.
    upstream_classes = tuple(
        lora_class for lora_class in lora_utils._all_lora_classes if lora_class not in ascend_classes
    )
    lora_utils._all_lora_classes = (
        *ascend_classes,
        *upstream_classes,
    )
