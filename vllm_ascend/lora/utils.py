import vllm

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)


def patch_mcp_apply_non_fully_sharded():
    """Fix the merged column-parallel LoRA apply for non fully-sharded TP.

    Upstream ``_mcp_apply`` unconditionally all-gathers the shrink buffer over
    the LoRA-rank dim. That is only correct for the fully-sharded (S-LoRA) path
    where ``lora_a`` is rank-sharded.
    """
    import vllm.lora.layers.column_parallel_linear as cpl

    if getattr(cpl._mcp_apply, "_ascend_non_fully_sharded_patched", False):
        return

    orig_mcp_apply = cpl._mcp_apply
    orig_all_gather = cpl.tensor_model_parallel_all_gather

    def _patched_mcp_apply(x, bias, layer):
        if getattr(layer.lora_config, "fully_sharded_loras", False):
            return orig_mcp_apply(x, bias, layer)
        
        cpl.tensor_model_parallel_all_gather = lambda buf, *a, **k: buf
        try:
            return orig_mcp_apply(x, bias, layer)
        finally:
            cpl.tensor_model_parallel_all_gather = orig_all_gather

    _patched_mcp_apply._ascend_non_fully_sharded_patched = True
    cpl._mcp_apply = _patched_mcp_apply


def refresh_all_lora_classes():
    # todo: Remove this patch once vLLM #35077 is merged and released.
    patch_mcp_apply_non_fully_sharded()
    ascend_classes = (
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Append the Ascend classes in a deterministic order.
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *vllm.lora.utils._all_lora_classes,
    )
