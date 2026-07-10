import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)
from vllm_ascend.ops.linear import (
    AscendQKVParallelLinear,
)


class AscendQKVParallelLinearWithLoRA(QKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 1


class AscendMergedQKVParallelLinearWithLoRA(MergedQKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


class AscendMergedQKVParallelLinearWithShardedLoRA(MergedQKVParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 3


class AscendQKVParallelLinearWithShardedLoRA(QKVParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendQKVParallelLinear and len(packed_modules_list) == 1


def patch_mcp_apply_non_fully_sharded():
    """Fix the merged column-parallel LoRA apply for non fully-sharded TP.

    Upstream ``_mcp_apply`` unconditionally all-gathers the shrink buffer over
    the LoRA-rank dim. That is only correct for the fully-sharded (S-LoRA) path
    where ``lora_a`` is rank-sharded. In the default (non fully-sharded) path
    ``lora_a`` is replicated, so every rank already holds the full-rank shrink
    result; all-gathering there duplicates the rank dim (e.g. 8 -> 16) and
    mismatches ``lora_b_stacked``. The Ascend ``sgmv_expand`` kernel is not
    tolerant of that mismatch, so the LoRA delta collapses to ~0 and TP>1
    output silently reverts to the base model.

    We wrap ``_mcp_apply`` and neutralize the rank-dim all-gather for the
    non fully-sharded case, leaving the fully-sharded path untouched. Wrapping
    (rather than replacing the body) keeps us robust to upstream changes.
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
    patch_mcp_apply_non_fully_sharded()
    ascend_classes = (
        AscendQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithShardedLoRA,
        AscendQKVParallelLinearWithShardedLoRA,
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Append the Ascend classes in a deterministic order.
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *vllm.lora.utils._all_lora_classes,
    )
