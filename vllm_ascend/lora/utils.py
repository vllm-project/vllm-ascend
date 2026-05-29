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

from vllm_ascend.ops.linear import (
    AscendQKVParallelLinear,
)

from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE


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

class AscendFusedMoEWithLoRA(FusedMoEWithLoRA):
    """Ascend-specific MoE LoRA that uses split MLP execution path.

    Unlike the GPU implementation which uses TritonExperts decorators to inject
    LoRA into the modular kernel pipeline, this Ascend implementation splits the
    MLP execution into discrete steps (w1 GEMM -> w13 LoRA -> activation ->
    w2 LoRA -> w2 GEMM) and inserts LoRA computation at the correct positions.

    This approach is mathematically correct because LoRA modifications are
    applied before the nonlinear activation function (for w13) and before the
    w2 GEMM (for w2), preserving the proper computation order:
        output = W2 @ act((W1 + A1@B1) @ x) + A2@B2 @ act(...)
    """
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return isinstance(source_layer, AscendFusedMoE) and len(packed_modules_list) == 2

    def __init__(self, base_layer: AscendFusedMoE) -> None:
        super().__init__(base_layer)

    def _inject_lora_into_fused_moe(self):
        self.base_layer.ensure_moe_quant_config_init()
        self.base_layer.quant_method._lora_enabled = True
        self.base_layer.quant_method._lora_wrapper = self

    def set_lora(self, index, lora_a, lora_b, embeddings_tensor=None, bias=None):
        if isinstance(lora_a, list) and len(lora_a) > 3:
            num_groups = len(lora_a) // 3

            w1_tensors = []
            w2_tensors = []
            w3_tensors = []

            for i in range(num_groups):
                w1_tensors.append(lora_a[i * 3 + 0])
                w2_tensors.append(lora_a[i * 3 + 1])
                w3_tensors.append(lora_a[i * 3 + 2])

            import torch
            w1_lora_a = torch.stack(w1_tensors)
            w2_lora_a = torch.stack(w2_tensors)
            w3_lora_a = torch.stack(w3_tensors)

            w1_tensors_b = []
            w2_tensors_b = []
            w3_tensors_b = []

            for i in range(num_groups):
                w1_tensors_b.append(lora_b[i * 3 + 0])
                w2_tensors_b.append(lora_b[i * 3 + 1])
                w3_tensors_b.append(lora_b[i * 3 + 2])

            w1_lora_b = torch.stack(w1_tensors_b)
            w2_lora_b = torch.stack(w2_tensors_b)
            w3_lora_b = torch.stack(w3_tensors_b)

            super().set_lora(index,
                            [w1_lora_a, w2_lora_a, w3_lora_a],
                            [w1_lora_b, w2_lora_b, w3_lora_b])
        else:
            super().set_lora(index, lora_a, lora_b)


def refresh_all_lora_classes():
    ascend_classes = (
        AscendQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithShardedLoRA,
        AscendQKVParallelLinearWithShardedLoRA,
        AscendFusedMoEWithLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Append the Ascend classes in a deterministic order.
    vllm.lora.utils._all_lora_classes.discard(FusedMoEWithLoRA)
    vllm.lora.utils._all_lora_classes = (
        *vllm.lora.utils._all_lora_classes,
        *ascend_classes,
    )
