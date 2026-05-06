import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
    VocabParallelEmbeddingWithLoRA,
)
from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA
from vllm.lora.layers.utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace

from vllm_ascend.ops.linear import (
    AscendColumnParallelLinear,
    AscendMergedColumnParallelLinear,
    AscendQKVParallelLinear,
    AscendReplicatedLinear,
    AscendRowParallelLinear,
)
from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding

from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE

class AscendColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendColumnParallelLinear


class AscendMergedColumnParallelLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendMergedColumnParallelLinear


class AscendRowParallelLinearWithLoRA(RowParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendRowParallelLinear


class AscendVocabParallelEmbeddingWithLoRA(VocabParallelEmbeddingWithLoRA):
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return type(source_layer) is AscendVocabParallelEmbedding


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


class AscendReplicatedLinearWithLoRA(ReplicatedLinearWithLoRA):
    # ReplicatedLinear should always be replaced, regardless of the fully
    # sharded LoRAs setting, because it is, by definition, copied per GPU.
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendReplicatedLinear


class AscendColumnParallelLinearWithShardedLoRA(ColumnParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendColumnParallelLinear


class AscendMergedColumnParallelLinearWithShardedLoRA(MergedColumnParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendMergedColumnParallelLinear


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


class AscendRowParallelLinearWithShardedLoRA(RowParallelLinearWithShardedLoRA):
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is AscendRowParallelLinear


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
        return type(source_layer) is AscendFusedMoE and len(packed_modules_list) == 2

    def __init__(self, base_layer: FusedMoE) -> None:
        from vllm.lora.layers.base import BaseLayerWithLoRA
        from vllm.lora.layers.utils import _get_lora_device

        BaseLayerWithLoRA.__init__(self)
        self.base_layer = base_layer

        assert not self.base_layer.use_ep, (
            "EP support for Fused MoE LoRA is not implemented yet."
        )
        assert not self.base_layer.quant_method.is_monolithic, (
            "Monolithic kernels are not supported for Fused MoE LoRA."
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.device = _get_lora_device(base_layer)
        self._w13_slices = 2 if base_layer.moe_config.is_act_and_mul else 1
        self._inject_lora_into_fused_moe()

    def _inject_lora_into_fused_moe(self):
        self.base_layer.ensure_moe_quant_config_init()
        self.base_layer.quant_method._lora_enabled = True
        self.base_layer.quant_method._lora_layer = self

def refresh_all_lora_classes():
    vllm.lora.utils._all_lora_classes.add(AscendColumnParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendMergedColumnParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendRowParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendVocabParallelEmbeddingWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendQKVParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendMergedQKVParallelLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendColumnParallelLinearWithShardedLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendMergedColumnParallelLinearWithShardedLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendMergedQKVParallelLinearWithShardedLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendQKVParallelLinearWithShardedLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendRowParallelLinearWithShardedLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendReplicatedLinearWithLoRA)
    vllm.lora.utils._all_lora_classes.discard(FusedMoEWithLoRA)
    vllm.lora.utils._all_lora_classes.add(AscendFusedMoEWithLoRA)
