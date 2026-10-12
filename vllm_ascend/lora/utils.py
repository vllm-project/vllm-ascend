import torch
import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.layers.utils import _not_fully_sharded_can_replace
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)
from vllm_ascend.ops.linear import AscendQKVParallelLinear


def _apply_packed_lora(layer, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    original_shape = output.shape if output.ndim == 3 else None
    if original_shape is not None and x.ndim == 3:
        x, output = x.flatten(0, 1), output.flatten(0, 1)
    lora_output = layer.punica_wrapper.add_lora_linear(
        output,
        x,
        layer.lora_a_stacked,
        layer.lora_b_stacked,
        1.0,
        layer.output_slices,
        packed_lora_a=getattr(layer, "lora_a_packed", None),
        packed_lora_b=layer.lora_b_packed,
    )
    if not current_platform.can_update_inplace():
        output = lora_output
    return output.reshape(original_shape) if original_shape is not None else output


class _PackedLoRAAWeightsMixin(MergedColumnParallelLinearWithLoRA):
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        rank = self.lora_a_stacked[0].size(2)
        self.lora_a_packed = torch.zeros(
            max_loras,
            1,
            self.n_slices * rank,
            self.input_size,
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        self.lora_b_packed = torch.zeros(
            max_loras,
            1,
            self.n_slices * rank,
            sum(self.output_slices),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        self.lora_a_packed[index].zero_()
        self.lora_b_packed[index].zero_()

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ) -> None:
        super().set_lora(index, lora_a, lora_b)
        rank = self.lora_a_stacked[0].size(2)
        for slice_index, slice_weight in enumerate(self.lora_a_stacked):
            packed_slice = self.lora_a_packed[index, 0].narrow(0, slice_index * rank, rank)
            packed_slice.copy_(slice_weight[index, 0], non_blocking=True)
        packed_b = self.lora_b_packed[index, 0].zero_()
        offset = 0
        for slice_index, slice_weight in enumerate(self.lora_b_stacked):
            out_size = self.output_slices[slice_index]
            block = packed_b.narrow(0, slice_index * rank, rank).narrow(1, offset, out_size)
            block.copy_(
                slice_weight[index, 0, :out_size].transpose(0, 1),
                non_blocking=True,
            )
            offset += out_size

    def _apply_lora_to_output(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return _apply_packed_lora(self, x, output)


class AscendMergedColumnParallelLinearWithLoRA(_PackedLoRAAWeightsMixin):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            lora_config.max_loras == 1
            and type(source_layer) is maybe_get_oot_by_class(MergedColumnParallelLinear)
            and len(packed_modules_list) == 2
        )


class AscendMergedQKVParallelLinearWithLoRA(_PackedLoRAAWeightsMixin, MergedQKVParallelLinearWithLoRA):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return (
            lora_config.max_loras == 1
            and type(source_layer) is AscendQKVParallelLinear
            and len(packed_modules_list) == 3
        )


class _TransposedLoRABMixin(BaseLinearLayerWithLoRA):
    """Keep a contiguous [rank, output] B copy for the Triton expand kernel."""

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        rank = self.lora_a_stacked[0].size(2)
        self.lora_b_packed = torch.zeros(
            max_loras,
            1,
            rank,
            self.output_slices[0],
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        self.lora_b_packed[index].zero_()

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ) -> None:
        super().set_lora(index, lora_a, lora_b)
        self.lora_b_packed[index, 0].copy_(
            self.lora_b_stacked[0][index, 0].transpose(0, 1),
            non_blocking=True,
        )

    def _apply_lora_to_output(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return _apply_packed_lora(self, x, output)


class _SingleLinearLoRAWrapper(_TransposedLoRABMixin):
    base_layer_cls: type[nn.Module]

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None,
    ) -> bool:
        return lora_config.max_loras == 1 and type(source_layer) is maybe_get_oot_by_class(cls.base_layer_cls)


class AscendRowParallelLinearWithLoRA(_SingleLinearLoRAWrapper, RowParallelLinearWithLoRA):
    base_layer_cls = RowParallelLinear


class AscendColumnParallelLinearWithLoRA(_SingleLinearLoRAWrapper, ColumnParallelLinearWithLoRA):
    base_layer_cls = ColumnParallelLinear


def refresh_all_lora_classes():
    ascend_classes = (
        AscendRowParallelLinearWithLoRA,
        AscendColumnParallelLinearWithLoRA,
        AscendMergedColumnParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendFusedMoEWithLoRA,
        AscendFusedMoE3DWithLoRA,
    )
    existing_classes = tuple(cls for cls in vllm.lora.utils._all_lora_classes if cls not in ascend_classes)
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *existing_classes,
    )
