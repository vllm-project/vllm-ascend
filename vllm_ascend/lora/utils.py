import torch
import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.lora.layers import MergedColumnParallelLinearWithLoRA, MergedQKVParallelLinearWithLoRA
from vllm.lora.layers.utils import _not_fully_sharded_can_replace
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.platforms import current_platform

from vllm_ascend.lora.fused_moe import (
    AscendFusedMoE3DWithLoRA,
    AscendFusedMoEWithLoRA,
)
from vllm_ascend.ops.linear import AscendQKVParallelLinear


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

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        if hasattr(self, "lora_a_packed"):
            self.lora_a_packed[index].zero_()

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

    def _apply_lora_to_output(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
            packed_lora_a=self.lora_a_packed,
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        if original_shape is not None:
            output = output.reshape(original_shape)
        return output


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


def refresh_all_lora_classes():
    ascend_classes = (
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
