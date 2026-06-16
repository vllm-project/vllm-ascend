import torch
import vllm
from torch import nn
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace
from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.platforms import current_platform

from vllm_ascend.ops.linear import (
    AscendColumnParallelLinear,
    AscendMergedColumnParallelLinear,
    AscendQKVParallelLinear,
)


def _mcp_apply_npu(x, bias, layer):
    """
    NPU-specific implementation of _mcp_apply for ColumnParallelLinearWithLoRA.
    
    This function differs from the vLLM's _mcp_apply by NOT performing all_gather
    on the buffer when fully_sharded_loras=False. On NPU, when LoRA A is not
    sharded, each TP rank already has the full rank, so no all_gather is needed
    on the rank dimension.
    
    For `ColumnParallelLinearWithLoRA` or classes that inherit from
    `ColumnParallelLinearWithLoRA`, they share the same `apply` logic.
    """
    assert (
        layer.n_slices
        == len(layer.lora_a_stacked)
        == len(layer.lora_b_stacked)
        == len(layer.output_slices)
    )

    output = layer.base_layer.quant_method.apply(layer.base_layer, x, bias)

    x = x.view(-1, x.shape[-1])
    output, out_orig_shape = output.view(-1, output.shape[-1]), output.shape

    # Since communication is needed, the buffer is directly initialized as a
    # tensor rather than a tuple of tensor.
    local_lora_rank = layer.lora_a_stacked[0].shape[2]
    buffer_shape = (layer.n_slices, x.shape[0], local_lora_rank)
    # Under torch.compile, the local-rank-1 fully-sharded path can otherwise
    # get lowered to a reinterpret view with a non-canonical layout. The
    # Triton shrink op mutates this buffer in place and expects the standard
    # contiguous [slice, token, rank] stride contract.
    buffers = torch.empty_strided(
        buffer_shape,
        (x.shape[0] * local_lora_rank, local_lora_rank, 1),
        dtype=torch.float32,
        device=x.device,
    )
    buffers.zero_()

    shrunk_buffers: torch.Tensor | None = layer.punica_wrapper.add_shrink(
        buffers, x, layer.lora_a_stacked, 1.0
    )

    if not current_platform.can_update_inplace():
        buffers = shrunk_buffers

    # Only all_gather when LoRA A is sharded (fully_sharded_loras=True)
    # When fully_sharded_loras=False, LoRA A is not sharded and each TP rank
    # has the full rank, so no all_gather is needed on the rank dimension.
    if layer.lora_config.fully_sharded_loras:
        buffers = tensor_model_parallel_all_gather(buffers)

    lora_output: torch.Tensor | None = layer.punica_wrapper.add_expand(
        output,
        buffers,
        layer.lora_b_stacked,
        layer.output_slices,
        offset_start=0,
        add_input=True,
    )

    if not current_platform.can_update_inplace():
        output = lora_output

    output = output.view(*out_orig_shape)
    # now have column partitioned and packed output
    return output


class AscendColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    """NPU-specific ColumnParallelLinearWithLoRA that uses _mcp_apply_npu."""
    
    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply_npu(x, bias, self)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        if type(source_layer) is AscendColumnParallelLinear:
            return True
        if isinstance(source_layer, AscendMergedColumnParallelLinear):
            if len(packed_modules_list) != 1:
                return False
            # Exclude layers with 3+ output sizes - those are handled by
            # MergedColumnParallelLinearVariableSliceWithLoRA since this
            # class's slice_lora_b assumes exactly 2 slices.
            return not (
                hasattr(source_layer, "output_sizes")
                and len(source_layer.output_sizes) >= 3
            )
        return False


class AscendMergedColumnParallelLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
    """NPU-specific MergedColumnParallelLinearWithLoRA that uses _mcp_apply_npu."""
    
    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply_npu(x, bias, self)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        # Only support effectively unsharded subclasses here. Sharded
        # subclasses may have custom communication semantics that the generic
        # merged-column LoRA path does not know how to preserve.
        return type(source_layer) is AscendMergedColumnParallelLinear and len(packed_modules_list) == 2


class AscendQKVParallelLinearWithLoRA(QKVParallelLinearWithLoRA):
    """NPU-specific QKVParallelLinearWithLoRA that uses _mcp_apply_npu."""
    
    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply_npu(x, bias, self)

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
    """NPU-specific MergedQKVParallelLinearWithLoRA that uses _mcp_apply_npu."""
    
    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply_npu(x, bias, self)

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
    """NPU-specific MergedQKVParallelLinearWithShardedLoRA that uses _mcp_apply_npu."""
    
    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply_npu(x, bias, self)

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
    """NPU-specific QKVParallelLinearWithShardedLoRA that uses _mcp_apply_npu."""
    
    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _mcp_apply_npu(x, bias, self)

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


def refresh_all_lora_classes():
    ascend_classes = (
        AscendColumnParallelLinearWithLoRA,
        AscendMergedColumnParallelLinearWithLoRA,
        AscendQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithLoRA,
        AscendMergedQKVParallelLinearWithShardedLoRA,
        AscendQKVParallelLinearWithShardedLoRA,
    )
    # vLLM #35077 changed _all_lora_classes from set to ordered tuple.
    # Insert the Ascend classes at the beginning so they are checked first
    # before the generic vLLM LoRA classes. This is important because
    # AscendQKVParallelLinear is a subclass of QKVParallelLinear, and we want
    # the Ascend-specific LoRA wrappers to be used for Ascend layers.
    vllm.lora.utils._all_lora_classes = (
        *ascend_classes,
        *vllm.lora.utils._all_lora_classes,
    )
