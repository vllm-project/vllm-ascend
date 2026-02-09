import torch
import torch_npu
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.scalar_type import scalar_types

from vllm_ascend.quantization.utils import unpack_from_int32


class AscendwNa16LinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not torch.npu.is_available():
            return False, "Ascend wNa16 only supported on NPU devices"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Get original shape before transpose
        weight_shape = layer.weight_packed.data.shape

        pack_factor = 8
        num_bits = 4
        if self.config.weight_type in [scalar_types.uint4, scalar_types.uint4b8]:
            num_bits = 4
        elif self.config.weight_type in [scalar_types.uint8, scalar_types.uint8b128]:
            num_bits = 8

        # Unpack from int32 to int8 (with int4 range)
        unpacked_weight = unpack_from_int32(
            weight=layer.weight_packed.data,
            shape=torch.Size([weight_shape[0], weight_shape[1] * pack_factor]),
            num_bits=num_bits,
            packed_dim=1,
        )

        # Transpose: [n, k] -> [k, n]
        unpacked_weight = unpacked_weight.transpose(0, 1).contiguous().int()

        # Repack to int32 using NPU int4 packing
        layer.weight_packed.data = torch_npu.npu_convert_weight_to_int4pack(unpacked_weight)

        # Transpose scales and offsets: [n, num_groups] -> [num_groups, n]
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).contiguous()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch_npu.npu_weight_quant_batchmatmul(
            x=x,
            weight=layer.weight_packed,
            antiquant_scale=layer.weight_scale,
            antiquant_offset=None,
            antiquant_group_size=self.config.group_size,
            bias=bias,
        )
        return output
