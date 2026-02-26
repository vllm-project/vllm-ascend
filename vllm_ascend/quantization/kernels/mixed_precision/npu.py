import torch
import torch_npu
from vllm.model_executor.layers.quantization.kernels.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.scalar_type import scalar_types

from vllm_ascend.quantization.utils import unpack_from_int32
from vllm_ascend.utils import maybe_trans_nz


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


class AscendW4A8LinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.int4]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not torch.npu.is_available():
            return False, "Ascend W4A8 only supported on NPU devices"
        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return False, f"Unsupported quant type {c.weight_type}"
        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide "
                f"the number of input features ({c.full_weight_shape[0]})",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = getattr(layer, self.w_q_name).data
        weight = weight.transpose(0, 1).contiguous()
        weight = maybe_trans_nz(weight)

        # Pack int4 values (stored as int8 in [-8, 7]) into NPU int4pack format.
        weight = torch_npu.npu_quantize(
            weight.to(torch.float32),
            torch.tensor([1.0]).npu(),
            None,
            torch.quint4x2,
            -1,
            False,
        )
        getattr(layer, self.w_q_name).data = weight

        scale = getattr(layer, self.w_s_name).data
        getattr(layer, self.w_s_name).data = scale.contiguous().to(torch.float32)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight, scale, _, _ = self._get_weight_params(layer)
        output = torch_npu.npu_weight_quant_batchmatmul(
            x=x,
            weight=weight,
            antiquant_scale=scale.to(x.dtype),
            antiquant_offset=None,
            antiquant_group_size=self.config.group_size,
            bias=bias,
        )
        return output
