import torch
import torch_npu
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

from vllm_ascend.utils import get_weight_prefetch_method, maybe_trans_nz


class AscendDynamicInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not torch.npu.is_available():
            return False, "requires Ascend NPU."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.is_static_input_scheme:
            return False, "AscendDynamicInt8ScaledMMLinearKernel does not support static input quantization scheme."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # cast quantized weight tensors in NZ format for higher inference speed
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        # layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        # layer.weight_offset.data = layer.weight_offset.data.flatten()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=x.dtype,
        )
        return output


class AscendStaticInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not torch.npu.is_available():
            return False, "requires Ascend NPU."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not c.is_static_input_scheme:
            return (
                False,
                "AscendStaticInt8ScaledMMLinearLayerConfig does not support dynamic input quantization scheme.",
            )
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_scale.data = layer.weight_scale.data.to(torch.bfloat16)
        layer.input_scale.data = layer.input_scale.data.to(torch.bfloat16)
        expanding_factor = layer.weight.data.shape[1]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor), requires_grad=False
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor), requires_grad=False
        )

        if layer.input_zero_point is None:
            layer.input_zero_point = torch.nn.Parameter(
                torch.zeros(1, dtype=torch.int8, device=layer.weight.device), requires_grad=False
            )

        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_zero_point.data.repeat(expanding_factor), requires_grad=False
        ).to(layer.aclnn_input_scale.dtype)

        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = torch.flatten(layer.weight_scale.data)
        # layer.weight_offset.data = torch.flatten(layer.weight_offset.data)

        deq_scale = layer.input_scale.data * layer.weight_scale.data
        layer.deq_scale = torch.nn.Parameter(deq_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.int8:
            layer_cls_name = layer.__class__.__name__
            weight_prefetch_method = get_weight_prefetch_method()
            # prefetch qkvo_proj.weight preprocess
            if weight_prefetch_method:
                weight_prefetch_method.maybe_prefetch_attn_weight_preprocess(
                    layer_cls_name=layer_cls_name,
                    weight=layer.weight,
                    start_flag=x,
                )
            try:
                quant_comm_config = layer._quant_comm_config
            except AttributeError:
                quant_comm_config = {}
            comm_fn = quant_comm_config.get("communication_fn")
            enable_flashcomm2_quant_comm = comm_fn is not None and (
                "o_proj" in layer.prefix or "out_proj" in layer.prefix
            )
            if enable_flashcomm2_quant_comm:
                quant_input_x = x.contiguous().view(-1, layer.aclnn_input_scale_reciprocal.size(0))
                quant_x = torch.ops.vllm.quantize(
                    quant_input_x,
                    layer.aclnn_input_scale,
                    layer.aclnn_input_scale_reciprocal,
                    layer.aclnn_input_offset,
                )
                comm_input = quant_x.view(x.size(0), -1)
                assert comm_fn is not None
                x = comm_fn(comm_input)
            else:
                # quant
                x = torch.ops.vllm.quantize(
                    x,
                    layer.aclnn_input_scale,
                    layer.aclnn_input_scale_reciprocal,
                    layer.aclnn_input_offset,
                )

            # prefetch qkvo_proj.weight postprocess
            if weight_prefetch_method:
                weight_prefetch_method.maybe_prefetch_attn_weight_postprocess(
                    layer_cls_name=layer_cls_name,
                    stop_flag=x,
                )

        # quant_bias = layer.quant_bias if tp_rank == 0 else None

        quant_bias = bias

        output = torch_npu.npu_quant_matmul(
            x,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=layer.params_dtype,
        )
        return output
