import os
import torch
import torch_npu
import torchair

from abc import abstractmethod
from typing import Optional
from vllm.model_executor.layers.quantization.base_config import (QuantizationConfig, 
                                                                 QuantizeMethodBase)
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
    get_tp_group
)
from omni.layers.activation import SiluAndMul
from omni.layers.linear import MergedColumnParallelFlashCommLinear, RowParallelFlashCommLinear

class FusedMLPMethodBase(QuantizeMethodBase):
    """Base method for FusedMLP

    This method and its subclasses do not define create_weights method.
    The weights' creation is handled by submodules.
    This method only implement the apply method.
    """
    def create_weights(self, layer: torch.nn.Module, *weight_args,
                       **extra_weight_attrs):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        x_transform: str = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError


class UnquantizedFusedMLPMethod(FusedMLPMethodBase):
    """MLP method without quantization."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        x_transform: str = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        if x_transform == "AG":
            x = tensor_model_parallel_all_gather(x, dim=0)
        elif x_transform == "A2A":
            x = get_tp_group().all_to_all(x)
        gate_up, _ = layer.gate_up_proj(x, x_transform=None)
        x = layer.act_fn(gate_up)
        x, _ = layer.down_proj(x, reduce_type=None)
        return x

class FusedMLP(torch.nn.Module):
    """FusedMLP layer 
    
    This layer relies on linear layer to create weights and 
    implements optimizations that consider MLP module as a whole.
    For example, fusing the dequant of up_gate, swigu and the quant
    of down (dequant_swiglu_quant fused kernel).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = MergedColumnParallelFlashCommLinear(
            hidden_size,
            [intermediate_size] * 2,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelFlashCommLinear(
            intermediate_size,
            hidden_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

        quant_method: Optional[QuantizeMethodBase] = None

        if quant_config is None:
            quant_method = UnquantizedFusedMLPMethod()
        else:
            quant_method = quant_config.get_quant_method(self, prefix)

        assert quant_method is not None
        assert isinstance(quant_method, FusedMLPMethodBase)
        self.quant_method = quant_method
    
    def forward(self, x, x_transform=None, reduce_type="AR", is_prefill=True):
        output = self.quant_method.apply(self, x, x_transform=x_transform, is_prefill=is_prefill)
        if self.down_proj.tp_size > 1:
            if reduce_type == "AR":
                output = tensor_model_parallel_all_reduce(output)
            elif reduce_type == "RS":
                output = tensor_model_parallel_reduce_scatter(output)
        return output


class W8A8DynamicFusedMLPMethod(FusedMLPMethodBase):
    """Apply dequant_swiglu_quant fused kernel.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.gate_up_proj.weight_scale = torch.nn.Parameter(
            layer.gate_up_proj.weight_scale.data.float(), requires_grad=False)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            x_transform: str = None,
            is_prefill: bool = True,
    ) -> torch.Tensor:
        bias = layer.gate_up_proj.bias if not layer.gate_up_proj.skip_bias_add else None
        x, x_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=None)
        scale_parallel = os.environ.get('SCALE_PARALLEL', '0') == '1'
        if x_transform is not None:
            if x_transform == 'AG':
                if is_prefill or (not scale_parallel):
                    x_scale = get_tp_group().all_gather(x_scale, dim=0)
                else:
                    with torchair.ops.NpuStreamSwitch('scale'):  # CANN包多流接口
                        x_scale = get_tp_group().all_gather(x_scale, dim=0)
                x = get_tp_group().all_gather(x, dim=0)
            elif x_transform == 'A2A':
                if is_prefill or (not scale_parallel):
                    x_scale = get_tp_group().all_to_all(x_scale, transpose=False)
                else:
                    with torchair.ops.NpuStreamSwitch('scale'):  # CANN包多流接口
                        x_scale = get_tp_group().all_to_all(x_scale, transpose=False)
                x = get_tp_group().all_to_all(x)
        y_int32 = torch_npu.npu_quant_matmul(
            x1=x,
            x2=layer.gate_up_proj.weight,
            scale=layer.gate_up_proj.weight_scale,
            pertoken_scale=None,
            bias=None,
            output_dtype=torch.int32
        )
        int_int32, int_scale = torch_npu.npu_dequant_swiglu_quant(
            y_int32,
            weight_scale=layer.gate_up_proj.weight_scale,
            activation_scale=x_scale,
            bias=bias,
            activate_left=True,
            quant_mode=1,
        )

        bias = None if (layer.down_proj.tp_rank > 0 or layer.down_proj.skip_bias_add) else layer.down_proj.bias
        output = torch_npu.npu_quant_matmul(
            x1=int_int32,
            x2=layer.down_proj.weight,
            scale=layer.down_proj.weight_scale,
            pertoken_scale=int_scale,
            bias=bias,
            output_dtype=layer.down_proj.orig_dtype
        )

        return output
