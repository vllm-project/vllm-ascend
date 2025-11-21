from abc import abstractmethod
from typing import Optional

import torch
from omni.models.common.layers.linear import (
    MergedColumnParallelFlashCommLinear, RowParallelFlashCommLinear)
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)


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
        output = self.quant_method.apply(self,
                                         x,
                                         x_transform=x_transform,
                                         is_prefill=is_prefill)
        if self.down_proj.tp_size > 1:
            if reduce_type == "AR":
                output = tensor_model_parallel_all_reduce(output)
            elif reduce_type == "RS":
                output = tensor_model_parallel_reduce_scatter(output)
        return output
