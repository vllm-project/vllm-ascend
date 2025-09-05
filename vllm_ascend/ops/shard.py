from typing import Optional, Union

import torch
from torch.nn.parameter import Parameter
from vllm.distributed import divide
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.distributed.parallel_state import get_o_shard_group


@CustomOp.register("row_shard_linear")
class RowShardLinear(LinearBase):
    """Linear layer with row shard storage.

    The linear layer is defined as Y = XA + b. A is shard stored along
    its first dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.self_attn.o_proj)
        return_bias: If true, return bias together with outputs in forward pass.
    """
    work: Optional[torch.distributed.Work]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        # Divide the weight matrix along the first dimension.
        self.tp_rank = get_o_shard_group().rank_in_group
        self.tp_size = get_o_shard_group().world_size
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias)

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader)
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        group_for_shard = get_o_shard_group()
        tp_rank = group_for_shard.rank_in_group
        input_dim = getattr(param, "input_dim", None)

        assert not getattr(param, "use_bitsandbytes_4bit", False)
        assert not getattr(param, "is_sharded_weight", False)
        assert not getattr(param, "is_gguf_weight", False)
        assert not getattr(param, "is_gguf_weight_type", False)

        param_data = param.data
        if input_dim is not None:
            shard_size = param_data.shape[input_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def forward(
        self,
        input,
        is_prefill=True,
        is_force_scatter=False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        # Matrix multiply.
        assert self.quant_method is not None
        if self.work is not None:
            self.work.wait()
            self.work = None
        bias_ = None if self.skip_bias_add else self.bias
        output = self.quant_method.apply(self, input, bias=bias_)
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        return s
