from typing import Optional, Union
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.linear import (
    set_weight_attrs, WEIGHT_LOADER_V2_SUPPORTED)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm_ascend import ascend_config
from vllm_ascend.distributed.parallel_state import get_otp_group
from vllm_ascend.ascend_config import get_ascend_config


class Oproj_RowParallelLinear(LinearBase):
    """Custom oproj Linear layer with row parallelism.
        Base on vllm.model_executor.layers -> linear.py -> RowParallelLinear

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.down_proj)
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        ascend_config = get_ascend_config()
        self._enable_otp = True \
            if ascend_config.oproj_tensor_parallel_size is not None else False

        # Divide the weight matrix along the first dimension.
        # Determine tensor parallel rank and size based on configuration:
        # Case 1: If oproj_tensor_parallel_size is set, use otp_group for rank/size
        # Case 2: Otherwise, use standard tensor model parallel rank/size
        if self._enable_otp:
            self.tp_rank = get_otp_group().rank_in_group
            self.tp_size = get_otp_group().world_size
        else:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        
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

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2 if self.quant_method.__class__.__name__
                in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError("When not reduce the results, adding bias to the "
                             "results can lead to incorrect results")

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
        tp_rank = self.tp_rank
        tp_size = self.tp_size
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
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

    def weight_loader_v2(self, param: BasevLLMParameter,
                         loaded_weight: torch.Tensor):

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        param.load_row_parallel_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        """Forward pass for oproj layer with tensor parallelism.

        The forward pass has two distinct execution paths:
        1. Standard tensor parallelism (when self._enable_otp=False)
        2. Oproj tensor parallelism (OTP) mode (when self._enable_otp=True)
           which adds all-to-all communication before computation and
           reduce-scatter after computation.
        Args:
            input_: Input tensor of shape [batch_size, input_dim].
            
        Returns:
            Either:
            - Output tensor (if skip_bias_add=False and return_bias=False)
            - Tuple of (output_tensor, bias) if skip_bias_add or return_bias is True
        """
        # Handle input parallelism - split or use as-is
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = self.tp_rank
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()
        
        
        if self._enable_otp: 
            # Prepare tensors for all-to-all communication
            local_batch_size = input_parallel.size(0)
            chunk_size = self.input_size_per_partition
            total_batch_size = local_batch_size * self.tp_size

            # Reshape tensor for efficient cross-device transfer:
            # [batch, dim] -> [tp_size, batch, chunk] -> flattened
            send_buf = (
                input_parallel.reshape(-1, self.tp_size, chunk_size)
                .transpose(0, 1)
                .contiguous()
                .view(-1))
            
            # Create receive buffer
            recv_buf = torch.empty(
                total_batch_size * chunk_size,
                dtype=input_parallel.dtype,
                device=input_.device)
            
            # Perform all-to-all communication
            dist.all_to_all_single(
                recv_buf, send_buf, group=get_otp_group().device_group)
            input_parallel = recv_buf.view(total_batch_size, chunk_size)

        # Matrix multiply with quantized method
        assert self.quant_method is not None
        # Only fuse bias add for rank 0 to avoid duplicate bias addition in TP>1
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(
            self, input_parallel, bias=bias_)

        if self._enable_otp:
            # otp-specific: Combine partial results across devices
            output = get_otp_group().reduce_scatter(output_parallel, dim=0)
        elif self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        # Handle bias return based on configuration
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias