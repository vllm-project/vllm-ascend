from vllm.model_executor.layers.linear import LinearBase,WEIGHT_LOADER_V2_SUPPORTED,adjust_bitblas_shard
from vllm.forward_context import get_forward_context
from torch.nn.parameter import Parameter, UninitializedParameter


from vllm_ascend.distributed.parallel_state import get_mlp_tensor_model_parallel_world_size, get_mlp_tensor_model_parallel_rank, get_mlp_tp_group
import torch
from typing import Any, Literal, Optional, Union
from vllm.config import QuantizationConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
# yapf: disable
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           BlockQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           PerTensorScaleParameter,
                                           RowvLLMParameter)
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)





class AttnColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj) 
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[list[int]] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        # Divide the weight matrix along the last dimension.
        # self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = get_mlp_tensor_model_parallel_world_size()
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        super().__init__(input_size,
                         output_size,
                         skip_bias_add,
                         params_dtype,
                         quant_config,
                         prefix,
                         return_bias=return_bias)

        self.gather_output = gather_output

        if output_sizes is None:
            output_sizes = [output_size]

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
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_mlp_tensor_model_parallel_rank()
        output_dim = getattr(param, "output_dim", None)

        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
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
            final_shape = list(loaded_weight.shape)
            if output_dim is not None:
                # tp_size = get_tensor_model_parallel_world_size()
                tp_size = get_mlp_tensor_model_parallel_world_size()
                assert final_shape[output_dim] % tp_size == 0
                final_shape[output_dim] = final_shape[output_dim] // tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: Parameter, loaded_weight: torch.Tensor):
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)
        param.load_column_parallel_weight(loaded_weight=loaded_weight)

    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        # self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        # Matrix multiply.
        assert self.quant_method is not None
        forward_context = get_forward_context()
        num_reqs_across_dp = forward_context.num_reqs_across_dp
        print("###########AttnColumnParallelLinear input_",input_.shape, input_.device, input_)
        # input2_ = tensor_model_parallel_all_gather_lm(input_, 0)
        
        gathered_input = [torch.empty(batch_size, input_.size(
            1), dtype=input_.dtype, device=input_.device) for batch_size in num_reqs_across_dp]
        torch.distributed.all_gather(
                gathered_input, input_, group=get_mlp_tp_group().device_group)
        input2_ = torch.cat(gathered_input, dim=0)
        
        print("###########AttnColumnParallelLinear qafter input_",input2_.shape, input2_.device, input2_)
        # print("###########AttnColumnParallelLinear input2_",input2_.shape)
        output = self.quant_method.apply(self, input2_, bias)
        # if self.gather_output:
        #     # All-gather across the partitions.
        #     output = tensor_model_parallel_all_gather(output_parallel)
        # else:
        #     output = output_parallel
        # print("###########AttnColumnParallelLinear output",output.shape)
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={get_mlp_tensor_model_parallel_world_size()}"
        s += f", gather_output={self.gather_output}"
        return s


class AttnRowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

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
        quant_config: Quantization configure.
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
        # self.lm_tp_size = get_mlp_tensor_model_parallel_world_size()
        # self.lm_tp_rank = get_mlp_tensor_model_parallel_rank()
        self.tp_rank = get_mlp_tensor_model_parallel_rank()
        self.tp_size = get_mlp_tensor_model_parallel_world_size()
        # Divide the weight matrix along the first dimension.
        # self.tp_rank = get_tensor_model_parallel_rank()
        # self.tp_size = get_tensor_model_parallel_world_size()
        # self.input_size_per_partition = divide(input_size, self.tp_size)
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
        tp_rank = get_mlp_tensor_model_parallel_rank()
        tp_size = get_mlp_tensor_model_parallel_world_size()
        
        # tp_rank = get_tensor_model_parallel_rank()
        # tp_size = get_tensor_model_parallel_world_size()
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
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # tp_rank = get_tensor_model_parallel_rank()
            tp_rank = get_mlp_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        # if self.reduce_results and self.tp_size > 1:
        #     output = tensor_model_parallel_all_reduce(output_parallel)
        # else:
        #     output = output_parallel
        output = tensor_model_parallel_reduce_scatter_lm(output_parallel, dim=0)

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s

class ATTNMergedColumnParallelLinear(AttnColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
    ):
        self.output_sizes = output_sizes
        tp_size = get_mlp_tensor_model_parallel_world_size()
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        super().__init__(input_size=input_size,
                         output_size=sum(output_sizes),
                         bias=bias,
                         gather_output=gather_output,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix,
                         return_bias=return_bias)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            if loaded_shard_id is not None:
                param.data[loaded_shard_id].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {
                    i: loaded_weight.item()
                    for i, _ in enumerate(self.output_sizes)
                }
            return

        if is_gguf_weight:
            tp_size = get_mlp_tensor_model_parallel_world_size()
            tp_rank = get_mlp_tensor_model_parallel_rank()

            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // tp_size
            start_idx = tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (mlp).
            # (e.g., Phi-3's gate_up_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                shard_size, shard_offset = adjust_bitblas_shard(
                    param, shard_size, shard_offset)

                if use_bitsandbytes_4bit:
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id))

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        tp_rank = get_mlp_tensor_model_parallel_rank()
        tp_size = get_mlp_tensor_model_parallel_world_size()
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)
            shard_size, shard_offset = adjust_bitblas_shard(
                param, shard_size, shard_offset)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * \
                    loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            start_idx = tp_rank * shard_size
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(self, param: BasevLLMParameter,
                                           loaded_weight: torch.Tensor):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0
        shard_offsets: list[tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if isinstance(param, (PackedColumnParameter, PackedvLLMParameter
                                  )) and param.packed_dim == param.output_dim:
                shard_size, shard_offset = \
                    param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset)

            loaded_weight_shard = loaded_weight.narrow(param.output_dim,
                                                       shard_offset,
                                                       shard_size)
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(self,
                         param: BasevLLMParameter,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: Optional[int] = None):
        if loaded_shard_id is None:
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight,
                                                shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id < len(self.output_sizes)

        tp_size = get_mlp_tensor_model_parallel_world_size()

        if isinstance(param, BlockQuantScaleParameter):
            from vllm.model_executor.layers.quantization.fp8 import (
                Fp8LinearMethod, Fp8MoEMethod)
            assert self.quant_method is not None
            assert isinstance(self.quant_method,
                              (Fp8LinearMethod, Fp8MoEMethod))
            weight_block_size = self.quant_method.quant_config.weight_block_size
            assert weight_block_size is not None
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (
                (sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) //
                block_n) // tp_size
            shard_size = ((self.output_sizes[loaded_shard_id] + block_n - 1) //
                          block_n // tp_size)
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
            shard_size = self.output_sizes[loaded_shard_id] // tp_size

        param.load_merged_column_weight(loaded_weight=loaded_weight,
                                        shard_id=loaded_shard_id,
                                        shard_offset=shard_offset,
                                        shard_size=shard_size)
