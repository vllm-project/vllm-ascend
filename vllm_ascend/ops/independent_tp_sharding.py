from dataclasses import dataclass
from typing import Optional, Union, Literal
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase
from vllm.distributed.parallel_state import get_dp_group, GroupCoordinator
from vllm.distributed.utils import (divide, split_tensor_along_last_dim)
from vllm.model_executor.layers.linear import (
    set_weight_attrs, WEIGHT_LOADER_V2_SUPPORTED)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.forward_context import get_forward_context

ExecuteType = Literal["eager", "graph"]

@dataclass
class ExecutionConfig:
    prefill_mode: ExecuteType = "eager"  # 'eager' 或 'graph'
    decode_mode: ExecuteType = "eager"  # 'eager' 或 'graph'

    def __post_init__(self):
        if self.prefill_mode not in ["eager", "graph"]:
            raise ValueError(f"Invalid prefill_mode: {self.prefill_mode}, must be 'eager' or 'graph'")
        if self.decode_mode not in ["eager", "graph"]:
            raise ValueError(f"Invalid decode_mode: {self.decode_mode}, must be 'eager' or 'graph'")

class DownProjectionParallelLinear(LinearBase):
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
        tp_group_cls: GroupCoordinator = None,
        execute_config: ExecutionConfig = None
    ):  
        assert tp_group_cls is not None
        self.tp_group_cls = tp_group_cls
        self.tp_rank = self.tp_group_cls.rank_in_group
        self.tp_size = self.tp_group_cls.world_size
        assert self.tp_size > 1
        self.dp_rank = get_dp_group().rank_in_group

        self.execute_config = execute_config or ExecutionConfig()
        self.prefill_mode = self.execute_config.prefill_mode
        self.decode_mode = self.execute_config.decode_mode
        
        # Divide the weight matrix along the first dimension.
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
                weight_shape[input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)
        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
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
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()
        
        # prefill or decode
        forward_context = get_forward_context()
        with_prefill = forward_context.with_prefill

        # Prepare tensors for all-to-all communication
        local_batch_size = input_parallel.size(0)
        chunk_size = self.input_size_per_partition

        if with_prefill:
            enable_graph_mode = (self.prefill_mode == "graph")
        else:
            enable_graph_mode = (self.decode_mode == "graph")
        
        if enable_graph_mode:
            total_batch_size = local_batch_size * self.tp_size
            # Reshape for all-to-all communication
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
                recv_buf, send_buf, group=self.tp_group_cls.device_group)
            
            all_to_all_result = recv_buf.view(total_batch_size, chunk_size)
        else:
            cu_tokens_across_dp_cpu = forward_context.dp_metadata.cu_tokens_across_dp_cpu
            prefix_array = cu_tokens_across_dp_cpu.numpy()
            global_batch_size = np.concatenate(
                ([prefix_array[0]], np.diff(prefix_array)))
            tp_group_id = self.dp_rank // self.tp_size
            tp_group_batchsize = global_batch_size[tp_group_id * self.tp_size: tp_group_id * self.tp_size + self.tp_size]
            tp_total_batchsize = sum(tp_group_batchsize)

            # Reshape for all-to-all communication
            send_buf = (
                input_parallel.reshape(-1, self.tp_size, chunk_size)
                .transpose(0, 1)
                .contiguous()
                .view(-1))

            # Create receive buffer
            recv_buf = torch.empty(
                tp_total_batchsize * chunk_size,
                dtype=input_parallel.dtype,
                device=input_parallel.device)

            # Create split array
            recv_splits = [size * chunk_size for size in tp_group_batchsize]
            send_splits = [local_batch_size * chunk_size] * self.tp_size

            # Perform all-to-all communication
            dist.all_to_all_single(
                recv_buf, 
                send_buf,
                recv_splits,
                send_splits,
                group=self.tp_group_cls.device_group)

            all_to_all_result = recv_buf.view(tp_total_batchsize, chunk_size)

        # Matrix multiply with quantized method
        assert self.quant_method is not None
        # Only fuse bias add for rank 0 to avoid duplicate bias addition in TP>1
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(
            self, all_to_all_result, bias=bias_)
        
        if enable_graph_mode:
            # Reduce-scatter the results across devices
            final_result = self.tp_group_cls.reduce_scatter(output_parallel, dim=0)
        else:
            # prepare all-reduce data
            final_result = torch.empty(
                    local_batch_size, 
                    output_parallel.size(1), 
                    dtype=output_parallel.dtype, 
                    device=output_parallel.device)

            recv_chunks = []
            start_idx = 0
            for size in tp_group_batchsize:
                chunk = output_parallel[start_idx:start_idx + size, :]
                recv_chunks.append(chunk.contiguous())
                start_idx += size

            # Reduce-scatter the results across devices
            dist.reduce_scatter(
                    final_result, 
                    recv_chunks, 
                    op=dist.ReduceOp.SUM, 
                    group=self.tp_group_cls.device_group)
        
        # Handle bias return based on configuration
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return final_result
        
        return final_result, output_bias