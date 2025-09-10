from typing import Optional, Union

import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parameter import Parameter
from vllm.distributed import split_tensor_along_last_dim
from vllm.forward_context import get_forward_context


def torchair_oproj_tp_forward(
    self,
    input_: torch.Tensor,
) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
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
        cu_tokens_across_dp_cpu = forward_context.dp_metadata.cu_tokens_across_dp_cpu
        prefix_array = cu_tokens_across_dp_cpu.cpu().numpy()
        global_batch_size = np.concatenate(
            ([prefix_array[0]], np.diff(prefix_array)))
        tp_group_id = self.dp_rank // self.tp_size
        tp_group_batchsize = global_batch_size[tp_group_id * self.tp_size: tp_group_id * self.tp_size + self.tp_size]
        total_batch_size = sum(tp_group_batchsize)

        # Reshape for all-to-all communication
        send_buf = (
            input_parallel.reshape(-1, self.tp_size, chunk_size)
            .transpose(0, 1)
            .contiguous()
            .view(-1))
        # Create receive buffer
        recv_buf = torch.zeros(
            total_batch_size * chunk_size,
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
            group=self.comm_group.device_group)
    else:
        total_batch_size = local_batch_size * self.tp_size

        # Reshape tensor for efficient cross-device transfer:
        # [batch, dim] -> [tp_size, batch, chunk] -> flattened
        send_buf = (input_parallel.reshape(-1,
                                        self.tp_size, chunk_size).transpose(
                                            0, 1).contiguous().view(-1))

        # Create receive buffer
        recv_buf = torch.empty(total_batch_size * chunk_size,
                            dtype=input_parallel.dtype,
                            device=input_parallel.device)

        # Perform all-to-all communication
        dist.all_to_all_single(recv_buf,
                            send_buf,
                            group=self.comm_group.device_group)
        
    input_parallel = recv_buf.view(total_batch_size, chunk_size)

    # Only fuse bias add for rank 0 to avoid duplicate bias addition in TP>1
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    assert self.quant_method is not None
    output_parallel = self.quant_method.apply(self,
                                                input_parallel,
                                                bias=bias_)

    if with_prefill:
        # prepare all-reduce data
        output = torch.empty(
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
                output, 
                recv_chunks, 
                op=dist.ReduceOp.SUM, 
                group=self.comm_group.device_group)
        
    else:
        # otp-specific: Combine partial results across devices
        output = self.comm_group.reduce_scatter(output_parallel, dim=0)

    # Handle bias return based on configuration
    output_bias = self.bias if self.skip_bias_add else None
    if not self.return_bias:
        return output
    return output, output_bias
