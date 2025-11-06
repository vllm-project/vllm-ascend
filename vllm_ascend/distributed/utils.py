import os

import torch
import torch.distributed as dist

from vllm_ascend.distributed.parallel_state import get_p_tp_group


def kv_alltoall_and_rearrange(pd_tp_ratio: int, key: torch.Tensor,
                              value: torch.TensorType):
    if pd_tp_ratio <= 1:
        return None, None
    elif key is None or value is None:
        raise ValueError("key or value is None")
    k_output = alltoall_and_rearrange(pd_tp_ratio, key)
    v_output = alltoall_and_rearrange(pd_tp_ratio, value)
    return k_output, v_output


def alltoall_and_rearrange(tp_ratio: int, input_tensor: torch.Tensor):
    num_kv_heads = input_tensor.size(1)
    output_tensor = torch.zeros_like(input_tensor)
    dist.all_to_all_single(output_tensor,
                           input_tensor,
                           group=get_p_tp_group().device_group)
    input_tensor = 0
    result = rearrange_output(output_tensor, tp_ratio, num_kv_heads)
    output_tensor = 0
    return result


def rearrange_output(base_output: torch.Tensor, cut_num: int,
                     num_kv_heads: int):
    size_0 = base_output.size(0)
    if size_0 % cut_num != 0:
        raise ValueError(
            f"The size of dim 0 [{size_0}] must be divisible by the cut_num [{cut_num}]"
        )
    chunk_size = size_0 // cut_num
    reshaped = base_output.view(cut_num, chunk_size, -1)
    transposed = reshaped.transpose(0, 1)
    return transposed.contiguous().view(size_0, num_kv_heads, -1)


def align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset):]


def get_transfer_timeout_value():
    ascend_transfer_timeout = os.getenv("ASCEND_TRANSFER_TIMEOUT", "")
    if len(ascend_transfer_timeout) > 0:
        return int(ascend_transfer_timeout)
    hccl_rdma_timeout = int(os.getenv('HCCL_RDMA_TIMEOUT',
                                      '20'))  # type: ignore
    hccl_rdma_retry_cnt = int(os.getenv('HCCL_RDMA_RETRY_CNT',
                                        '7'))  # type: ignore
    return int((4.096 * (2**hccl_rdma_timeout)) * hccl_rdma_retry_cnt // 1000 +
               3000)
