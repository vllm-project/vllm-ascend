import os
import ipaddress
import threading
from typing import Optional

import torch
import torch.distributed as dist
from mooncake.engine import TransferEngine  # type: ignore

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


class GlobalTE():

    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: Optional[str]):
        try:
            ip = ipaddress.ip_address(hostname)
            if isinstance(ip, ipaddress.IPv6Address):
                raise RuntimeError(
                    "The backend of mooncake's Ascend Direct Xfer Library currently does not support IPv6."
                )
        except ValueError:
            pass
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    if TransferEngine is None:
                        raise RuntimeError("mooncake is not available")
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(
                        hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(
                            f"TransferEngine initialization failed with ret_value: {ret_value}"
                        )
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True


global_te = GlobalTE()
