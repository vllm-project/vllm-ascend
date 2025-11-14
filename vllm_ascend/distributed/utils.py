import hashlib
import os
import struct
import time

import torch
import torch.distributed as dist
import zmq
from vllm.utils import logger

from vllm_ascend.distributed.parallel_state import get_p_tp_group
from vllm_ascend.utils import vllm_version_is

# ZMQ communication constants
GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
DONE_SENDING_MSG = b"done_sending_msg"


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


def ensure_zmq_send(
        socket: zmq.Socket,  # type: ignore
        data: bytes,
        max_retries: int = 3):
    """Send data over a ZMQ socket with retry logic.
    
    Args:
        socket: ZMQ socket to send data through
        data: Bytes data to send
        max_retries: Maximum number of retry attempts (default: 3)
        
    Raises:
        RuntimeError: If send fails after all retries
    """
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning(
                    f"Send failed: {e}, retrying... ({retries_left} "
                    "attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Send failed after all retries: {e}")
                raise RuntimeError(f"Failed to send data after {max_retries} "
                                   f"retries: {e}")


def ensure_zmq_recv(
        socket: zmq.Socket,  # type: ignore
        poller: zmq.Poller,  # type: ignore
        timeout: float = 1.0,
        max_retries: int = 3) -> bytes:
    """Receive data from a ZMQ socket with retry logic.
    
    Args:
        socket: ZMQ socket to receive data from
        poller: ZMQ poller for timeout detection
        timeout: Timeout in seconds for each receive attempt (default: 1.0)
        max_retries: Maximum number of retry attempts (default: 3)
        
    Returns:
        Received bytes data
        
    Raises:
        RuntimeError: If receive fails after all retries
    """
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning(f"Receive failed: {e}, retrying... "
                               f"({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Receive failed after all retries: {e}")
                raise RuntimeError(f"Failed to receive data after "
                                   f"{max_retries} retries: {e}")


def get_network_utils():
    """Get network utility functions based on vllm version.
    
    Returns:
        Tuple of (get_ip, make_zmq_path, make_zmq_socket) functions
    """
    if vllm_version_is("0.11.0"):
        from vllm.utils import get_ip, make_zmq_path, make_zmq_socket
    else:
        from vllm.utils.network_utils import (get_ip, make_zmq_path,
                                              make_zmq_socket)
    return get_ip, make_zmq_path, make_zmq_socket


def string_to_int64_hash(input_str: str) -> int:
    """Hash a string using SHA-256 and convert it into an int64 integer.
    
    Args:
        input_str: The string to hash
        
    Returns:
        An int64 integer representation of the hash
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value
