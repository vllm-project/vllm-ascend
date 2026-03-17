import pytest
import torch

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_batch_memcpy(dtype):
    element_size = 2 if dtype == torch.bfloat16 else 4
    device = "npu:0"
    # this is a typical case when used in mamba states copy.
    sizes = torch.tensor([24576, 262144, 24576, 262144], device=device, dtype=torch.int32)

    src_tensors_list = []
    src_addr_list = []
    dst_tensors_list = []
    dst_addr_list = []
    for i in range(len(sizes)):
        src_tensors_list.append(
            torch.rand(sizes[i].item() // element_size, dtype=dtype, device=device)
        )
        src_addr_list.append(src_tensors_list[-1].data_ptr())
        dst_tensors_list.append(
            torch.empty(sizes[i].item() // element_size, dtype=dtype, device=device)
        )
        dst_addr_list.append(dst_tensors_list[-1].data_ptr())
    
    src_addr_list = torch.tensor(src_addr_list, dtype=torch.int64, device=device)
    dst_addr_list = torch.tensor(dst_addr_list, dtype=torch.int64, device=device)

    batch = sizes.shape[0]

    grid = (batch,)
    # using larger block_size to accelerate copy.
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[grid](src_addr_list, dst_addr_list, sizes, BLOCK_SIZE=BLOCK_SIZE)

    for i in range(len(sizes)):
        torch.testing.assert_close(src_tensors_list[i], dst_tensors_list[i], rtol=0, atol=0)
