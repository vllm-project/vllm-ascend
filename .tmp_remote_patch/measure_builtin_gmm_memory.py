import sys

import torch
import torch_npu


m = int(sys.argv[1])
n = int(sys.argv[2])
dtype = getattr(torch, sys.argv[3])
groups = int(sys.argv[4]) if len(sys.argv) > 4 else 8
rank = 16
torch.npu.set_device(0)
x = torch.ones((m, rank), dtype=dtype, device="npu")
weight = torch.ones((groups, n, rank), dtype=dtype, device="npu")
counts = torch.tensor([m] + [0] * (groups - 1), dtype=torch.int64, device="npu")
weight_view = weight.transpose(1, 2)
torch.npu.synchronize()
torch.npu.empty_cache()
torch.npu.reset_peak_memory_stats(0)
allocated_before = torch.npu.memory_allocated(0)
reserved_before = torch.npu.memory_reserved(0)
output = torch_npu.npu_grouped_matmul(
    [x],
    [weight_view],
    group_list=counts,
    split_item=2,
    group_type=0,
    group_list_type=1,
)[0]
torch.npu.synchronize()
allocated_after = torch.npu.memory_allocated(0)
reserved_after = torch.npu.memory_reserved(0)
peak_allocated = torch.npu.max_memory_allocated(0)
peak_reserved = torch.npu.max_memory_reserved(0)
output_bytes = output.numel() * output.element_size()
print(
    {
        "m": m,
        "n": n,
        "dtype": str(dtype),
        "output_shape": tuple(output.shape),
        "output_dtype": str(output.dtype),
        "output_contiguous": output.is_contiguous(),
        "output_storage_offset": output.storage_offset(),
        "output_stride": output.stride(),
        "output_bytes": output_bytes,
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "allocated_delta": allocated_after - allocated_before,
        "peak_allocated": peak_allocated,
        "peak_delta": peak_allocated - allocated_before,
        "workspace_peak_upper_bound": max(0, peak_allocated - allocated_before - output_bytes),
        "reserved_before": reserved_before,
        "reserved_after": reserved_after,
        "reserved_delta": reserved_after - reserved_before,
        "peak_reserved": peak_reserved,
    }
)
