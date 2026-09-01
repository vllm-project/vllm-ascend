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
weight_view = weight.transpose(1, 2)
counts = torch.tensor([m] + [0] * (groups - 1), dtype=torch.int64, device="npu")
output = torch_npu.npu_grouped_matmul(
    [x],
    [weight_view],
    group_list=counts,
    split_item=2,
    group_type=0,
    group_list_type=1,
)[0]
torch.npu.synchronize()
if output.shape != (m, n):
    raise RuntimeError(f"unexpected output shape: {output.shape}")
