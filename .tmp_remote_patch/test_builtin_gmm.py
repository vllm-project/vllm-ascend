import sys

import torch
import torch_npu


n = int(sys.argv[1])
m = int(sys.argv[2])
k = int(sys.argv[3])
groups = 8
torch.npu.set_device(0)
x = torch.ones((m, k), dtype=torch.float16, device="npu")
# Keep the checkpoint-compatible [G, N, K] allocation and expose only a
# transposed metadata view to the built-in grouped matmul.
weight_nk = torch.ones((groups, n, k), dtype=torch.float16, device="npu")
weight_kn = weight_nk.transpose(1, 2)
group_list = torch.zeros((groups,), dtype=torch.int64, device="npu")
group_list[0] = m
y = torch_npu.npu_grouped_matmul(
    [x],
    [weight_kn],
    group_list=group_list,
    split_item=3,
    group_type=0,
    group_list_type=1,
)[0]
torch.npu.synchronize()
print(y.shape, y[0, 0].cpu().item(), weight_kn.is_contiguous())
