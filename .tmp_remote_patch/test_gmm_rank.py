import os
import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op


rank = int(sys.argv[1])
m = int(sys.argv[2])
k = int(sys.argv[3])
groups = 8
enable_custom_op()
if len(sys.argv) >= 5 and sys.argv[4] == "system":
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        "/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_xllm_math"
    )
torch.npu.set_device(0)
x = torch.ones((m, k), dtype=torch.float16).npu()
weight = torch.ones((groups, rank, k), dtype=torch.float16).npu()
group_list = torch.zeros((groups, 2), dtype=torch.int32)
group_list[0] = torch.tensor([0, m], dtype=torch.int32)
y = torch.ops._C_ascend.moe_lora_grouped_matmul(x, weight, group_list.npu())
torch.npu.synchronize()
print(y.shape, y[0, 0].cpu().item())
