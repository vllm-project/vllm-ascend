import glob

import torch
import torch_npu


torch.npu.set_device(0)
torch.ops.load_library(glob.glob("build_kernel/vllm_ascend_C*.so")[0])

m, width, top_k, experts, adapters = 16, 32, 2, 4, 2
groups = experts * adapters
gp = (groups + 7) // 8 * 8
cores = torch.npu.get_device_properties(0).vector_core_num
x = torch.arange(m * width, dtype=torch.float16).reshape(m, width).npu()
expanded = torch.arange(m, dtype=torch.int32)
expanded[7] = -1
topk = torch.arange(m, dtype=torch.int32).remainder(experts)
token_lora = torch.arange(m // top_k, dtype=torch.int64).remainder(adapters)
enabled = torch.ones(adapters, dtype=torch.bool, device="npu")

local_count = torch.empty((cores, gp), dtype=torch.int32, device="npu")
core_prefix = torch.empty_like(local_count)
group_total = torch.empty(groups, dtype=torch.int32, device="npu")
group_start = torch.empty(groups, dtype=torch.int32, device="npu")
group_count = torch.empty(groups, dtype=torch.int64, device="npu")
perm = torch.empty((m, 8), dtype=torch.int32, device="npu")
error_per_core = torch.empty((cores, 8), dtype=torch.int32, device="npu")
route_error = torch.empty(8, dtype=torch.int32, device="npu")
grouped = torch.empty_like(x)

torch.ops._C_ascend.moe_lora_prefill_route_allgather(
    x,
    expanded.npu(),
    topk.npu(),
    token_lora.npu(),
    enabled,
    local_count,
    core_prefix,
    group_total,
    group_start,
    group_count,
    perm,
    error_per_core,
    route_error,
    grouped,
    top_k,
    experts,
    0,
)
torch.npu.synchronize()
sink = int(group_total[0].cpu())
assert int(group_count.sum().cpu()) == m
assert int(group_count[0].cpu()) == sink + 1
assert int(route_error[0].cpu()) & 2
assert int(perm[sink, 0].cpu()) == -(2**31)
assert torch.count_nonzero(grouped[sink]).item() == 0

source = torch.randn_like(x)
gathered = torch.empty_like(x)
torch.ops._C_ascend.moe_lora_prefill_gather_by_perm(source, perm, gathered)
torch.npu.synchronize()
assert torch.count_nonzero(gathered[sink]).item() == 0

delta = torch.ones_like(x)
y = torch.zeros_like(x)
torch.ops._C_ascend.moe_lora_prefill_scatter_add(delta, perm, y, 0)
torch.npu.synchronize()
assert torch.count_nonzero(y[7]).item() == 0
print(
    "SENTINEL_GATE_PASS",
    "canonical=16 local=16 positive=15",
    "count_sum=", int(group_count.sum().cpu()),
    "sink=", sink,
    "route_error=", int(route_error[0].cpu()),
)
