import statistics
import sys
import time

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def measure(fn, warmup=10, iterations=50, repeats=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    values = []
    for _ in range(repeats):
        begin = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.npu.synchronize()
        values.append((time.perf_counter() - begin) * 1e6 / iterations)
    return statistics.median(values), max(values)


m, k, groups = map(int, sys.argv[1:4])
distribution = sys.argv[4]
dtype = getattr(torch, sys.argv[5]) if len(sys.argv) > 5 else torch.float16
enable_custom_op()
torch.npu.set_device(0)
if distribution == "hotspot":
    counts = [m] + [0] * (groups - 1)
else:
    q, r = divmod(m, groups)
    counts = [q + (g < r) for g in range(groups)]
indices = torch.cat(
    [torch.full((count,), g, dtype=torch.int64) for g, count in enumerate(counts)]
).npu()
group_list = torch.tensor(counts, dtype=torch.int64, device="npu")
x = torch.randn((m, k), dtype=dtype, device="npu") * 0.05
weight = torch.randn((groups, 16, k), dtype=dtype, device="npu") * 0.05
weight_kn = weight.transpose(1, 2)
bgmv_out = torch.empty((m, 16), dtype=torch.float32, device="npu")


def gmm():
    return torch_npu.npu_grouped_matmul(
        [x],
        [weight_kn],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=1,
    )[0]


def bgmv():
    torch.ops._C_ascend.bgmv_shrink(x, weight, indices, bgmv_out, 1.0)
    return bgmv_out


gmm_out = gmm()
bgmv()
torch.npu.synchronize()
max_abs = (gmm_out.float() - bgmv_out).abs().max().cpu().item()
gmm_p50, gmm_max = measure(gmm)
bgmv_p50, bgmv_max = measure(bgmv)
print(
    {
        "m": m,
        "k": k,
        "groups": groups,
        "distribution": distribution,
        "dtype": str(dtype),
        "max_abs_vs_bgmv": max_abs,
        "gmm_p50_us": gmm_p50,
        "gmm_max_us": gmm_max,
        "bgmv_p50_us": bgmv_p50,
        "bgmv_max_us": bgmv_max,
        "speedup": bgmv_p50 / gmm_p50,
    }
)
