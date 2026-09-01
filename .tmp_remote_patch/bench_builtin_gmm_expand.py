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


m, n, groups = map(int, sys.argv[1:4])
distribution = sys.argv[4]
weight_dtype = getattr(torch, sys.argv[5]) if len(sys.argv) > 5 else torch.float16
rank = 16
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
x_fp32 = torch.randn((m, rank), dtype=torch.float32, device="npu") * 0.05
x_low = x_fp32.to(weight_dtype)
weight = torch.randn((groups, n, rank), dtype=weight_dtype, device="npu") * 0.05
weight_kn = weight.transpose(1, 2)
bgmv_out = torch.zeros((m, n), dtype=weight_dtype, device="npu")


def gmm_with_input(inputs):
    return torch_npu.npu_grouped_matmul(
        [inputs],
        [weight_kn],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=1,
    )[0]


def gmm():
    return gmm_with_input(x_low)


def cast_only():
    return x_fp32.to(weight_dtype)


def cast_gmm():
    return gmm_with_input(x_fp32.to(weight_dtype))


def bgmv():
    torch.ops._C_ascend.bgmv_expand(x_fp32, weight, indices, bgmv_out, 0, n)
    return bgmv_out


try:
    gmm_out = gmm()
    gmm_supported = True
except Exception as error:
    gmm_supported = False
    print(
        {
            "m": m,
            "n": n,
            "groups": groups,
            "distribution": distribution,
            "weight_dtype": str(weight_dtype),
            "input_dtype": str(x_low.dtype),
            "gmm_supported": False,
            "error": repr(error),
        }
    )

if gmm_supported:
    bgmv()
    torch.npu.synchronize()
    error = (gmm_out.float() - bgmv_out.float()).abs()
    max_abs = error.max().cpu().item()
    mean_abs = error.mean().cpu().item()
    ref_max_abs = bgmv_out.float().abs().max().cpu().item()
    cast_p50, cast_max = measure(cast_only)
    gmm_p50, gmm_max = measure(gmm)
    cast_gmm_p50, cast_gmm_max = measure(cast_gmm)
    bgmv_out.zero_()
    bgmv_p50, bgmv_max = measure(bgmv)
    print(
        {
            "m": m,
            "n": n,
            "groups": groups,
            "distribution": distribution,
            "weight_dtype": str(weight_dtype),
            "input_dtype": str(x_low.dtype),
            "gmm_supported": True,
            "weight_view_contiguous": weight_kn.is_contiguous(),
            "max_abs_vs_bgmv": max_abs,
            "mean_abs_vs_bgmv": mean_abs,
            "reference_max_abs": ref_max_abs,
            "cast_p50_us": cast_p50,
            "cast_max_us": cast_max,
            "gmm_p50_us": gmm_p50,
            "gmm_max_us": gmm_max,
            "cast_gmm_p50_us": cast_gmm_p50,
            "cast_gmm_max_us": cast_gmm_max,
            "bgmv_p50_us": bgmv_p50,
            "bgmv_max_us": bgmv_max,
            "speedup": bgmv_p50 / gmm_p50,
            "cast_gmm_speedup": bgmv_p50 / cast_gmm_p50,
        }
    )
