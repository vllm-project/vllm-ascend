import statistics
import sys
import time

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def measure(fn, warmup=5, iterations=20, repeats=5):
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
    return statistics.median(values), max(values), values


def distribution_counts(m, groups, distribution):
    if distribution == "hotspot":
        return [m] + [0] * (groups - 1)
    if distribution == "longtail":
        weights = [groups - group for group in range(groups)]
        total = sum(weights)
        counts = [m * weight // total for weight in weights]
        counts[0] += m - sum(counts)
        return counts
    quotient, remainder = divmod(m, groups)
    return [quotient + (group < remainder) for group in range(groups)]


m = int(sys.argv[1])
distribution = sys.argv[2]
dtype = getattr(torch, sys.argv[3]) if len(sys.argv) > 3 else torch.float16
groups = int(sys.argv[4]) if len(sys.argv) > 4 else 8
hidden = 4096
intermediate = 2048
rank = 16
torch.manual_seed(17)
enable_custom_op()
torch.npu.set_device(0)

counts_host = distribution_counts(m, groups, distribution)
counts = torch.tensor(counts_host, dtype=torch.int64, device="npu")
indices = torch.cat(
    [torch.full((count,), group, dtype=torch.int64) for group, count in enumerate(counts_host)]
).npu()


def randn(shape):
    return torch.randn(shape, dtype=dtype, device="npu") * 0.02


x_w13 = randn((m, hidden))
x_w2 = randn((m, intermediate))
a_w13_0 = randn((groups, rank, hidden))
a_w13_1 = randn((groups, rank, hidden))
b_w13_0 = randn((groups, intermediate, rank))
b_w13_1 = randn((groups, intermediate, rank))
a_w2 = randn((groups, rank, intermediate))
b_w2 = randn((groups, hidden, rank))


def gmm(inputs, weight):
    return torch_npu.npu_grouped_matmul(
        [inputs],
        [weight.transpose(1, 2)],
        group_list=counts,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]


def grouped_full():
    shrink_0 = gmm(x_w13, a_w13_0)
    delta_0 = gmm(shrink_0, b_w13_0)
    shrink_1 = gmm(x_w13, a_w13_1)
    delta_1 = gmm(shrink_1, b_w13_1)
    shrink_2 = gmm(x_w2, a_w2)
    delta_2 = gmm(shrink_2, b_w2)
    return delta_0, delta_1, delta_2


bgmv_shrink_0 = torch.empty((m, rank), dtype=torch.float32, device="npu")
bgmv_shrink_1 = torch.empty((m, rank), dtype=torch.float32, device="npu")
bgmv_shrink_2 = torch.empty((m, rank), dtype=torch.float32, device="npu")
bgmv_delta_0 = torch.zeros((m, intermediate), dtype=dtype, device="npu")
bgmv_delta_1 = torch.zeros((m, intermediate), dtype=dtype, device="npu")
bgmv_delta_2 = torch.zeros((m, hidden), dtype=dtype, device="npu")


def bgmv_projection(inputs, a_weight, b_weight, shrink, delta, width):
    torch.ops._C_ascend.bgmv_shrink(inputs, a_weight, indices, shrink, 1.0)
    torch.ops._C_ascend.bgmv_expand(shrink, b_weight, indices, delta, 0, width)


def bgmv_full():
    bgmv_projection(x_w13, a_w13_0, b_w13_0, bgmv_shrink_0, bgmv_delta_0, intermediate)
    bgmv_projection(x_w13, a_w13_1, b_w13_1, bgmv_shrink_1, bgmv_delta_1, intermediate)
    bgmv_projection(x_w2, a_w2, b_w2, bgmv_shrink_2, bgmv_delta_2, hidden)
    return bgmv_delta_0, bgmv_delta_1, bgmv_delta_2


grouped_outputs = grouped_full()
bgmv_full()
torch.npu.synchronize()

full_errors = []
for grouped_output, bgmv_output in zip(grouped_outputs, (bgmv_delta_0, bgmv_delta_1, bgmv_delta_2)):
    error = (grouped_output.float() - bgmv_output.float()).abs()
    full_errors.append(
        {
            "max_abs": error.max().cpu().item(),
            "mean_abs": error.mean().cpu().item(),
            "reference_max_abs": bgmv_output.float().abs().max().cpu().item(),
        }
    )

# CPU FP32 golden on a stable row sample. This checks the extra T rounding at
# the A/B boundary without making the Phase0 benchmark CPU-bound.
sample_rows = torch.linspace(0, m - 1, steps=min(m, 16), dtype=torch.int64).tolist()
boundaries = []
end = 0
for group_count in counts_host:
    end += group_count
    boundaries.append(end)


def row_group(row):
    for group, end in enumerate(boundaries):
        if row < end:
            return group
    raise RuntimeError("row is outside group boundaries")


x_w13_cpu = x_w13.cpu().float()
x_w2_cpu = x_w2.cpu().float()
a_weights_cpu = [a_w13_0.cpu().float(), a_w13_1.cpu().float(), a_w2.cpu().float()]
b_weights_cpu = [b_w13_0.cpu().float(), b_w13_1.cpu().float(), b_w2.cpu().float()]
grouped_cpu = [output.cpu().float() for output in grouped_outputs]
cpu_max_abs = [0.0, 0.0, 0.0]
for row in sample_rows:
    group = row_group(row)
    inputs_list = [x_w13_cpu, x_w13_cpu, x_w2_cpu]
    for projection in range(3):
        golden = (
            inputs_list[projection][row]
            @ a_weights_cpu[projection][group].transpose(0, 1)
            @ b_weights_cpu[projection][group].transpose(0, 1)
        )
        difference = (grouped_cpu[projection][row] - golden).abs().max().item()
        cpu_max_abs[projection] = max(cpu_max_abs[projection], difference)

grouped_p50, grouped_max, grouped_values = measure(grouped_full)
bgmv_delta_0.zero_()
bgmv_delta_1.zero_()
bgmv_delta_2.zero_()
bgmv_p50, bgmv_max, bgmv_values = measure(bgmv_full)
print(
    {
        "m": m,
        "hidden": hidden,
        "intermediate": intermediate,
        "rank": rank,
        "groups": groups,
        "distribution": distribution,
        "dtype": str(dtype),
        "split_item": 2,
        "counts": counts_host,
        "max_abs_vs_bgmv": [item["max_abs"] for item in full_errors],
        "mean_abs_vs_bgmv": [item["mean_abs"] for item in full_errors],
        "reference_max_abs": [item["reference_max_abs"] for item in full_errors],
        "sample_cpu_fp32_max_abs": cpu_max_abs,
        "grouped_p50_us": grouped_p50,
        "grouped_max_us": grouped_max,
        "grouped_repeats_us": grouped_values,
        "bgmv_p50_us": bgmv_p50,
        "bgmv_max_us": bgmv_max,
        "bgmv_repeats_us": bgmv_values,
        "speedup": bgmv_p50 / grouped_p50,
    }
)
