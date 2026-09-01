import glob

import torch
import torch_npu


def safe_experts(counts, rows):
    endpoints = []
    endpoint = 0
    error = 0
    for raw in counts.tolist():
        count = int(raw)
        if count < 0:
            count = 0
            error |= 1
        remaining = rows - endpoint
        if count > remaining:
            endpoint = rows
            error |= 1
        else:
            endpoint += count
        endpoints.append(endpoint)
    if endpoint != rows:
        error |= 2
    experts = []
    for row in range(rows):
        expert = -1
        for candidate, end in enumerate(endpoints):
            if row < end:
                expert = candidate
                break
        experts.append(expert)
    return experts, error


def reference(x, counts, lora, enabled):
    rows = x.shape[0]
    num_experts = counts.numel()
    num_groups = enabled.numel() * num_experts
    row_experts, error = safe_experts(counts, rows)
    buckets = [[] for _ in range(num_groups)]
    for row, expert in enumerate(row_experts):
        adapter = int(lora[row])
        active = (
            expert >= 0
            and 0 <= adapter < enabled.numel()
            and bool(enabled[adapter])
        )
        group = adapter * num_experts + expert if active else 0
        encoded = row if active else -(row + 1)
        buckets[group].append((row, encoded))
    count = torch.tensor([len(bucket) for bucket in buckets], dtype=torch.int64)
    ordered = [item for bucket in buckets for item in bucket]
    grouped = torch.stack([x[row] for row, _ in ordered])
    perm = torch.zeros((rows, 8), dtype=torch.int32)
    perm[:, 0] = torch.tensor([encoded for _, encoded in ordered], dtype=torch.int32)
    return count, grouped, perm, error


def allocate(rows, width, groups, dtype):
    pitch = (groups + 7) // 8 * 8
    cores = torch.npu.get_device_properties(0).vector_core_num
    return (
        torch.empty((cores, pitch), dtype=torch.int32, device="npu"),
        torch.empty((cores, pitch), dtype=torch.int32, device="npu"),
        torch.empty(groups, dtype=torch.int32, device="npu"),
        torch.empty(groups, dtype=torch.int32, device="npu"),
        torch.empty(groups, dtype=torch.int64, device="npu"),
        torch.empty((rows, 8), dtype=torch.int32, device="npu"),
        torch.empty((cores, 8), dtype=torch.int32, device="npu"),
        torch.empty(8, dtype=torch.int32, device="npu"),
        torch.empty((rows, width), dtype=dtype, device="npu"),
    )


def invoke(x, counts, lora, enabled):
    work = allocate(x.shape[0], x.shape[1], counts.numel() * enabled.numel(), x.dtype)
    torch.ops._C_ascend.moe_lora_prefill_route_alltoall(
        x, counts, lora, enabled, *work
    )
    torch.npu.synchronize()
    return work


def run_valid(dtype, count_dtype, enabled_dtype):
    rows, width = 16, 33
    x_cpu = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) / 13
    counts_cpu = torch.tensor([4, 0, 5, 7], dtype=count_dtype)
    lora_cpu = torch.tensor(
        [0, 1, -1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, -1, 0, 1],
        dtype=torch.int64,
    )
    enabled_cpu = torch.tensor([1, 1], dtype=enabled_dtype)
    x = x_cpu.to(dtype).npu()
    work = invoke(x, counts_cpu.npu(), lora_cpu.npu(), enabled_cpu.npu())
    _, _, _, _, group_count, perm, _, route_error, grouped = work
    ref_count, ref_grouped, ref_perm, ref_error = reference(
        x_cpu.to(dtype), counts_cpu, lora_cpu, enabled_cpu
    )
    torch.testing.assert_close(group_count.cpu(), ref_count)
    torch.testing.assert_close(grouped.cpu(), ref_grouped, rtol=0, atol=0)
    torch.testing.assert_close(perm.cpu(), ref_perm)
    assert int(route_error[0].cpu()) == ref_error == 0
    assert int(group_count.sum().cpu()) == rows
    print("VALID PASS", dtype, count_dtype, enabled_dtype, ref_count.tolist())


def run_repair(name, raw_counts, expected_error):
    rows, width = 16, 33
    counts_cpu = torch.tensor(raw_counts, dtype=torch.int64)
    lora_cpu = torch.tensor(
        [0, 1, -1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, -1, 0, 1],
        dtype=torch.int64,
    )
    enabled_cpu = torch.tensor([1, 1], dtype=torch.bool)
    x_cpu = torch.arange(rows * width, dtype=torch.float16).reshape(rows, width)
    work = invoke(x_cpu.npu(), counts_cpu.npu(), lora_cpu.npu(), enabled_cpu.npu())
    _, _, _, _, group_count, perm, _, route_error, grouped = work
    ref_count, ref_grouped, ref_perm, ref_error = reference(
        x_cpu, counts_cpu, lora_cpu, enabled_cpu
    )
    torch.testing.assert_close(group_count.cpu(), ref_count)
    torch.testing.assert_close(grouped.cpu(), ref_grouped, rtol=0, atol=0)
    torch.testing.assert_close(perm.cpu(), ref_perm)
    assert ref_error == expected_error
    assert int(route_error[0].cpu()) == expected_error
    assert int(group_count.sum().cpu()) == rows
    print("REPAIR PASS", name, "error=", expected_error, "count=", ref_count.tolist())


if __name__ == "__main__":
    torch.npu.set_device(0)
    so = glob.glob(
        "/home/l00832868/codexWork/vllm-ascend/build_kernel/vllm_ascend_C*.so"
    )[0]
    torch.ops.load_library(so)
    for data_dtype in (torch.float16, torch.bfloat16):
        for count_dtype in (torch.int32, torch.int64):
            for enabled_dtype in (torch.bool, torch.int32, torch.int64):
                run_valid(data_dtype, count_dtype, enabled_dtype)
    run_repair("negative", [4, -3, 5, 7], 1)
    run_repair("overflow", [4, 20, 5, 7], 1)
    run_repair("shortfall", [4, 0, 5, 2], 2)
