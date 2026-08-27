import glob

import torch
import torch_npu


def reference(x, expanded, topk, token_lora, enabled, top_k, experts, first_expert=0):
    groups = enabled.numel() * experts
    buckets = [[] for _ in range(groups)]
    for r in range(expanded.numel()):
        dst = int(expanded[r])
        if dst < 0:
            continue
        lora = int(token_lora[r // top_k])
        expert = int(topk[r]) - first_expert
        active = (
            0 <= dst < x.shape[0]
            and 0 <= lora < enabled.numel()
            and 0 <= expert < experts
            and bool(enabled[lora])
        )
        group = lora * experts + expert if active else 0
        encoded = dst if active else -(dst + 1)
        buckets[group].append((dst if 0 <= dst < x.shape[0] else 0, encoded))
    count = torch.tensor([len(bucket) for bucket in buckets], dtype=torch.int64)
    rows = [item for bucket in buckets for item in bucket]
    grouped = torch.stack([x[row] for row, _ in rows])
    perm = torch.zeros((x.shape[0], 8), dtype=torch.int32)
    perm[:, 0] = torch.tensor([encoded for _, encoded in rows], dtype=torch.int32)
    return count, grouped, perm


def run(dtype, index_dtype, enabled_dtype):
    torch.npu.set_device(0)
    m, width, top_k, experts, adapters = 16, 32, 2, 4, 2
    groups = experts * adapters
    gp = (groups + 7) // 8 * 8
    c = torch.npu.get_device_properties(0).vector_core_num
    x_cpu = torch.arange(m * width, dtype=torch.float32).reshape(m, width) / 17
    expanded_cpu = torch.tensor(
        [5, 0, 12, 3, 8, 1, 15, 7, 9, 2, 14, 6, 13, 4, 11, 10],
        dtype=index_dtype,
    )
    topk_cpu = torch.tensor(
        [0, 1, 2, 3, 1, 0, 3, 2, 0, 2, 1, 3, 2, 1, 0, 3],
        dtype=index_dtype,
    )
    token_lora_cpu = torch.tensor([0, 1, -1, 0, 1, 0, 1, 0], dtype=torch.int64)
    enabled_cpu = torch.tensor([1, 1], dtype=enabled_dtype)

    x = x_cpu.to(dtype).npu()
    expanded = expanded_cpu.npu()
    topk = topk_cpu.npu()
    token_lora = token_lora_cpu.npu()
    enabled = enabled_cpu.npu()
    local_count = torch.empty((c, gp), dtype=torch.int32, device="npu")
    core_prefix = torch.empty_like(local_count)
    group_total = torch.empty(groups, dtype=torch.int32, device="npu")
    group_start = torch.empty(groups, dtype=torch.int32, device="npu")
    group_count = torch.empty(groups, dtype=torch.int64, device="npu")
    perm = torch.empty((m, 8), dtype=torch.int32, device="npu")
    error_per_core = torch.empty((c, 8), dtype=torch.int32, device="npu")
    route_error = torch.empty(8, dtype=torch.int32, device="npu")
    grouped = torch.empty((m, width), dtype=dtype, device="npu")

    torch.ops._C_ascend.moe_lora_prefill_route_allgather(
        x,
        expanded,
        topk,
        token_lora,
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
    ref_count, ref_grouped, ref_perm = reference(
        x_cpu.to(dtype), expanded_cpu, topk_cpu, token_lora_cpu, enabled_cpu, top_k, experts
    )
    torch.testing.assert_close(group_count.cpu(), ref_count)
    torch.testing.assert_close(grouped.cpu(), ref_grouped, rtol=0, atol=0)
    torch.testing.assert_close(perm.cpu(), ref_perm)
    assert int(route_error[0].cpu()) == 0
    assert int(group_count.sum().cpu()) == m
    print(
        "PASS",
        dtype,
        index_dtype,
        enabled_dtype,
        "count=", group_count.cpu().tolist(),
    )


def run_boundary(groups, all_inactive=False):
    torch.npu.set_device(0)
    torch.manual_seed(31 + groups)
    m, width, top_k, experts, adapters = 512, 32, 2, groups, 1
    gp = (groups + 7) // 8 * 8
    c = torch.npu.get_device_properties(0).vector_core_num
    first_expert = 100
    x_cpu = torch.randn(m, width, dtype=torch.float16)
    permutation = torch.randperm(m, dtype=torch.int64).to(torch.int32)
    expanded_cpu = torch.full((2 * m,), -1, dtype=torch.int32)
    expanded_cpu[0::2] = permutation
    topk_cpu = torch.zeros(2 * m, dtype=torch.int32)
    topk_cpu[0::2] = first_expert + torch.arange(m, dtype=torch.int32).remainder(experts)
    token_lora_cpu = torch.zeros(m, dtype=torch.int64)
    enabled_cpu = torch.tensor([not all_inactive], dtype=torch.bool)

    x = x_cpu.npu()
    expanded = expanded_cpu.npu()
    topk = topk_cpu.npu()
    token_lora = token_lora_cpu.npu()
    enabled = enabled_cpu.npu()
    local_count = torch.empty((c, gp), dtype=torch.int32, device="npu")
    core_prefix = torch.empty((c, gp), dtype=torch.int32, device="npu")
    group_total = torch.empty(groups, dtype=torch.int32, device="npu")
    group_start = torch.empty(groups, dtype=torch.int32, device="npu")
    group_count = torch.empty(groups, dtype=torch.int64, device="npu")
    perm = torch.empty((m, 8), dtype=torch.int32, device="npu")
    error_per_core = torch.empty((c, 8), dtype=torch.int32, device="npu")
    route_error = torch.empty(8, dtype=torch.int32, device="npu")
    grouped = torch.empty((m, width), dtype=torch.float16, device="npu")
    torch.ops._C_ascend.moe_lora_prefill_route_allgather(
        x, expanded, topk, token_lora, enabled,
        local_count, core_prefix, group_total, group_start, group_count,
        perm, error_per_core, route_error, grouped,
        top_k, experts, first_expert,
    )
    torch.npu.synchronize()
    ref_count, ref_grouped, ref_perm = reference(
        x_cpu, expanded_cpu, topk_cpu, token_lora_cpu, enabled_cpu,
        top_k, experts, first_expert,
    )
    torch.testing.assert_close(group_count.cpu(), ref_count)
    torch.testing.assert_close(grouped.cpu(), ref_grouped, rtol=0, atol=0)
    torch.testing.assert_close(perm.cpu(), ref_perm)
    assert int(route_error[0].cpu()) == 0
    assert int(group_count.sum().cpu()) == m
    print("BOUNDARY PASS", "G=", groups, "Gp=", gp, "inactive=", all_inactive)


if __name__ == "__main__":
    so = glob.glob("/home/l00832868/codexWork/vllm-ascend/build_kernel/vllm_ascend_C*.so")[0]
    torch.ops.load_library(so)
    for dtype in (torch.float16, torch.bfloat16):
        for index_dtype in (torch.int32, torch.int64):
            for enabled_dtype in (torch.bool, torch.int32, torch.int64):
                run(dtype, index_dtype, enabled_dtype)
    for groups in (2, 7, 8, 9, 255, 256, 257, 258, 1023, 1024):
        run_boundary(groups)
    run_boundary(8, all_inactive=True)
