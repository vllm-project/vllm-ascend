import argparse
import glob
import json
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu


torch.ops.load_library(
    glob.glob(
        "/home/l00832868/codexWork/vllm-ascend/build_kernel/vllm_ascend_C*.so"
    )[0]
)
import vllm_ascend.ops  # noqa: E402,F401
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU  # noqa: E402
from vllm_ascend.lora.fused_moe import (  # noqa: E402
    _prepare_moe_lora_routing_allgather_indices,
)


def percentile(samples, fraction):
    samples = sorted(samples)
    return samples[min(len(samples) - 1, int((len(samples) - 1) * fraction))]


def make_route(rows, adapters, experts, distribution):
    if distribution == "hotspot":
        counts = [rows] + [0] * (experts - 1)
        lora = torch.zeros(rows, dtype=torch.int64, device="npu")
    else:
        if distribution == "longtail":
            raw = [2 ** (experts - index - 1) for index in range(experts)]
            total = sum(raw)
            counts = [rows * value // total for value in raw]
            counts[0] += rows - sum(counts)
        else:
            counts = [rows // experts] * experts
            for index in range(rows % experts):
                counts[index] += 1
        lora_cpu = torch.empty(rows, dtype=torch.int64)
        begin = 0
        for count in counts:
            if distribution == "longtail":
                values = torch.zeros(count, dtype=torch.int64)
                if adapters > 1 and count:
                    tail = max(1, count // 5)
                    values[-tail:] = torch.arange(tail).remainder(adapters)
            elif distribution == "one_row_per_group":
                values = torch.zeros(count, dtype=torch.int64)
                values[: min(count, adapters)] = torch.arange(min(count, adapters))
            else:
                values = torch.arange(count, dtype=torch.int64).remainder(adapters)
            lora_cpu[begin : begin + count] = values
            begin += count
        if distribution == "inactive25":
            lora_cpu[::4] = -1
        elif distribution == "inactive100":
            lora_cpu.fill_(-1)
        lora = lora_cpu.npu()
    return torch.tensor(counts, dtype=torch.int64, device="npu"), lora


def measure(function, *, warmup, samples, repeats):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            function()
        torch.npu.synchronize()
        values.append((time.perf_counter_ns() - start) / repeats / 1000.0)
    return {
        "p50_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
        "min_us": min(values),
    }


def run_case(args, rows, adapters, distribution):
    dtype = getattr(torch, args.dtype)
    experts, hidden, intermediate, rank = args.experts, args.hidden, args.intermediate, 16
    torch.manual_seed(43 + rows + adapters)

    def randn(shape):
        return (torch.randn(shape, dtype=torch.float32, device="npu") / 64).to(dtype)

    x = randn((rows, hidden))
    gate_base = randn((rows, 2 * intermediate))
    down_base = randn((rows, hidden))
    gate_new = torch.empty_like(gate_base)
    down_new = torch.empty_like(down_base)
    gate_old = torch.empty_like(gate_base)
    down_old = torch.empty_like(down_base)
    w13_a = (
        randn((adapters, experts, rank, hidden)),
        randn((adapters, experts, rank, hidden)),
    )
    w13_b = (
        randn((adapters, experts, intermediate, rank)),
        randn((adapters, experts, intermediate, rank)),
    )
    w2_a = (randn((adapters, experts, rank, intermediate)),)
    w2_b = (randn((adapters, experts, hidden, rank)),)
    enabled = torch.ones(adapters, dtype=torch.bool, device="npu")
    counts, lora_indices = make_route(rows, adapters, experts, distribution)
    expert_range = torch.arange(experts, dtype=torch.int64, device="npu")
    sorted_expert_ids = torch.repeat_interleave(expert_range, counts)
    expanded_row_idx = None
    routed_topk_ids = None
    token_lora_indices = None
    if args.route_mode == "allgather":
        canonical_order = torch.randperm(rows, dtype=torch.int64).npu()
        expanded_row_idx = canonical_order
        routed_topk_ids = sorted_expert_ids[canonical_order]
        token_lora_indices = lora_indices[canonical_order]

    new_wrapper = PunicaWrapperNPU(rows, 1, "npu:0", max_loras=adapters)
    old_wrapper = PunicaWrapperNPU(rows, 1, "npu:0", max_loras=adapters)
    new_wrapper.is_prefill = True
    old_wrapper.is_prefill = True
    assert new_wrapper._has_moe_lora_prefill_backend()
    if args.route_mode == "allgather":
        old_wrapper.indices_len = [rows, None, None, None]
        old_wrapper._token_lora_indices = token_lora_indices
        old_lora_context = SimpleNamespace(
            top_k=1,
            punica_wrapper=old_wrapper,
            adapter_enabled=enabled,
            w13_lora_a_stacked=w13_a,
        )

    def new_chain():
        gate_new.copy_(gate_base)
        down_new.copy_(down_base)
        context = new_wrapper.prepare_moe_lora_prefill(
            x=x,
            y=gate_new,
            w13_lora_a=w13_a,
            w13_lora_b=w13_b,
            w2_lora_a=w2_a,
            w2_lora_b=w2_b,
            adapter_enabled=enabled,
            route_mode=args.route_mode,
            group_list_type=1,
            expanded_row_idx=expanded_row_idx,
            routed_topk_ids=routed_topk_ids,
            token_lora_indices=token_lora_indices,
            top_k=1,
            expert_count=counts,
            exchanged_lora_indices=lora_indices,
        )
        assert context is not None
        new_wrapper.apply_moe_lora_prefill(
            context=context,
            y=gate_new,
            x=x,
            lora_a_stacked=w13_a,
            lora_b_stacked=w13_b,
        )
        activated = torch_npu.npu_swiglu(gate_new)
        new_wrapper.apply_moe_lora_prefill(
            context=context,
            y=down_new,
            x=activated,
            lora_a_stacked=w2_a,
            lora_b_stacked=w2_b,
            gather_input=True,
        )

    def old_chain():
        gate_old.copy_(gate_base)
        down_old.copy_(down_base)
        if args.route_mode == "allgather":
            routing = _prepare_moe_lora_routing_allgather_indices(
                old_lora_context, expanded_row_idx, routed_topk_ids
            )
            expert_ids, routed_lora_indices, combined = routing
        else:
            expert_ids = torch.repeat_interleave(expert_range, counts)
            routed_lora_indices = lora_indices
            lora_safe = lora_indices.clamp(min=0)
            active = (lora_indices >= 0) & enabled[lora_safe].bool()
            combined = torch.where(
                active,
                lora_safe * experts + expert_ids,
                torch.full_like(lora_indices, -1),
            ).contiguous()
        old_wrapper.add_lora_fused_moe(
            y=gate_old,
            x=x,
            lora_a_stacked=w13_a,
            lora_b_stacked=w13_b,
            expert_ids=expert_ids,
            adapter_enabled=enabled,
            token_lora_mapping=routed_lora_indices,
            combined_indices=combined,
        )
        activated = torch_npu.npu_swiglu(gate_old)
        old_wrapper.add_lora_fused_moe(
            y=down_old,
            x=activated,
            lora_a_stacked=w2_a,
            lora_b_stacked=w2_b,
            expert_ids=expert_ids,
            adapter_enabled=enabled,
            token_lora_mapping=routed_lora_indices,
            combined_indices=combined,
        )

    # Compile/warm both paths, then check the benchmark is comparing the same result.
    new_chain()
    old_chain()
    torch.npu.synchronize()
    tolerance = 2**-6 if dtype == torch.bfloat16 else 2**-9
    torch.testing.assert_close(down_new.cpu(), down_old.cpu(), rtol=tolerance, atol=tolerance)
    new_result = measure(
        new_chain, warmup=args.warmup, samples=args.samples, repeats=args.repeats
    )
    old_result = measure(
        old_chain, warmup=args.warmup, samples=args.samples, repeats=args.repeats
    )
    result = {
        "dtype": args.dtype,
        "M": rows,
        "L": adapters,
        "E": experts,
        "G": adapters * experts,
        "distribution": distribution,
        "route_mode": args.route_mode,
        "new": new_result,
        "old": old_result,
        "speedup_p50": old_result["p50_us"] / new_result["p50_us"],
        "speedup_p95": old_result["p95_us"] / new_result["p95_us"],
        "explicit_workspace_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in next(iter(new_wrapper._moe_lora_prefill_workspaces.values())).values()
        ),
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--rows", default="512,768,1024,2048,4096")
    parser.add_argument("--adapters", default="1,2,4")
    parser.add_argument(
        "--distributions",
        default="uniform,hotspot,longtail,one_row_per_group,inactive25,inactive100",
    )
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument(
        "--route-mode", choices=("alltoall", "allgather"), default="alltoall"
    )
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    torch.npu.set_device(0)
    for rows in map(int, args.rows.split(",")):
        for adapters in map(int, args.adapters.split(",")):
            for distribution in args.distributions.split(","):
                run_case(args, rows, adapters, distribution)


if __name__ == "__main__":
    main()
