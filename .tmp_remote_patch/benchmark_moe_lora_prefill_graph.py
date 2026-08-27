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
from vllm_ascend.lora.fused_moe import (  # noqa: E402
    _prepare_moe_lora_routing_allgather_indices,
)
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU  # noqa: E402


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def measure_graph(graph, samples=100):
    for _ in range(10):
        graph.replay()
    torch.npu.synchronize()
    values = []
    for _ in range(samples):
        begin = time.perf_counter_ns()
        graph.replay()
        torch.npu.synchronize()
        values.append((time.perf_counter_ns() - begin) / 1000.0)
    return {
        "p50_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
        "min_us": min(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--route-mode", choices=("alltoall", "allgather"), required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    dtype = getattr(torch, args.dtype)
    rows = 1024 if args.route_mode == "alltoall" else 2048
    hidden, intermediate, rank, adapters, experts = 4096, 2048, 16, 1, 8
    torch.manual_seed(53)

    def randn(shape):
        return (torch.randn(shape, dtype=torch.float32, device="npu") / 64).to(dtype)

    x = randn((rows, hidden))
    gate_base = randn((rows, 2 * intermediate))
    down_base = randn((rows, hidden))
    gate_new, down_new = torch.empty_like(gate_base), torch.empty_like(down_base)
    gate_old, down_old = torch.empty_like(gate_base), torch.empty_like(down_base)
    w13_a = (randn((adapters, experts, rank, hidden)),) * 2
    w13_b = (randn((adapters, experts, intermediate, rank)),) * 2
    w2_a = (randn((adapters, experts, rank, intermediate)),)
    w2_b = (randn((adapters, experts, hidden, rank)),)
    counts = torch.full((experts,), rows // experts, dtype=torch.int64, device="npu")
    enabled = torch.ones(adapters, dtype=torch.bool, device="npu")
    lora_sorted = torch.zeros(rows, dtype=torch.int64, device="npu")
    expert_range = torch.arange(experts, dtype=torch.int64, device="npu")
    sorted_experts = torch.repeat_interleave(expert_range, counts)
    expanded_row_idx = routed_topk_ids = token_lora_indices = None
    if args.route_mode == "allgather":
        canonical_order = torch.randperm(rows, dtype=torch.int64).npu()
        expanded_row_idx = canonical_order
        routed_topk_ids = sorted_experts[canonical_order]
        token_lora_indices = lora_sorted[canonical_order]

    new_wrapper = PunicaWrapperNPU(rows, 1, "npu:0", max_loras=adapters)
    old_wrapper = PunicaWrapperNPU(rows, 1, "npu:0", max_loras=adapters)
    new_wrapper.is_prefill = old_wrapper.is_prefill = True
    if args.route_mode == "allgather":
        old_wrapper.indices_len = [rows, None, None, None]
        old_wrapper._token_lora_indices = token_lora_indices
        old_context = SimpleNamespace(
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
            exchanged_lora_indices=lora_sorted,
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
        return gate_new, down_new

    def old_chain():
        gate_old.copy_(gate_base)
        down_old.copy_(down_base)
        if args.route_mode == "allgather":
            expert_ids, routed_lora, combined = _prepare_moe_lora_routing_allgather_indices(
                old_context, expanded_row_idx, routed_topk_ids
            )
        else:
            expert_ids = torch.repeat_interleave(expert_range, counts)
            routed_lora = lora_sorted
            combined = lora_sorted * experts + expert_ids
        old_wrapper.add_lora_fused_moe(
            y=gate_old,
            x=x,
            lora_a_stacked=w13_a,
            lora_b_stacked=w13_b,
            expert_ids=expert_ids,
            adapter_enabled=enabled,
            token_lora_mapping=routed_lora,
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
            token_lora_mapping=routed_lora,
            combined_indices=combined,
        )
        return gate_old, down_old

    new_chain()
    torch.npu.synchronize()
    eager_down = down_new.cpu()
    new_graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        new_graph, capture_error_mode="thread_local", auto_dispatch_capture=True
    ):
        new_outputs = new_chain()
    torch.npu.synchronize()
    ptrs = tuple(t.data_ptr() for t in new_outputs)
    allocated_before = torch.npu.memory_allocated(0)
    reserved_before = torch.npu.memory_reserved(0)
    new_perf = measure_graph(new_graph)
    allocated_after = torch.npu.memory_allocated(0)
    reserved_after = torch.npu.memory_reserved(0)
    torch.testing.assert_close(new_outputs[1].cpu(), eager_down, rtol=0, atol=0)
    assert ptrs == tuple(t.data_ptr() for t in new_outputs)
    assert allocated_before == allocated_after and reserved_before == reserved_after

    old_result = {"captured": False}
    try:
        old_chain()
        torch.npu.synchronize()
        old_graph = torch.npu.NPUGraph()
        with torch.npu.graph(
            old_graph, capture_error_mode="thread_local", auto_dispatch_capture=True
        ):
            old_chain()
        torch.npu.synchronize()
        old_result = {"captured": True, **measure_graph(old_graph)}
    except Exception as error:
        old_result = {
            "captured": False,
            "error_type": type(error).__name__,
            "error": str(error)[:300],
        }
    print(
        json.dumps(
            {
                "dtype": args.dtype,
                "route_mode": args.route_mode,
                "M": rows,
                "new": new_perf,
                "old": old_result,
                "allocated_growth": allocated_after - allocated_before,
                "reserved_growth": reserved_after - reserved_before,
                "output_ptrs_stable": True,
            }
        )
    )


if __name__ == "__main__":
    main()
