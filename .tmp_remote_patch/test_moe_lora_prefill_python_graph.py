import glob
import sys

import torch
import torch_npu

torch.ops.load_library(
    glob.glob(
        "/home/l00832868/codexWork/vllm-ascend/build_kernel/vllm_ascend_C*.so"
    )[0]
)


def make_wrapper():
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.lora import lora_ops
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    wrapper = PunicaWrapperNPU.__new__(PunicaWrapperNPU)
    wrapper.is_prefill = True
    wrapper.moe_lora_prefill_route_allgather = lora_ops.moe_lora_prefill_route_allgather
    wrapper.moe_lora_prefill_route_alltoall = lora_ops.moe_lora_prefill_route_alltoall
    wrapper.moe_lora_prefill_gather_by_perm = lora_ops.moe_lora_prefill_gather_by_perm
    wrapper.moe_lora_prefill_scatter_add = lora_ops.moe_lora_prefill_scatter_add
    wrapper._moe_lora_prefill_workspaces = {}
    wrapper._moe_lora_prefill_weight_views = {}
    wrapper._moe_lora_prefill_capability = None
    return wrapper


dtype = getattr(torch, sys.argv[1]) if len(sys.argv) > 1 else torch.bfloat16
torch.manual_seed(19)
torch.npu.set_device(0)
rows, hidden, intermediate, rank = 512, 64, 32, 16
adapters, experts = 2, 4


def randn(shape):
    return (torch.randn(shape, dtype=torch.float32, device="npu") / 64).to(dtype)


hidden_states = randn((rows, hidden))
gate_base = randn((rows, 2 * intermediate))
down_base = randn((rows, hidden))
gate_out = torch.empty_like(gate_base)
down_out = torch.empty_like(down_base)
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
uniform = torch.tensor([128, 128, 128, 128], dtype=torch.int64, device="npu")
hotspot = torch.tensor([rows, 0, 0, 0], dtype=torch.int64, device="npu")
counts = uniform.clone()
lora_uniform = torch.arange(rows, dtype=torch.int64, device="npu").remainder(adapters)
lora_fragmented = lora_uniform.clone()
lora_fragmented[::5] = -1
lora_indices = lora_uniform.clone()
wrapper = make_wrapper()


def full_chain():
    gate_out.copy_(gate_base)
    down_out.copy_(down_base)
    context = wrapper.prepare_moe_lora_prefill(
        x=hidden_states,
        y=gate_out,
        w13_lora_a=w13_a,
        w13_lora_b=w13_b,
        w2_lora_a=w2_a,
        w2_lora_b=w2_b,
        adapter_enabled=enabled,
        route_mode="alltoall",
        group_list_type=1,
        expert_count=counts,
        exchanged_lora_indices=lora_indices,
    )
    assert context is not None
    wrapper.apply_moe_lora_prefill(
        context=context,
        y=gate_out,
        x=hidden_states,
        lora_a_stacked=w13_a,
        lora_b_stacked=w13_b,
    )
    silu_out = torch_npu.npu_swiglu(gate_out)
    wrapper.apply_moe_lora_prefill(
        context=context,
        y=down_out,
        x=silu_out,
        lora_a_stacked=w2_a,
        lora_b_stacked=w2_b,
        gather_input=True,
    )
    return gate_out, down_out


# Warmup fills all Python-side caches before capture.
full_chain()
torch.npu.synchronize()
assert wrapper._has_moe_lora_prefill_backend()
workspace = next(iter(wrapper._moe_lora_prefill_workspaces.values()))
workspace_ptrs = {name: tensor.data_ptr() for name, tensor in workspace.items()}

graph = torch.npu.NPUGraph()
with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
    captured_outputs = full_chain()
torch.npu.synchronize()
output_ptrs = tuple(tensor.data_ptr() for tensor in captured_outputs)
count_ptr = counts.data_ptr()
lora_ptr = lora_indices.data_ptr()
allocated_before = torch.npu.memory_allocated(0)
reserved_before = torch.npu.memory_reserved(0)

for replay in range(100):
    if replay % 2:
        counts.copy_(hotspot)
        lora_indices.copy_(lora_fragmented)
    else:
        counts.copy_(uniform)
        lora_indices.copy_(lora_uniform)
    graph.replay()

counts.copy_(uniform)
lora_indices.copy_(lora_uniform)
graph.replay()
torch.npu.synchronize()
captured_cpu = tuple(tensor.cpu() for tensor in captured_outputs)
allocated_after = torch.npu.memory_allocated(0)
reserved_after = torch.npu.memory_reserved(0)

eager_outputs = full_chain()
torch.npu.synchronize()
errors = [
    (captured.float() - eager.cpu().float()).abs().max().item()
    for captured, eager in zip(captured_cpu, eager_outputs)
]
assert errors == [0.0, 0.0]
assert counts.data_ptr() == count_ptr
assert lora_indices.data_ptr() == lora_ptr
assert tuple(tensor.data_ptr() for tensor in captured_outputs) == output_ptrs
assert workspace_ptrs == {name: tensor.data_ptr() for name, tensor in workspace.items()}
assert allocated_after == allocated_before
assert reserved_after == reserved_before
print(
    {
        "dtype": str(dtype),
        "replays": 101,
        "max_abs_vs_eager": errors,
        "input_ptrs_stable": True,
        "output_ptrs_stable": True,
        "workspace_ptrs_stable": True,
        "allocated_growth": allocated_after - allocated_before,
        "reserved_growth": reserved_after - reserved_before,
        "route_error": int(workspace["route_error"][0].cpu()),
    }
)


if __name__ == "__main__":
    pass
