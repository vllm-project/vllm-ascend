import glob

import torch
import torch_npu


def make_wrapper():
    # Match plugin startup ordering; importing Punica first exposes an existing
    # lora.fused_moe <-> ops.fused_moe package-initialization cycle.
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


def projection_t(x, a, b, dtype):
    shrink = (x.float() @ a.float().transpose(0, 1)).to(dtype)
    return (shrink.float() @ b.float().transpose(0, 1)).to(dtype)


def add_reference(y, x, a_stacked, b_stacked, experts, lora, enabled):
    result = y.clone()
    experts_tensor = torch.tensor(experts, dtype=torch.int64)
    offset = 0
    for a, b in zip(a_stacked, b_stacked):
        width = b.shape[-2]
        for adapter in range(enabled.numel()):
            if not bool(enabled[adapter]):
                continue
            for expert in range(a.shape[1]):
                mask = (lora == adapter) & (experts_tensor == expert)
                if not bool(mask.any()):
                    continue
                delta = projection_t(
                    x[mask], a[adapter, expert], b[adapter, expert], x.dtype
                )
                result[mask, offset : offset + width] = (
                    result[mask, offset : offset + width].float() + delta.float()
                ).to(x.dtype)
        offset += width
    return result


def run(dtype):
    torch.manual_seed(11)
    torch.npu.set_device(0)
    rows, hidden, intermediate, rank = 512, 64, 32, 16
    adapters, experts = 2, 4
    counts_cpu = torch.tensor([128, 128, 128, 128], dtype=torch.int64)
    expert_per_row = [row // 128 for row in range(rows)]
    lora_cpu = torch.arange(rows, dtype=torch.int64).remainder(adapters)
    lora_cpu[::7] = -1
    enabled_cpu = torch.tensor([1, 1], dtype=torch.bool)

    def weight(shape):
        return (torch.randn(shape, dtype=torch.float32) / 64).to(dtype).npu()

    w13_a = (
        weight((adapters, experts, rank, hidden)),
        weight((adapters, experts, rank, hidden)),
    )
    w13_b = (
        weight((adapters, experts, intermediate, rank)),
        weight((adapters, experts, intermediate, rank)),
    )
    w2_a = (weight((adapters, experts, rank, intermediate)),)
    w2_b = (weight((adapters, experts, hidden, rank)),)
    hidden_cpu = torch.randn((rows, hidden), dtype=torch.float32).to(dtype)
    gate_base_cpu = torch.randn((rows, 2 * intermediate), dtype=torch.float32).to(dtype)
    down_base_cpu = torch.randn((rows, hidden), dtype=torch.float32).to(dtype)
    hidden_npu = hidden_cpu.npu()
    gate_npu = gate_base_cpu.npu()
    wrapper = make_wrapper()
    assert wrapper._has_moe_lora_prefill_backend(), "exact backend guard unexpectedly disabled"
    context = wrapper.prepare_moe_lora_prefill(
        x=hidden_npu,
        y=gate_npu,
        w13_lora_a=w13_a,
        w13_lora_b=w13_b,
        w2_lora_a=w2_a,
        w2_lora_b=w2_b,
        adapter_enabled=enabled_cpu.npu(),
        route_mode="alltoall",
        group_list_type=1,
        expert_count=counts_cpu.npu(),
        exchanged_lora_indices=lora_cpu.npu(),
    )
    assert context is not None
    workspace = context["workspace"]
    pointers = {name: tensor.data_ptr() for name, tensor in workspace.items()}
    wrapper.apply_moe_lora_prefill(
        context=context,
        y=gate_npu,
        x=hidden_npu,
        lora_a_stacked=w13_a,
        lora_b_stacked=w13_b,
    )
    torch.npu.synchronize()
    gate_ref = add_reference(
        gate_base_cpu,
        hidden_cpu,
        tuple(t.cpu() for t in w13_a),
        tuple(t.cpu() for t in w13_b),
        expert_per_row,
        lora_cpu,
        enabled_cpu,
    )
    torch.testing.assert_close(
        gate_npu.cpu(), gate_ref, rtol=2**-6 if dtype == torch.bfloat16 else 2**-9,
        atol=2**-6 if dtype == torch.bfloat16 else 2**-9,
    )

    silu_npu = torch_npu.npu_swiglu(gate_npu)
    down_npu = down_base_cpu.npu()
    wrapper.apply_moe_lora_prefill(
        context=context,
        y=down_npu,
        x=silu_npu,
        lora_a_stacked=w2_a,
        lora_b_stacked=w2_b,
        gather_input=True,
    )
    torch.npu.synchronize()
    down_ref = add_reference(
        down_base_cpu,
        silu_npu.cpu(),
        tuple(t.cpu() for t in w2_a),
        tuple(t.cpu() for t in w2_b),
        expert_per_row,
        lora_cpu,
        enabled_cpu,
    )
    torch.testing.assert_close(
        down_npu.cpu(), down_ref, rtol=2**-6 if dtype == torch.bfloat16 else 2**-9,
        atol=2**-6 if dtype == torch.bfloat16 else 2**-9,
    )
    context2 = wrapper.prepare_moe_lora_prefill(
        x=hidden_npu,
        y=gate_npu,
        w13_lora_a=w13_a,
        w13_lora_b=w13_b,
        w2_lora_a=w2_a,
        w2_lora_b=w2_b,
        adapter_enabled=enabled_cpu.npu(),
        route_mode="alltoall",
        group_list_type=1,
        expert_count=counts_cpu.npu(),
        exchanged_lora_indices=lora_cpu.npu(),
    )
    assert context2 is not None
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in context2["workspace"].items()
    }
    assert len(wrapper._moe_lora_prefill_workspaces) == 1
    assert len(wrapper._moe_lora_prefill_weight_views) == 3
    print(
        "PYTHON FULL CHAIN PASS",
        dtype,
        "workspace_ptrs_stable=1",
        "route_error=",
        int(workspace["route_error"][0].cpu()),
    )


if __name__ == "__main__":
    so = glob.glob(
        "/home/l00832868/codexWork/vllm-ascend/build_kernel/vllm_ascend_C*.so"
    )[0]
    torch.ops.load_library(so)
    for data_dtype in (torch.float16, torch.bfloat16):
        run(data_dtype)
