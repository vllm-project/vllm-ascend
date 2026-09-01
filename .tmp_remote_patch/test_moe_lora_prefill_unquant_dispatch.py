import glob
import os
import sys
import types
from types import SimpleNamespace

# This test lives beside an unrelated ``triton/`` test package. Remove that
# directory before importing torch so it cannot shadow the installed Triton.
_THIS_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path[:] = [path for path in sys.path if os.path.realpath(path or os.getcwd()) != _THIS_DIR]

import torch
import torch_npu


torch.ops.load_library(
    glob.glob(
        "/home/l00832868/codexWork/vllm-ascend/build_kernel/vllm_ascend_C*.so"
    )[0]
)
import vllm_ascend.ops  # noqa: E402,F401
from vllm_ascend.lora import lora_ops  # noqa: E402
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU  # noqa: E402
from vllm_ascend.ops.fused_moe.moe_mlp import unquant_apply_mlp  # noqa: E402


def make_wrapper():
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


def run(dtype):
    torch.manual_seed(37)
    rows, hidden, intermediate, rank = 1024, 4096, 2048, 16
    adapters, experts = 2, 8

    def randn(shape):
        return (torch.randn(shape, dtype=torch.float32, device="npu") / 64).to(dtype)

    hidden_states = randn((rows, hidden))
    w1 = randn((experts, 2 * intermediate, hidden))
    w2 = randn((experts, hidden, intermediate))
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
    group_list = torch.full(
        (experts,), rows // experts, dtype=torch.int64, device="npu"
    )
    lora_indices = torch.arange(rows, dtype=torch.int64, device="npu").remainder(adapters)
    lora_indices[::7] = -1
    enabled = torch.ones(adapters, dtype=torch.bool, device="npu")
    wrapper = make_wrapper()
    wrapper.indices_len = [rows, None, None, None]
    wrapper._token_lora_indices = lora_indices
    calls = []
    original_prepare = wrapper.prepare_moe_lora_prefill
    original_apply = wrapper.apply_moe_lora_prefill

    def traced_prepare(self, **kwargs):
        calls.append(("prepare", kwargs["route_mode"]))
        return original_prepare(**kwargs)

    def traced_apply(self, **kwargs):
        calls.append(("apply", kwargs.get("gather_input", False)))
        return original_apply(**kwargs)

    wrapper.prepare_moe_lora_prefill = types.MethodType(traced_prepare, wrapper)
    wrapper.apply_moe_lora_prefill = types.MethodType(traced_apply, wrapper)
    lora_context = SimpleNamespace(
        punica_wrapper=wrapper,
        w13_lora_a_stacked=w13_a,
        w13_lora_b_stacked=w13_b,
        w2_lora_a_stacked=w2_a,
        w2_lora_b_stacked=w2_b,
        adapter_enabled=enabled,
        top_k=1,
        fully_sharded=False,
        exchanged_lora_indices=lora_indices,
        local_num_experts=experts,
        tp_rank=0,
    )
    actual, _ = unquant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        group_list=group_list,
        group_list_type=1,
        need_trans=True,
        lora_context=lora_context,
    )

    # Reproduce the same full path directly. This isolates the real
    # unquant_apply_mlp integration from the already validated kernels.
    manual_wrapper = make_wrapper()
    gate = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1.transpose(1, 2)],
        bias=None,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]
    context = manual_wrapper.prepare_moe_lora_prefill(
        x=hidden_states,
        y=gate,
        w13_lora_a=w13_a,
        w13_lora_b=w13_b,
        w2_lora_a=w2_a,
        w2_lora_b=w2_b,
        adapter_enabled=enabled,
        route_mode="alltoall",
        group_list_type=1,
        expert_count=group_list,
        exchanged_lora_indices=lora_indices,
    )
    assert context is not None
    manual_wrapper.apply_moe_lora_prefill(
        context=context,
        y=gate,
        x=hidden_states,
        lora_a_stacked=w13_a,
        lora_b_stacked=w13_b,
    )
    activated = torch_npu.npu_swiglu(gate)
    expected = torch_npu.npu_grouped_matmul(
        x=[activated],
        weight=[w2.transpose(1, 2)],
        bias=None,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]
    manual_wrapper.apply_moe_lora_prefill(
        context=context,
        y=expected,
        x=activated,
        lora_a_stacked=w2_a,
        lora_b_stacked=w2_b,
        gather_input=True,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert calls == [("prepare", "alltoall"), ("apply", False), ("apply", True)]
    assert len(wrapper._moe_lora_prefill_workspaces) == 1
    assert len(wrapper._moe_lora_prefill_weight_views) == 3
    assert not hasattr(lora_context, "exchanged_lora_indices")
    print("UNQUANT OPTIMIZED DISPATCH PASS", dtype, calls)

    # M<512 must keep the original BGMV path. Use a no-op fallback spy to
    # validate scheduling without invoking unrelated legacy kernels here.
    fallback_wrapper = make_wrapper()
    fallback_wrapper.indices_len = [511, None, None, None]
    fallback_wrapper._token_lora_indices = lora_indices[:511].clone()
    fallback_calls = []

    def fallback_spy(self, **kwargs):
        fallback_calls.append(kwargs["x"].shape[-1])

    fallback_wrapper.add_lora_fused_moe = types.MethodType(fallback_spy, fallback_wrapper)
    fallback_context = SimpleNamespace(
        punica_wrapper=fallback_wrapper,
        w13_lora_a_stacked=w13_a,
        w13_lora_b_stacked=w13_b,
        w2_lora_a_stacked=w2_a,
        w2_lora_b_stacked=w2_b,
        adapter_enabled=enabled,
        top_k=1,
        fully_sharded=False,
        exchanged_lora_indices=lora_indices[:511].clone(),
        local_num_experts=experts,
        tp_rank=0,
    )
    fallback_counts = torch.tensor(
        [64, 64, 64, 64, 64, 64, 64, 63], dtype=torch.int64, device="npu"
    )
    unquant_apply_mlp(
        hidden_states=hidden_states[:511].clone(),
        w1=w1,
        w2=w2,
        group_list=fallback_counts,
        group_list_type=1,
        need_trans=True,
        lora_context=fallback_context,
    )
    torch.npu.synchronize()
    assert fallback_calls == [hidden, intermediate]
    assert not fallback_wrapper._moe_lora_prefill_workspaces
    print("UNQUANT BGMV FALLBACK DISPATCH PASS", dtype, fallback_calls)

    no_lora_wrapper = make_wrapper()
    no_lora_wrapper.no_lora = True

    def unexpected_lora_launch(*args, **kwargs):
        raise AssertionError("Host-static no_lora batch launched a LoRA op")

    no_lora_wrapper.prepare_moe_lora_prefill = unexpected_lora_launch
    no_lora_wrapper.add_lora_fused_moe = unexpected_lora_launch
    no_lora_context = SimpleNamespace(
        punica_wrapper=no_lora_wrapper,
        w13_lora_a_stacked=w13_a,
        w13_lora_b_stacked=w13_b,
        w2_lora_a_stacked=w2_a,
        w2_lora_b_stacked=w2_b,
        adapter_enabled=enabled,
        top_k=1,
        fully_sharded=False,
        exchanged_lora_indices=torch.full(
            (rows,), -1, dtype=torch.int64, device="npu"
        ),
        local_num_experts=experts,
        tp_rank=0,
    )
    unquant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        group_list=group_list,
        group_list_type=1,
        need_trans=True,
        lora_context=no_lora_context,
    )
    torch.npu.synchronize()
    assert not no_lora_wrapper._moe_lora_prefill_workspaces
    print("UNQUANT HOST-STATIC NO_LORA PASS", dtype)


def test_moe_lora_prefill_unquant_dispatch():
    torch.npu.set_device(0)
    for test_dtype in (torch.float16, torch.bfloat16):
        run(test_dtype)


if __name__ == "__main__":
    test_moe_lora_prefill_unquant_dispatch()
