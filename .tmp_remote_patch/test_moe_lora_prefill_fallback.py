import glob

import torch


torch.ops.load_library(
    glob.glob(
        "/home/l00832868/codexWork/vllm-ascend/build_kernel/vllm_ascend_C*.so"
    )[0]
)


def make_wrapper(*, capability=True):
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.lora.punica_npu import PunicaWrapperNPU

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("fallback guard launched a route kernel")

    wrapper = PunicaWrapperNPU.__new__(PunicaWrapperNPU)
    wrapper.is_prefill = True
    wrapper.moe_lora_prefill_route_allgather = unexpected_launch
    wrapper.moe_lora_prefill_route_alltoall = unexpected_launch
    wrapper.moe_lora_prefill_gather_by_perm = unexpected_launch
    wrapper.moe_lora_prefill_scatter_add = unexpected_launch
    wrapper._moe_lora_prefill_workspaces = {}
    wrapper._moe_lora_prefill_weight_views = {}
    wrapper._moe_lora_prefill_capability = capability
    return wrapper


torch.manual_seed(29)
torch.npu.set_device(0)
dtype = torch.bfloat16
rows, hidden, intermediate, rank = 512, 64, 32, 16
adapters, experts = 2, 4


def weight(shape):
    return torch.empty(shape, dtype=dtype, device="npu")


valid = {
    "x": torch.empty((rows, hidden), dtype=dtype, device="npu"),
    "y": torch.empty((rows, 2 * intermediate), dtype=dtype, device="npu"),
    "w13_lora_a": (
        weight((adapters, experts, rank, hidden)),
        weight((adapters, experts, rank, hidden)),
    ),
    "w13_lora_b": (
        weight((adapters, experts, intermediate, rank)),
        weight((adapters, experts, intermediate, rank)),
    ),
    "w2_lora_a": (weight((adapters, experts, rank, intermediate)),),
    "w2_lora_b": (weight((adapters, experts, hidden, rank)),),
    "adapter_enabled": torch.ones(adapters, dtype=torch.bool, device="npu"),
    "route_mode": "alltoall",
    "group_list_type": 1,
    "expert_count": torch.tensor([128, 128, 128, 128], dtype=torch.int64, device="npu"),
    "exchanged_lora_indices": torch.zeros(rows, dtype=torch.int64, device="npu"),
}


def expect_fallback(name, *, capability=True, **updates):
    wrapper = make_wrapper(capability=capability)
    kwargs = dict(valid)
    kwargs.update(updates)
    result = wrapper.prepare_moe_lora_prefill(**kwargs)
    assert result is None, f"{name}: optimized context unexpectedly created"
    assert not wrapper._moe_lora_prefill_workspaces, f"{name}: workspace allocated"
    assert not wrapper._moe_lora_prefill_weight_views, f"{name}: weight view allocated"
    print("FALLBACK PASS", name)


expect_fallback("capability_disabled", capability=False)
# The decode case uses an instance flag rather than a keyword.
decode_wrapper = make_wrapper()
decode_wrapper.is_prefill = False
assert decode_wrapper.prepare_moe_lora_prefill(**valid) is None
assert not decode_wrapper._moe_lora_prefill_workspaces
print("FALLBACK PASS decode_flag")

expect_fallback(
    "M511",
    x=torch.empty((511, hidden), dtype=dtype, device="npu"),
    y=torch.empty((511, 2 * intermediate), dtype=dtype, device="npu"),
    exchanged_lora_indices=torch.zeros(511, dtype=torch.int64, device="npu"),
)
expect_fallback("group_list_type0", group_list_type=0)
expect_fallback("fully_sharded", fully_sharded=True)
expect_fallback("mul_routed_weight", mul_routed_weight=True)
expect_fallback("invalid_route_mode", route_mode="unknown")
expect_fallback(
    "noncontiguous_x",
    x=torch.empty((rows, hidden * 2), dtype=dtype, device="npu")[:, ::2],
)
expect_fallback(
    "offset_x",
    x=torch.empty((rows + 1, hidden), dtype=dtype, device="npu").narrow(0, 1, rows),
)
expect_fallback(
    "mismatched_y_rows",
    y=torch.empty((rows - 1, 2 * intermediate), dtype=dtype, device="npu"),
)
expect_fallback(
    "adapter_enabled_cpu",
    adapter_enabled=torch.ones(adapters, dtype=torch.bool),
)
expect_fallback(
    "rank8",
    w13_lora_a=(
        weight((adapters, experts, 8, hidden)),
        weight((adapters, experts, 8, hidden)),
    ),
    w13_lora_b=(
        weight((adapters, experts, intermediate, 8)),
        weight((adapters, experts, intermediate, 8)),
    ),
)
expect_fallback(
    "G1",
    w13_lora_a=(weight((1, 1, rank, hidden)), weight((1, 1, rank, hidden))),
    w13_lora_b=(
        weight((1, 1, intermediate, rank)),
        weight((1, 1, intermediate, rank)),
    ),
    w2_lora_a=(weight((1, 1, rank, intermediate)),),
    w2_lora_b=(weight((1, 1, hidden, rank)),),
    adapter_enabled=torch.ones(1, dtype=torch.bool, device="npu"),
    expert_count=torch.tensor([rows], dtype=torch.int64, device="npu"),
)
expect_fallback(
    "noncontiguous_count",
    expert_count=torch.tensor(
        [128, 0, 128, 0, 128, 0, 128, 0], dtype=torch.int64, device="npu"
    )[::2],
)
expect_fallback(
    "offset_lora_indices",
    exchanged_lora_indices=torch.zeros(rows + 1, dtype=torch.int64, device="npu").narrow(
        0, 1, rows
    ),
)

print("ALL FALLBACK GUARDS PASS")
