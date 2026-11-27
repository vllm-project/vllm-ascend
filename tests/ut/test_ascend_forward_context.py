from types import SimpleNamespace

import pytest

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType


@pytest.fixture(autouse=True)
def reset_mc2_tokens_capacity(monkeypatch):
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
    monkeypatch.setattr(afc, "_catccos_tokens_capacity", None)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_prefill_mc2=False, enable_fused_mc2=0),
    )


def _make_vllm_config(
    *,
    enable_expert_parallel: bool = True,
    world_size: int = 8,
    pipeline_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    num_experts: int = 128,
    quant_type: str | None = None,
    top_k_experts: int = 1,
    num_experts_per_tok: int | None = None,
    cudagraph_capture_sizes: list[int] | None = None,
    max_cudagraph_capture_size: int = 0,
    max_num_batched_tokens: int = 0,
):
    hf_text_config_attrs: dict[str, object] = {"top_k_experts": top_k_experts}
    if quant_type is not None:
        hf_text_config_attrs["quantize"] = quant_type
    if num_experts_per_tok is not None:
        hf_text_config_attrs["num_experts_per_tok"] = num_experts_per_tok

    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(**hf_text_config_attrs),
        get_num_experts=lambda: num_experts,
    )
    parallel_config = SimpleNamespace(
        enable_expert_parallel=enable_expert_parallel,
        world_size_across_dp=world_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
    )
    compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
        max_cudagraph_capture_size=max_cudagraph_capture_size,
    )
    scheduler_config = SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens)
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=scheduler_config,
    )


def _patch_select_moe_comm_method_deps(
    monkeypatch,
    *,
    device_type,
    capacity: int = 128,
    ep_world_size: int = 8,
    enable_fused_mc2: int = 0,
    is_moe: bool = True,
    spec_decode_enabled: bool = False,
):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: is_moe)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: capacity)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=enable_fused_mc2))
    monkeypatch.setattr(
        afc,
        "speculative_enable_dispatch_gmm_combine_decode",
        lambda _: spec_decode_enabled,
    )


def test_set_mc2_tokens_capacity_without_cudagraph_aligns_per_tp_rank():
    vllm_config = _make_vllm_config(tensor_parallel_size=6)

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=200, uniform_decode_query_len=3)

    assert afc.get_mc2_tokens_capacity() == 600


def test_set_mc2_tokens_capacity_with_cudagraph_uses_capture_size_and_aligns():
    vllm_config = _make_vllm_config(
        tensor_parallel_size=8,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=257,
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 264


def test_set_mc2_tokens_capacity_prefill_mc2_uses_max_num_batched_tokens(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_prefill_mc2=True, enable_fused_mc2=0),
    )
    vllm_config = _make_vllm_config(tensor_parallel_size=8, max_num_batched_tokens=513)

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 520


def test_catccos_capacity_does_not_expand_native_mc2_capacity(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=True,
            enable_fused_mc2=1,
            fused_mc2_backend="catccos",
            catccos_max_tokens_per_rank=4096,
        ),
    )
    config = _make_vllm_config(tensor_parallel_size=4, max_num_batched_tokens=16384)

    afc.set_mc2_tokens_capacity(config, max_num_reqs=8, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 2048
    assert afc.get_catccos_tokens_capacity() == 16384


def test_catccos_prefill_capacity_is_independent_of_native_prefill_mc2(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=1,
            fused_mc2_backend="catccos",
            catccos_max_tokens_per_rank=1024,
        ),
    )
    config = _make_vllm_config(tensor_parallel_size=4, max_num_batched_tokens=4096)

    afc.set_mc2_tokens_capacity(config, max_num_reqs=8, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 8
    assert afc.get_catccos_tokens_capacity() == 4096


def test_select_moe_comm_method_returns_none_for_non_moe(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        is_moe=False,
    )

    assert afc.select_moe_comm_method(16, _make_vllm_config()) is None


@pytest.mark.parametrize(
    ("enable_expert_parallel", "ep_world_size"),
    [
        (False, 8),
        (True, 1),
    ],
)
def test_select_moe_comm_method_uses_allgather_without_effective_expert_parallel(
    monkeypatch,
    enable_expert_parallel,
    ep_world_size,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        ep_world_size=ep_world_size,
    )
    vllm_config = _make_vllm_config(enable_expert_parallel=enable_expert_parallel)

    assert afc.select_moe_comm_method(16, vllm_config) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (128, MoECommType.MC2),
        (129, MoECommType.ALLGATHER),
    ],
)
def test_select_moe_comm_method_a2_uses_mc2_within_capacity(monkeypatch, num_tokens, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A2,
        capacity=128,
        ep_world_size=16,
    )
    vllm_config = _make_vllm_config(world_size=16, num_experts=128)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
        (128, 64, MoECommType.MC2),
        (129, 8, MoECommType.FUSED_MC2),
        (129, 64, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_enable_fused_mc2_mode_1(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
    )

    assert afc.select_moe_comm_method(num_tokens, _make_vllm_config()) == expected


@pytest.mark.parametrize(
    ("num_tokens", "quant_type", "spec_decode_enabled", "expected"),
    [
        (128, "w8a8_dynamic", True, MoECommType.FUSED_MC2),
        (128, "w8a8_dynamic", False, MoECommType.MC2),
        (128, "w4a8", True, MoECommType.MC2),
        (129, "w8a8_dynamic", True, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_enable_fused_mc2_mode_2(
    monkeypatch,
    num_tokens,
    quant_type,
    spec_decode_enabled,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        enable_fused_mc2=2,
        spec_decode_enabled=spec_decode_enabled,
    )
    vllm_config = _make_vllm_config(quant_type=quant_type)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "world_size", "top_k_experts", "expected"),
    [
        (128, 4, 2, MoECommType.MC2),
        (129, 2, 4, MoECommType.ALLGATHER),
        (129, 8, 4, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a5(monkeypatch, num_tokens, world_size, top_k_experts, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
    )
    vllm_config = _make_vllm_config(world_size=world_size, top_k_experts=top_k_experts)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (1, MoECommType.MC2),
        (63, MoECommType.MC2),
        (64, MoECommType.FUSED_MC2),
        (177, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a5_catccos(monkeypatch, num_tokens, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=512,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            fused_mc2_backend="catccos",
            catccos_min_tokens=64,
        ),
    )
    monkeypatch.setattr(afc, "get_catccos_tokens_capacity", lambda: 512)
    vllm_config = _make_vllm_config(world_size=4, top_k_experts=8)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (63, MoECommType.MC2),
        (64, MoECommType.FUSED_MC2),
        (2048, MoECommType.FUSED_MC2),
        (2049, MoECommType.ALLGATHER),
        (4096, MoECommType.ALLGATHER),
        (4097, MoECommType.ALLGATHER),
    ],
)
def test_select_catccos_only_replaces_native_mc2(monkeypatch, num_tokens, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=2048,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=1, fused_mc2_backend="catccos", catccos_min_tokens=64),
    )
    monkeypatch.setattr(afc, "get_catccos_tokens_capacity", lambda: 4096)
    config = _make_vllm_config(world_size=4, tensor_parallel_size=4, top_k_experts=8)

    assert afc.select_moe_comm_method(num_tokens, config) == expected
    if num_tokens == 2048:
        assert afc.select_moe_comm_method(num_tokens, config, is_draft_model=True) == MoECommType.MC2


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (1024, MoECommType.FUSED_MC2),
        (1025, MoECommType.MC2),
        (2049, MoECommType.ALLTOALL),
    ],
)
def test_catccos_smaller_capacity_and_native_alltoall_fallback(monkeypatch, num_tokens, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=2048,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=1, fused_mc2_backend="catccos", catccos_min_tokens=1),
    )
    monkeypatch.setattr(afc, "get_catccos_tokens_capacity", lambda: 1024)
    config = _make_vllm_config(world_size=8, tensor_parallel_size=4, top_k_experts=4)

    assert afc.select_moe_comm_method(num_tokens, config) == expected


def test_catccos_cannot_expand_decode_only_native_mc2_capacity(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=1, fused_mc2_backend="catccos", catccos_min_tokens=1),
    )
    monkeypatch.setattr(afc, "get_catccos_tokens_capacity", lambda: 4096)
    config = _make_vllm_config(world_size=4, tensor_parallel_size=4, top_k_experts=8)

    assert afc.select_moe_comm_method(8, config) == MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(9, config) == MoECommType.ALLGATHER


def test_catccos_minimum_falls_back_without_exceeding_native_mc2_limit(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=2048,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=1, fused_mc2_backend="catccos", catccos_min_tokens=4096),
    )
    monkeypatch.setattr(afc, "get_catccos_tokens_capacity", lambda: 8192)
    config = _make_vllm_config(world_size=4, tensor_parallel_size=4, top_k_experts=8)

    assert afc.select_moe_comm_method(2048, config) == MoECommType.MC2
    assert afc.select_moe_comm_method(2049, config) == MoECommType.ALLGATHER


def test_select_moe_comm_method_310p_uses_allgather(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType._310P,
    )

    assert afc.select_moe_comm_method(128, _make_vllm_config()) == MoECommType.ALLGATHER
