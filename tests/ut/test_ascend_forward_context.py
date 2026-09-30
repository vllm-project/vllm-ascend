import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType


@pytest.fixture(autouse=True)
def reset_mc2_tokens_capacity(monkeypatch):
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
    monkeypatch.setattr(afc, "_dispatch_v2_tokens_capacity", None)
    monkeypatch.setattr(afc, "_moe_quant_mismatch", None)
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
    hidden_size: int = 2048,
    routed_expert_hidden_size: int | None = None,
    routed_expert_intermediate_size: int | None = None,
    kv_connector: str | None = None,
    kv_role: str | None = None,
    recompute_scheduler_enable: bool = False,
):
    hf_text_config_attrs: dict[str, object] = {
        "top_k_experts": top_k_experts,
        "moe_intermediate_size": 2048,
    }
    if quant_type is not None:
        hf_text_config_attrs["quantize"] = quant_type
    if num_experts_per_tok is not None:
        hf_text_config_attrs["num_experts_per_tok"] = num_experts_per_tok
    hf_text_config_attrs["hidden_size"] = hidden_size

    if routed_expert_hidden_size is not None:
        hf_text_config_attrs["routed_expert_hidden_size"] = routed_expert_hidden_size
    if routed_expert_intermediate_size is not None:
        hf_text_config_attrs["routed_expert_intermediate_size"] = routed_expert_intermediate_size
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
    kv_transfer_config = (
        SimpleNamespace(kv_connector=kv_connector, kv_role=kv_role)
        if kv_connector is not None or kv_role is not None
        else None
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            recompute_scheduler_enable=recompute_scheduler_enable,
        ),
        kv_transfer_config=kv_transfer_config,
    )


def _patch_select_moe_comm_method_deps(
    monkeypatch,
    *,
    device_type,
    capacity: int = 128,
    ep_world_size: int = 8,
    enable_fused_mc2: int = 0,
    enable_prefill_mc2: int = 0,
    is_moe: bool = True,
):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: is_moe)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: capacity)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=enable_fused_mc2, enable_prefill_mc2=enable_prefill_mc2),
    )


def test_forward_context_exposes_device_metadata_executor(monkeypatch):
    forward_context = SimpleNamespace(dp_metadata=None)
    moe_comm_method = ModuleType("vllm_ascend.ops.fused_moe.moe_comm_method")
    moe_comm_method.__dict__["get_moe_comm_method"] = lambda comm_type: None

    @contextmanager
    def fake_set_forward_context(**kwargs):
        yield

    executor = object()
    monkeypatch.setattr(afc, "set_forward_context", fake_set_forward_context)
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(afc, "select_moe_comm_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda config: False)
    monkeypatch.setattr(afc, "is_moe_model", lambda config: False)
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(afc, "has_layer_idx", lambda model: False)
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(afc, "get_mc2_mask", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.fused_moe.moe_comm_method",
        moe_comm_method,
    )

    with afc.set_ascend_forward_context(
        None,
        _make_vllm_config(),
        num_tokens=1,
        device_metadata_executor=executor,
    ):
        assert forward_context.device_metadata_executor is executor


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


def test_set_mc2_tokens_capacity_decode_only_uses_cudagraph_capture_size(monkeypatch):
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    vllm_config = _make_vllm_config(
        tensor_parallel_size=8,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=257,
        max_num_batched_tokens=8192,
        kv_connector="DecodeBenchConnector",
        kv_role="kv_both",
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 264


def test_set_mc2_tokens_capacity_decode_only_without_cudagraph_uses_decode_shape(monkeypatch):
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    vllm_config = _make_vllm_config(
        tensor_parallel_size=4,
        max_num_batched_tokens=8192,
        kv_connector="DecodeBenchConnector",
        kv_role="kv_both",
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=2)

    assert afc.get_mc2_tokens_capacity() == 32


def test_set_mc2_tokens_capacity_disable_recompute_decode_uses_max_num_batched_tokens(monkeypatch):
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=1,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=False),
        ),
    )
    vllm_config = _make_vllm_config(
        tensor_parallel_size=8,
        cudagraph_capture_sizes=[1, 2],
        max_cudagraph_capture_size=257,
        max_num_batched_tokens=8192,
        kv_connector="DecodeBenchConnector",
        kv_role="kv_both",
        recompute_scheduler_enable=False,
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 8192


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


def test_select_moe_comm_method_a2_uses_allgather_for_more_than_512_experts(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A2,
        capacity=128,
        ep_world_size=64,
    )
    vllm_config = _make_vllm_config(world_size=64, num_experts=896)

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
        (128, 128, MoECommType.MC2),
        (4097, 8, MoECommType.FUSED_MC2),
        (4097, 128, MoECommType.ALLTOALL),
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

    vllm_config = _make_vllm_config(quant_type="w4a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (128, MoECommType.MC2),
        (129, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_without_fused_mc2(
    monkeypatch,
    num_tokens,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        enable_prefill_mc2=1,
    )
    vllm_config = _make_vllm_config()

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w4a16(
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
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a16")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w4a8(
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
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
        (128, 16, MoECommType.FUSED_MC2),
        (128, 32, MoECommType.FUSED_MC2),
        (128, 64, MoECommType.FUSED_MC2),
        (128, 128, MoECommType.FUSED_MC2),
        (128, 129, MoECommType.MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w8a8(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    monkeypatch.setattr(afc, "_CANN_OPS_TRANSFORMER_AVAILABLE", True)
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
        enable_prefill_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w8a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("hidden_size", "routed_expert_hidden_size", "routed_expert_intermediate_size", "expected"),
    [
        (7168, 3584, 3072, True),
        (7168, 768, 3072, False),
        (768, None, None, False),
    ],
)
def test_cann_megamoe_supported_by_config_uses_routed_hidden_size(
    hidden_size,
    routed_expert_hidden_size,
    routed_expert_intermediate_size,
    expected,
):
    vllm_config = _make_vllm_config(
        hidden_size=hidden_size,
        routed_expert_hidden_size=routed_expert_hidden_size,
        routed_expert_intermediate_size=routed_expert_intermediate_size,
    )

    assert afc.is_megamoe_supported_by_config(vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_mc2_invalid_hidden_size(
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
        enable_prefill_mc2=0,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8", hidden_size=512)

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


def test_select_moe_comm_method_310p_uses_allgather(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType._310P,
    )

    assert afc.select_moe_comm_method(128, _make_vllm_config()) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("is_draft_model", "moe_quant_mismatch", "num_tokens", "expected"),
    [
        (True, True, 128, MoECommType.MC2),
        (True, True, 129, MoECommType.ALLTOALL),
        (False, True, 128, MoECommType.FUSED_MC2),
        (True, False, 128, MoECommType.FUSED_MC2),
    ],
)
def test_select_a3_draft_quant_mismatch(
    monkeypatch,
    is_draft_model,
    moe_quant_mismatch,
    num_tokens,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=256,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "get_dispatch_v2_tokens_capacity", lambda: 128)
    monkeypatch.setattr(afc, "_moe_quant_mismatch", moe_quant_mismatch)

    vllm_config = _make_vllm_config()

    assert afc.select_moe_comm_method(num_tokens, vllm_config, is_draft_model=is_draft_model) == expected
