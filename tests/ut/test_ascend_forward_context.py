import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture(autouse=True)
def reset_mc2_tokens_capacity(monkeypatch):
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
    monkeypatch.setattr(afc, "_dispatch_v2_tokens_capacity", None)
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
    kv_connector: str | None = None,
    kv_role: str | None = None,
    recompute_scheduler_enable: bool = False,
):
    hf_text_config_attrs: dict[str, object] = {"top_k_experts": top_k_experts}
    if quant_type is not None:
        hf_text_config_attrs["quantize"] = quant_type
    if num_experts_per_tok is not None:
        hf_text_config_attrs["num_experts_per_tok"] = num_experts_per_tok
    hf_text_config_attrs["hidden_size"] = hidden_size

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
    is_moe: bool = True,
):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: is_moe)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: capacity)
    monkeypatch.setattr(afc, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_fused_mc2=enable_fused_mc2),
    )


def test_deepseek_v4_forward_passes_input_ids_to_layers(monkeypatch):
    from vllm.forward_context import ForwardContext, override_forward_context

    from vllm_ascend.models.deepseek_v4 import model as deepseek_v4

    monkeypatch.setattr(afc.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setattr(
        deepseek_v4,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    layer = MagicMock()
    layer.layer_idx = 0
    layer.side_effect = lambda _positions, hidden_states, *_args, **_kwargs: (hidden_states, None)
    model = SimpleNamespace(
        hc_mult=1,
        use_sequence_parallel_moe=False,
        layers=[layer],
        start_layer=0,
        end_layer=1,
        aux_hidden_state_layers=set(),
        _needs_mtp_hidden_states=True,
        _mtp_hidden_buffer=torch.empty(3, 4),
        hc_head=lambda hidden_states, *_: hidden_states.squeeze(1),
        hc_head_fn=None,
        hc_head_scale=None,
        hc_head_base=None,
        norm=lambda hidden_states: hidden_states,
    )
    input_ids = torch.tensor([11, 22, 33])
    forward_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={},
    )

    with override_forward_context(forward_context):
        deepseek_v4.DeepseekV4Model.forward(
            model,
            input_ids,
            positions=torch.arange(input_ids.numel()),
            intermediate_tensors=None,
            inputs_embeds=torch.randn(input_ids.numel(), 4),
        )

    assert layer.call_args.kwargs["input_ids"] is input_ids
    assert "input_ids" not in forward_context.additional_kwargs


def test_set_mc2_tokens_capacity_without_cudagraph_aligns_per_tp_rank():
    vllm_config = _make_vllm_config(tensor_parallel_size=6)

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=200, uniform_decode_query_len=3)

    assert afc.get_mc2_tokens_capacity() == 600


def test_set_mc2_tokens_capacity_with_cudagraph_uses_capture_size_and_aligns(monkeypatch):
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
        kv_role="kv_consumer",
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


def test_is_decode_only_node_false_without_kv_transfer():
    assert afc._is_decode_only_node(_make_vllm_config()) is False


def test_is_decode_only_node_true_for_decode_bench_connector(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    vllm_config = _make_vllm_config(kv_connector="DecodeBenchConnector", kv_role="kv_both")

    assert afc._is_decode_only_node(vllm_config) is True


def test_is_decode_only_node_true_for_kv_consumer(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    vllm_config = _make_vllm_config(kv_role="kv_consumer")

    assert afc._is_decode_only_node(vllm_config) is True


def test_is_decode_only_node_false_without_recompute_scheduler(monkeypatch):
    # With recompute scheduling disabled, prefill runs locally on the
    # decode node, so it is not a decode-only node.
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=False),
        ),
    )
    vllm_config = _make_vllm_config(
        kv_connector="DecodeBenchConnector",
        kv_role="kv_both",
        recompute_scheduler_enable=False,
    )

    assert afc._is_decode_only_node(vllm_config) is False


def test_select_moe_comm_method_returns_none_for_non_moe(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType.A3,
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
        device_type=AscendDeviceType.A3,
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
        device_type=AscendDeviceType.A2,
        capacity=128,
        ep_world_size=16,
    )
    vllm_config = _make_vllm_config(world_size=16, num_experts=128)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


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
        device_type=AscendDeviceType.A3,
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
        device_type=AscendDeviceType.A3,
        capacity=128,
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
        device_type=AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
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
        device_type=AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w4a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("num_tokens", "ep_world_size", "expected"),
    [
        (128, 8, MoECommType.FUSED_MC2),
    ],
)
def test_select_moe_comm_method_a3_quant_w8a8(
    monkeypatch,
    num_tokens,
    ep_world_size,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
    )

    vllm_config = _make_vllm_config(quant_type="w8a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


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
        device_type=AscendDeviceType.A3,
        capacity=128,
        ep_world_size=ep_world_size,
        enable_fused_mc2=1,
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
        device_type=AscendDeviceType.A5,
        capacity=128,
    )
    vllm_config = _make_vllm_config(world_size=world_size, top_k_experts=top_k_experts)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize("num_tokens", [128, 4096])
def test_select_moe_comm_method_a5_uses_megamoe_when_enabled(monkeypatch, num_tokens):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    vllm_config = _make_vllm_config(world_size=8, top_k_experts=8)

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == MoECommType.FUSED_MC2


@pytest.mark.parametrize("num_tokens", [128, 4096])
@pytest.mark.parametrize("draft_quant", [QuantType.NONE, QuantType.W8A8MXFP, QuantType.W4A8MXFP, QuantType.W4A4MXFP])
def test_select_moe_comm_method_a5_preserves_draft_quant_guard(monkeypatch, num_tokens, draft_quant):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    vllm_config = _make_vllm_config(world_size=8, top_k_experts=4)
    expected = MoECommType.FUSED_MC2
    if draft_quant == QuantType.NONE:
        expected = MoECommType.MC2 if num_tokens <= 128 else MoECommType.ALLTOALL
    assert (
        afc.select_moe_comm_method(num_tokens, vllm_config, is_draft_model=True, draft_moe_quant_type=draft_quant)
        == expected
    )


def test_select_moe_comm_method_310p_uses_allgather(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType._310P,
    )

    assert afc.select_moe_comm_method(128, _make_vllm_config()) == MoECommType.ALLGATHER


@pytest.mark.parametrize("model_owned", [False, True])
def test_set_ascend_forward_context_pins_current_vllm_config(monkeypatch, model_owned):
    vllm_config = _make_vllm_config()
    seen: dict[str, object] = {"config": None, "inside": False}

    @contextmanager
    def fake_set_current(config):
        seen["config"] = config
        seen["inside"] = True
        try:
            yield
        finally:
            seen["inside"] = False

    @contextmanager
    def fake_set_forward_context(**_kwargs):
        yield

    forward_context = SimpleNamespace(dp_metadata=None)

    monkeypatch.setattr(afc, "set_current_vllm_config", fake_set_current)
    monkeypatch.setattr(afc, "set_forward_context", fake_set_forward_context)
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(afc, "has_layer_idx", lambda _model: False)
    monkeypatch.setattr(afc, "select_moe_comm_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(afc, "get_mc2_mask", lambda: None)

    legacy_method = object()
    target_method = object()
    draft_method = object()
    moe_mod_name = "vllm_ascend.ops.fused_moe.moe_comm_method"
    if moe_mod_name in sys.modules:
        monkeypatch.setattr(sys.modules[moe_mod_name], "get_moe_comm_method", lambda _t: legacy_method)
    else:
        monkeypatch.setitem(sys.modules, moe_mod_name, SimpleNamespace(get_moe_comm_method=lambda _t: legacy_method))

    for expected in (target_method, draft_method, target_method):
        model = SimpleNamespace(moe_comm_methods={None: expected}) if model_owned else None
        with afc.set_ascend_forward_context(None, vllm_config, num_tokens=4, model_instance=model):
            assert seen["inside"] is True
            assert seen["config"] is vllm_config
            assert forward_context.moe_comm_method is (expected if model_owned else legacy_method)

    assert seen["inside"] is False


def _patch_zercmoe_config_gate(monkeypatch):
    config = _make_vllm_config(tensor_parallel_size=8, max_num_batched_tokens=65536)
    config.model_config.dtype = torch.bfloat16
    config.model_config.is_quantized = False
    config.model_config.hf_text_config.moe_intermediate_size = 1536
    config.parallel_config.data_parallel_size = 1
    config.use_v2_model_runner = False
    config.speculative_config = None
    ascend_config = SimpleNamespace(
        enable_zerc_moe=1,
        enable_fused_mc2=1,
        enable_prefill_mc2=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
    )
    monkeypatch.setattr(afc, "get_ascend_config", lambda: ascend_config)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "enable_sp", lambda _: False)
    monkeypatch.setattr(afc, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A2))
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    return config, ascend_config


def test_zercmoe_config_gate_does_not_access_worker_group_or_wheel(monkeypatch):
    config, _ = _patch_zercmoe_config_gate(monkeypatch)
    monkeypatch.setattr(afc, "get_ep_group", MagicMock(side_effect=AssertionError("worker group accessed")))

    assert afc.is_zercmoe_active_by_config(config)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("ascend", "enable_zerc_moe", 0),
        ("ascend", "enable_fused_mc2", 0),
        ("parallel", "enable_expert_parallel", False),
        ("parallel", "enable_eplb", True),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "data_parallel_size", 2),
        ("parallel", "tensor_parallel_size", 1),
        ("parallel", "tensor_parallel_size", 65),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("model", "dtype", torch.float16),
        ("model", "is_quantized", True),
        ("model", "enable_sleep_mode", True),
        ("root", "weight_transfer_config", object()),
        ("root", "lora_config", object()),
        ("hf", "moe_intermediate_size", 1537),
        ("eplb", "dynamic_eplb", True),
    ],
)
def test_zercmoe_config_gate_rejects_unsupported_configs(monkeypatch, owner, field, value):
    config, ascend_config = _patch_zercmoe_config_gate(monkeypatch)
    objects = {
        "ascend": ascend_config,
        "parallel": config.parallel_config,
        "model": config.model_config,
        "root": config,
        "hf": config.model_config.hf_text_config,
        "eplb": ascend_config.eplb_config,
    }
    setattr(objects[owner], field, value)

    assert not afc.is_zercmoe_active_by_config(config)


@pytest.mark.parametrize("device_type", [AscendDeviceType.A3, AscendDeviceType.A5, AscendDeviceType._310P])
def test_zercmoe_config_gate_requires_a2(monkeypatch, device_type):
    config, _ = _patch_zercmoe_config_gate(monkeypatch)
    monkeypatch.setattr(afc, "get_current_hardware_profile", lambda: get_hardware_profile(device_type))

    assert not afc.is_zercmoe_active_by_config(config)


def test_zercmoe_config_gate_excludes_sequence_parallel(monkeypatch):
    config, _ = _patch_zercmoe_config_gate(monkeypatch)
    monkeypatch.setattr(afc, "enable_sp", lambda _: True)

    assert not afc.is_zercmoe_active_by_config(config)


def test_zercmoe_config_gate_excludes_v2_speculative_decoding(monkeypatch):
    config, _ = _patch_zercmoe_config_gate(monkeypatch)
    config.use_v2_model_runner = True
    config.speculative_config = object()

    assert not afc.is_zercmoe_active_by_config(config)


@pytest.mark.parametrize(
    ("ep_size", "wheel_available", "expected"),
    [(8, True, True), (8, False, False), (1, True, False), (65, True, False)],
)
def test_zercmoe_runtime_gate_checks_wheel_and_ep_group(monkeypatch, ep_size, wheel_available, expected):
    from vllm_ascend.ops.fused_moe import moe_utils

    config, _ = _patch_zercmoe_config_gate(monkeypatch)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))
    monkeypatch.setattr(moe_utils, "zercmoe_lib_available", lambda: wheel_available)

    assert afc.use_cann_zercmoe(config) is expected


@pytest.mark.parametrize(
    ("num_tokens", "is_decode", "with_prefill", "is_draft_model", "available", "expected"),
    [
        (64, False, True, False, True, MoECommType.ALLGATHER),
        (65, False, True, False, True, MoECommType.ZERC_MOE),
        (4096, False, True, False, True, MoECommType.ZERC_MOE),
        (4097, False, True, False, True, MoECommType.ALLGATHER),
        (None, False, True, False, True, MoECommType.ZERC_MOE),
        (128, False, None, False, True, MoECommType.ZERC_MOE),
        (128, False, False, False, True, MoECommType.ALLGATHER),
        (128, True, True, False, True, MoECommType.ALLGATHER),
        (128, False, True, True, True, MoECommType.ALLGATHER),
        (128, False, True, False, False, MoECommType.ALLGATHER),
    ],
)
def test_zercmoe_selection_respects_prefill_graph_draft_and_capacity(
    monkeypatch, num_tokens, is_decode, with_prefill, is_draft_model, available, expected
):
    _patch_select_moe_comm_method_deps(monkeypatch, device_type=AscendDeviceType.A2, capacity=4096)
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: available)

    assert (
        afc.select_moe_comm_method(
            num_tokens,
            _make_vllm_config(),
            is_draft_model=is_draft_model,
            is_decode=is_decode,
            with_prefill=with_prefill,
        )
        == expected
    )


@pytest.mark.parametrize(("capture_size", "expected_dispatch_capacity"), [(257, 264), (8192, 4096)])
def test_zercmoe_capacity_preserves_separate_dispatch_v2_limit(monkeypatch, capture_size, expected_dispatch_capacity):
    config, _ = _patch_zercmoe_config_gate(monkeypatch)
    config.compilation_config.cudagraph_capture_sizes = [capture_size]
    config.compilation_config.max_cudagraph_capture_size = capture_size
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)

    afc.set_mc2_tokens_capacity(config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 4096 * 8
    assert afc.get_dispatch_v2_tokens_capacity() == expected_dispatch_capacity


def test_zercmoe_dispatch_capacity_without_graph_uses_decode_requests(monkeypatch):
    config, _ = _patch_zercmoe_config_gate(monkeypatch)
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)

    afc.set_mc2_tokens_capacity(config, max_num_reqs=200, uniform_decode_query_len=3)

    assert afc.get_mc2_tokens_capacity() == 4096 * 8
    assert afc.get_dispatch_v2_tokens_capacity() == 600


@pytest.mark.parametrize("num_tokens,expected", [(1024, MoECommType.MC2), (1025, MoECommType.ALLGATHER)])
def test_zercmoe_decode_fallback_uses_dispatch_capacity(monkeypatch, num_tokens, expected):
    _patch_select_moe_comm_method_deps(
        monkeypatch, device_type=AscendDeviceType.A2, capacity=65536, ep_world_size=16, enable_fused_mc2=1
    )
    monkeypatch.setattr(afc, "use_cann_zercmoe", lambda _: True)
    monkeypatch.setattr(afc, "get_dispatch_v2_tokens_capacity", lambda: 1024)
    config = _make_vllm_config(world_size=16, tensor_parallel_size=16, num_experts=128)

    assert afc.select_moe_comm_method(num_tokens, config, is_decode=True, with_prefill=False) == expected
