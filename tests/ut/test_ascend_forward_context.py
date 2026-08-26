from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture(autouse=True)
def reset_mc2_tokens_capacity(monkeypatch):
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
    afc._A5_MOE_QUANT_TYPES_BY_CONFIG_ID.clear()
    monkeypatch.setattr(afc, "is_a3_mega_moe_enabled", lambda _: False)
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
    quantization_config: dict[str, object] | None = None,
    quant_description: dict[str, object] | None = None,
    top_k_experts: int = 1,
    num_experts_per_tok: int | None = None,
    cudagraph_capture_sizes: list[int] | None = None,
    max_cudagraph_capture_size: int = 0,
    max_num_batched_tokens: int = 65536,
    resolved_moe_quant_type: QuantType | None = None,
    hidden_act: str | None = None,
):
    hf_text_config_attrs: dict[str, object] = {"top_k_experts": top_k_experts}
    if quant_type is not None:
        hf_text_config_attrs["quantize"] = quant_type
    if quantization_config is not None:
        hf_text_config_attrs["quantization_config"] = quantization_config
    if num_experts_per_tok is not None:
        hf_text_config_attrs["num_experts_per_tok"] = num_experts_per_tok
    if hidden_act is not None:
        hf_text_config_attrs["hidden_act"] = hidden_act

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
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=scheduler_config,
        quant_config=None if quant_description is None else SimpleNamespace(quant_description=quant_description),
    )
    if resolved_moe_quant_type is not None:
        afc.cache_a5_moe_quant_type(vllm_config, resolved_moe_quant_type, "test_layer")
    return vllm_config


def _make_model_instance(quant_type: QuantType, activation=None):
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        quant_type=quant_type,
                        activation=activation,
                    ),
                ),
            ],
        ),
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
    mega_moe_max_tokens: int = 65536,
    dynamic_eplb: bool = False,
    num_redundant_experts: int = 0,
    mix_placement: bool = False,
    a3_mega_moe_enabled: bool = False,
):
    monkeypatch.setattr(afc, "is_moe_model", lambda _: is_moe)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: capacity)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    monkeypatch.setattr(afc, "is_a3_mega_moe_enabled", lambda _: a3_mega_moe_enabled)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=enable_fused_mc2,
            mega_moe_max_tokens=mega_moe_max_tokens,
            mix_placement=mix_placement,
            eplb_config=SimpleNamespace(
                dynamic_eplb=dynamic_eplb,
                num_redundant_experts=num_redundant_experts,
            ),
        ),
    )
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


def test_set_mc2_tokens_capacity_a3_mega_moe_uses_larger_per_rank_limit(monkeypatch):
    monkeypatch.setattr(afc, "is_a3_mega_moe_enabled", lambda _: True)
    vllm_config = _make_vllm_config(
        tensor_parallel_size=1,
        max_num_batched_tokens=8192,
    )
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_prefill_mc2=True, enable_fused_mc2=1),
    )

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=16, uniform_decode_query_len=1)

    assert afc.get_mc2_tokens_capacity() == 4096


def test_get_a5_mega_moe_buffer_capacity_uses_mc2_execution_capacity(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(mega_moe_max_tokens=65536),
    )
    vllm_config = _make_vllm_config(world_size=4, max_num_batched_tokens=120)

    assert afc.get_a5_mega_moe_buffer_tokens_per_rank(vllm_config, 64) == 64


def test_get_a5_mega_moe_buffer_capacity_honors_configured_global_limit(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(mega_moe_max_tokens=256),
    )
    vllm_config = _make_vllm_config(world_size=8)

    assert afc.get_a5_mega_moe_buffer_tokens_per_rank(vllm_config, 128) == 32


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


@pytest.mark.parametrize("num_tokens", [128, 4096])
def test_select_moe_comm_method_a3_uses_mega_moe_up_to_ep64(monkeypatch, num_tokens):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=64,
        enable_fused_mc2=1,
        a3_mega_moe_enabled=True,
    )

    assert afc.select_moe_comm_method(num_tokens, _make_vllm_config(world_size=64)) == MoECommType.FUSED_MC2


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
    ("num_tokens", "world_size", "top_k_experts", "enable_fused_mc2", "quant_type", "expected"),
    [
        (128, 4, 2, 0, None, MoECommType.MC2),
        (129, 2, 4, 0, None, MoECommType.ALLGATHER),
        (129, 8, 4, 0, None, MoECommType.ALLTOALL),
        (128, 8, 4, 1, QuantType.W4A8MXFP, MoECommType.FUSED_MC2),
        (129, 8, 4, 1, QuantType.W4A8MXFP, MoECommType.ALLTOALL),
        (512, 8, 4, 1, QuantType.W4A8MXFP, MoECommType.ALLTOALL),
        (128, 8, 4, 1, QuantType.MXFP8, MoECommType.FUSED_MC2),
        (128, 8, 4, 1, QuantType.MXFP4, MoECommType.FUSED_MC2),
        (129, 8, 4, 2, QuantType.W4A8MXFP, MoECommType.ALLTOALL),
        (129, 8, 4, 1, QuantType.W8A8, MoECommType.ALLTOALL),
        (129, 8, 4, 1, QuantType.W4A16MXFP4, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a5(
    monkeypatch,
    num_tokens,
    world_size,
    top_k_experts,
    enable_fused_mc2,
    quant_type,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=enable_fused_mc2,
    )
    vllm_config = _make_vllm_config(
        world_size=world_size,
        top_k_experts=top_k_experts,
        resolved_moe_quant_type=quant_type,
    )

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


def test_select_moe_comm_method_a5_reads_quant_type_from_model_instance(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
    )

    assert (
        afc.select_moe_comm_method(128, vllm_config, model_instance=_make_model_instance(QuantType.W4A8MXFP))
        == MoECommType.FUSED_MC2
    )
    assert not hasattr(vllm_config, "ascend_moe_quant_type")


def test_select_moe_comm_method_a5_logs_mega_moe_decisions_at_debug(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    matched_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        resolved_moe_quant_type=QuantType.W4A8MXFP,
    )
    fallback_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
    )

    with patch.object(afc.logger, "debug") as mock_debug:
        assert afc.select_moe_comm_method(64, matched_config) == MoECommType.FUSED_MC2
        assert afc.select_moe_comm_method(64, fallback_config) == MoECommType.MC2

    messages = [call.args[0] for call in mock_debug.call_args_list]
    assert sum(message.startswith("A5 MegaMoE condition check") for message in messages) == 2
    assert any(message.startswith("A5 MoE comm selected FUSED_MC2/MegaMoE") for message in messages)
    assert any(message.startswith("A5 MoE comm selected fallback") for message in messages)
    assert any("mc2_tokens_capacity=%s" in message for message in messages)


def test_select_moe_comm_method_a5_uses_mega_moe_for_draft_model(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        resolved_moe_quant_type=QuantType.W4A8MXFP,
    )

    assert afc.select_moe_comm_method(64, vllm_config) == MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(64, vllm_config, is_draft_model=True) == MoECommType.FUSED_MC2


def test_select_moe_comm_method_a5_honors_configured_mega_moe_capacity(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
        mega_moe_max_tokens=512,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        resolved_moe_quant_type=QuantType.W4A8MXFP,
    )

    assert afc.select_moe_comm_method(64, vllm_config) == MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(65, vllm_config) == MoECommType.MC2


def test_select_moe_comm_method_a5_clamps_capacity_to_mc2_limit(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=2048,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(
        world_size=4,
        top_k_experts=4,
        max_num_batched_tokens=8192,
        resolved_moe_quant_type=QuantType.W4A8MXFP,
    )

    assert afc.select_moe_comm_method(2048, vllm_config) == MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(2049, vllm_config) == MoECommType.ALLGATHER


def test_select_moe_comm_method_a5_uses_decode_graph_bucket_instead_of_scheduler_limit(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=64,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(
        world_size=4,
        top_k_experts=6,
        max_num_batched_tokens=120,
        resolved_moe_quant_type=QuantType.W4A8MXFP,
    )

    assert afc.select_moe_comm_method(64, vllm_config) == MoECommType.FUSED_MC2
    assert afc.select_moe_comm_method(65, vllm_config) == MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("dynamic_eplb", "num_redundant_experts"),
    [(True, 0), (False, 1)],
)
def test_select_moe_comm_method_a5_falls_back_for_unsupported_eplb(
    monkeypatch,
    dynamic_eplb,
    num_redundant_experts,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
        dynamic_eplb=dynamic_eplb,
        num_redundant_experts=num_redundant_experts,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        resolved_moe_quant_type=QuantType.MXFP8,
    )

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.MC2


@pytest.mark.parametrize(
    ("num_tokens", "expected_comm_type"),
    [(64, MoECommType.MC2), (129, MoECommType.ALLTOALL)],
)
def test_select_moe_comm_method_a5_falls_back_for_mix_placement(
    monkeypatch,
    num_tokens,
    expected_comm_type,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
        mix_placement=True,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        resolved_moe_quant_type=QuantType.W4A8MXFP,
    )

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected_comm_type


@pytest.mark.parametrize("hidden_act", ["swigluoai", "swiglustep", "situglu"])
def test_select_moe_comm_method_a5_falls_back_for_unsupported_activation(monkeypatch, hidden_act):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        resolved_moe_quant_type=QuantType.MXFP8,
        hidden_act=hidden_act,
    )

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.MC2


def test_select_moe_comm_method_a5_reads_activation_from_model_instance(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(world_size=8, top_k_experts=4)
    model_instance = _make_model_instance(QuantType.W4A8MXFP, activation="swiglustep")

    assert afc.select_moe_comm_method(128, vllm_config, model_instance=model_instance) == MoECommType.MC2


def test_select_moe_comm_method_a5_falls_back_for_unsupported_mxfp_group_size(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        resolved_moe_quant_type=QuantType.MXFP8,
        quant_description={"group_size": 64},
    )

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.MC2


def test_select_moe_comm_method_a5_reads_quant_type_from_vllm_config_cache(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        resolved_moe_quant_type=QuantType.MXFP8,
    )

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.FUSED_MC2


def test_select_moe_comm_method_a5_does_not_parse_config_quant_string(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A5,
        capacity=128,
        enable_fused_mc2=1,
    )
    vllm_config = _make_vllm_config(
        world_size=8,
        top_k_experts=4,
        quant_type="W4A8_MXFP",
        quantization_config={"moe_quant_type": "W8A8_MXFP8"},
        quant_description={"model.layers.0.mlp.experts.w13_weight": "W4A8_MXFP"},
    )

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.MC2


def test_select_moe_comm_method_310p_uses_allgather(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType._310P,
    )

    assert afc.select_moe_comm_method(128, _make_vllm_config()) == MoECommType.ALLGATHER
