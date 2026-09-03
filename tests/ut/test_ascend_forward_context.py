import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType


@pytest.fixture(autouse=True)
def reset_mc2_tokens_capacity(monkeypatch):
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
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
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: device_type)
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
        layers=[layer],
        start_layer=0,
        end_layer=1,
        aux_hidden_state_layers=set(),
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


def test_set_mc2_tokens_capacity_without_cudagraph_aligns_per_tp_rank(monkeypatch):
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_prefill_mc2=False,
            enable_fused_mc2=0,
            scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
        ),
    )
    vllm_config = _make_vllm_config(tensor_parallel_size=6, kv_role="kv_consumer")

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
        device_type=afc.AscendDeviceType.A3,
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
        device_type=afc.AscendDeviceType.A3,
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


# ---------------------------------------------------------------------------
# mega_moe ratio guard + dispatch_ffn_combine HCCL guard
# ---------------------------------------------------------------------------


def test_hccl_buffsize_would_overflow_false_within_limit(monkeypatch):
    monkeypatch.setenv("HCCL_BUFFSIZE", "200")
    vllm_config = _make_vllm_config(num_experts_per_tok=8, hidden_size=2048)

    assert afc._hccl_buffsize_would_overflow(128, vllm_config) is False


def test_hccl_buffsize_would_overflow_true_when_exceeds(monkeypatch):
    monkeypatch.setenv("HCCL_BUFFSIZE", "1")
    vllm_config = _make_vllm_config(num_experts_per_tok=8, hidden_size=2048)

    assert afc._hccl_buffsize_would_overflow(128, vllm_config) is True


def test_hccl_buffsize_would_overflow_none_tokens(monkeypatch):
    monkeypatch.setenv("HCCL_BUFFSIZE", "1")
    vllm_config = _make_vllm_config()

    assert afc._hccl_buffsize_would_overflow(None, vllm_config) is False


def test_hccl_buffsize_would_overflow_uses_per_tp_rank(monkeypatch):
    # The op sees M=ceil(num_tokens/tp), not the full num_tokens. With tp=8 and
    # num_tokens=128, M=16. HCCL_BUFFSIZE=12MB: the full-num_tokens check (16MB)
    # would overflow, but the per-TP-rank check (~10.75MB) does not -> False.
    monkeypatch.setenv("HCCL_BUFFSIZE", "12")
    vllm_config = _make_vllm_config(num_experts_per_tok=8, hidden_size=2048, tensor_parallel_size=8)

    assert afc._hccl_buffsize_would_overflow(128, vllm_config) is False


@pytest.mark.parametrize(
    ("num_tokens", "mega_moe_max_tokens", "ratio", "expected"),
    [
        # worst_case = 128 * 8 * 8 = 8192 < 65536
        (128, 65536, 1.0, MoECommType.FUSED_MC2),
        # worst_case = 1100 * 8 * 8 = 70400 > 65536 -> all2all
        (1100, 65536, 1.0, MoECommType.ALLTOALL),
        # ratio < 1 keeps mega_moe: 70400 * 0.5 = 35200 < 65536
        (1100, 65536, 0.5, MoECommType.FUSED_MC2),
        # ratio > 1 is stricter: 8192 * 2.0 = 16384 < 65536 still ok
        (128, 65536, 2.0, MoECommType.FUSED_MC2),
    ],
)
def test_select_mega_moe_or_all2all_ratio_guard(monkeypatch, num_tokens, mega_moe_max_tokens, ratio, expected):
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_max_tokens=mega_moe_max_tokens,
            mega_moe_threshold_ratio=ratio,
        ),
    )
    monkeypatch.setattr(afc, "_is_decode_only_node", lambda _vc: False)
    vllm_config = _make_vllm_config(num_experts=128, num_experts_per_tok=8, tensor_parallel_size=1)

    assert afc._select_mega_moe_or_all2all(num_tokens, vllm_config) == expected


def test_select_mega_moe_or_all2all_decode_only_exempt(monkeypatch):
    # Even when worst_case exceeds the buffer, decode-only nodes are exempt.
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_max_tokens=1,
            mega_moe_threshold_ratio=1.0,
        ),
    )
    monkeypatch.setattr(afc, "_is_decode_only_node", lambda _vc: True)
    vllm_config = _make_vllm_config(num_experts=128, num_experts_per_tok=8, tensor_parallel_size=1)

    assert afc._select_mega_moe_or_all2all(1100, vllm_config) == MoECommType.FUSED_MC2


def test_select_a3_mega_moe_falls_back_to_all2all(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _vc: True)
    monkeypatch.setattr(
        afc,
        "get_ascend_config",
        lambda: SimpleNamespace(
            enable_fused_mc2=1,
            mega_moe_max_tokens=65536,
            mega_moe_threshold_ratio=1.0,
        ),
    )
    monkeypatch.setattr(afc, "_is_decode_only_node", lambda _vc: False)
    vllm_config = _make_vllm_config(num_experts=128, num_experts_per_tok=8, tensor_parallel_size=1)

    # worst_case = 1100 * 8 * 8 = 70400 > 65536 -> all2all
    assert afc.select_moe_comm_method(1100, vllm_config) == MoECommType.ALLTOALL


def test_select_a3_dispatch_ffn_combine_hccl_fallback(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _vc: False)
    monkeypatch.setenv("HCCL_BUFFSIZE", "1")
    vllm_config = _make_vllm_config(num_experts=128, num_experts_per_tok=8, tensor_parallel_size=1, hidden_size=2048)

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.ALLTOALL


def test_select_a3_dispatch_ffn_combine_no_hccl_overflow(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=afc.AscendDeviceType.A3,
        capacity=128,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _vc: False)
    monkeypatch.setenv("HCCL_BUFFSIZE", "200")
    vllm_config = _make_vllm_config(num_experts=128, num_experts_per_tok=8, tensor_parallel_size=1, hidden_size=2048)

    assert afc.select_moe_comm_method(128, vllm_config) == MoECommType.FUSED_MC2


def test_set_ascend_forward_context_pins_current_vllm_config(monkeypatch):
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

    moe_mod_name = "vllm_ascend.ops.fused_moe.moe_comm_method"
    if moe_mod_name in sys.modules:
        monkeypatch.setattr(sys.modules[moe_mod_name], "get_moe_comm_method", lambda _t: None)
    else:
        monkeypatch.setitem(sys.modules, moe_mod_name, SimpleNamespace(get_moe_comm_method=lambda _t: None))

    with afc.set_ascend_forward_context(None, vllm_config, num_tokens=4):
        assert seen["inside"] is True
        assert seen["config"] is vllm_config

    assert seen["inside"] is False
