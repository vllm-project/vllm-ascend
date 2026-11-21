import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import vllm_version_is


@pytest.fixture(autouse=True)
def reset_mc2_tokens_capacity(monkeypatch):
    monkeypatch.setattr(afc, "_mc2_tokens_capacity", None)
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", False)
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: False)
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
        enforce_eager=True,
        hf_text_config=SimpleNamespace(**hf_text_config_attrs),
        get_num_experts=lambda: num_experts,
    )
    parallel_config = SimpleNamespace(
        enable_expert_parallel=enable_expert_parallel,
        world_size_across_dp=world_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.NONE,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
        max_cudagraph_capture_size=max_cudagraph_capture_size,
    )
    if not vllm_version_is("0.33.0"):
        compilation_config.decode_cudagraph_capture_request_sizes = None
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

    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", True)
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


@pytest.mark.parametrize(
    "prefill_tokens,decode_sizes,query_lens,token_cap,tp_size,expected",
    [
        (16, [1, 2, 3, 4], {4: 1024}, None, 2, 16),
        (16, [1, 2, 3, 4], {4: 256}, None, 2, 16),
        (8, [1, 4, 16], {4: 16}, None, 2, 64),
        (8, [1, 4, 16], {4: 16}, 20, 2, 20),
        (8, [1, 4, 16], {1: 16, 4: 3}, None, 2, 16),
        (17, [1, 2], {4: 16}, None, 6, 18),
        (4096, [1, 2], {4: 16}, None, 2, 1024),
    ],
)
def test_mc2_capacity_excludes_independent_draft_capture_grid(
    monkeypatch, prefill_tokens, decode_sizes, query_lens, token_cap, tp_size, expected
):
    if vllm_version_is("0.33.0"):
        pytest.skip("Separate target/draft graph grids are a main-only API")

    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    vllm_config = _make_vllm_config(
        tensor_parallel_size=tp_size,
        cudagraph_capture_sizes=[4, 8, 12, 16, 1024],
        max_cudagraph_capture_size=1024,
        max_num_batched_tokens=8192,
    )
    compilation_config = vllm_config.compilation_config
    compilation_config.decode_cudagraph_capture_request_sizes = decode_sizes
    compilation_config.max_decode_cudagraph_capture_tokens = token_cap
    compilation_config.max_prefill_cudagraph_capture_tokens = prefill_tokens
    vllm_config.get_decode_cudagraph_query_lens = lambda _: query_lens
    original_sizes = compilation_config.cudagraph_capture_sizes.copy()

    afc.set_mc2_tokens_capacity(vllm_config, max_num_reqs=1024, uniform_decode_query_len=4)

    assert afc.get_mc2_tokens_capacity() == expected
    assert compilation_config.cudagraph_capture_sizes == original_sizes
    assert compilation_config.max_cudagraph_capture_size == 1024
    assert compilation_config.decode_cudagraph_capture_request_sizes == decode_sizes
    if expected == 16:
        assert afc._select_fused_or_capacity_moe_comm_method(16, vllm_config, expected) == MoECommType.MC2
        assert afc._select_fused_or_capacity_moe_comm_method(1024, vllm_config, expected) == MoECommType.ALLTOALL


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
    monkeypatch.setattr(afc, "use_v2_model_runner", lambda _cfg: True)

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
            assert afc._USE_V2_EXTRA_KWARGS is True

    assert seen["inside"] is False


def test_sync_v2_extra_kwargs_caches_actual_bool(monkeypatch):
    monkeypatch.setattr(afc, "use_v2_model_runner", lambda _cfg: True)
    afc.sync_v2_extra_kwargs(SimpleNamespace())
    assert afc._USE_V2_EXTRA_KWARGS is True

    monkeypatch.setattr(afc, "use_v2_model_runner", lambda _cfg: False)
    afc.sync_v2_extra_kwargs(SimpleNamespace())
    assert afc._USE_V2_EXTRA_KWARGS is False

    monkeypatch.setattr(afc, "use_v2_model_runner", lambda _cfg: MagicMock())
    afc.sync_v2_extra_kwargs(SimpleNamespace())
    assert afc._USE_V2_EXTRA_KWARGS is False


def test_extra_ctx_getattr_does_not_call_use_v2_model_runner(monkeypatch):
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", True)

    def _boom(_cfg=None):
        raise AssertionError("compiled extra-ctx must not call use_v2_model_runner")

    monkeypatch.setattr(afc, "use_v2_model_runner", _boom)
    forward_context = SimpleNamespace(additional_kwargs={}, capturing=True)
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)

    assert afc._EXTRA_CTX.capturing is None
    afc._EXTRA_CTX.capturing = False
    assert forward_context.additional_kwargs["capturing"] is False


def test_extra_ctx_default_v2_hides_gpu_capturing_flag(monkeypatch):
    # GPU V2 ForwardContext has no vllm_config. Isolation follows the
    # eager-cached _USE_V2_EXTRA_KWARGS flag, not ctx.vllm_config.
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", True)
    forward_context = SimpleNamespace(
        additional_kwargs={},
        capturing=True,
    )
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)

    assert afc._EXTRA_CTX.capturing is None
    afc._EXTRA_CTX.capturing = False
    assert afc._EXTRA_CTX.capturing is False
    assert forward_context.capturing is True
    assert forward_context.additional_kwargs["capturing"] is False


def test_extra_ctx_v1_stores_capturing_on_context(monkeypatch):
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", False)
    forward_context = SimpleNamespace(
        additional_kwargs={},
        capturing=False,
    )
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)

    afc._EXTRA_CTX.capturing = True
    assert afc._EXTRA_CTX.capturing is True
    assert forward_context.capturing is True
    assert "capturing" not in forward_context.additional_kwargs


def test_extra_ctx_v1_override_uses_context_capturing_flag(monkeypatch):
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", False)
    forward_context = SimpleNamespace(
        additional_kwargs={},
        capturing=False,
    )
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)

    afc._EXTRA_CTX.capturing = True
    assert forward_context.capturing is True
    assert "capturing" not in forward_context.additional_kwargs


def test_extra_ctx_magicmock_forward_context_stays_on_v1_attrs(monkeypatch):
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", False)
    forward_context = MagicMock(capturing=False)
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)

    assert afc._EXTRA_CTX.capturing is False
    afc._EXTRA_CTX.capturing = True
    assert forward_context.capturing is True


def test_extra_ctx_unset_vllm_config_stays_on_v1_attrs(monkeypatch):
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", False)
    forward_context = MagicMock(capturing=False)
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)

    assert afc._EXTRA_CTX.capturing is False
    afc._EXTRA_CTX.capturing = True
    assert forward_context.capturing is True


def test_extra_ctx_env_true_uses_additional_kwargs(monkeypatch):
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", True)
    forward_context = SimpleNamespace(
        additional_kwargs={},
        capturing=True,
    )
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)

    assert afc._EXTRA_CTX.capturing is None
    afc._EXTRA_CTX.capturing = False
    assert afc._EXTRA_CTX.capturing is False
    assert forward_context.capturing is True
    assert forward_context.additional_kwargs["capturing"] is False


@pytest.mark.parametrize(
    "tp,pcp,tokens,requests,query,method,expected",
    [
        (1, 8, 16384, 500, 2, "mtp", 3923),
        (8, 1, 16384, 500, 2, "mtp", 16384),
        (3, 8, 513, 16, 1, "mtp", 111),
        (1, 16, 16384, 512, 4, "mtp", 3968),
        (1, 8, 16384, 500, 2, "dflash", 16384),
    ],
)
@pytest.mark.parametrize("prefill_mc2", [True, False])
def test_prefill_mc2_capacity_reuses_profile_budget(
    monkeypatch, tp, pcp, tokens, requests, query, method, expected, prefill_mc2
):
    monkeypatch.setattr(
        afc, "get_ascend_config", lambda: SimpleNamespace(enable_prefill_mc2=prefill_mc2, enable_fused_mc2=1)
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: True)
    config = _make_vllm_config(tensor_parallel_size=tp, max_num_batched_tokens=tokens)
    config.parallel_config.prefill_context_parallel_size = pcp
    config.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=query - 1)
    afc.set_mc2_tokens_capacity(config, requests, query)
    assert afc.get_mc2_tokens_capacity() == expected


@pytest.mark.parametrize("pcp", [1, 8])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("mega_moe", [False, True])
def test_decode_mc2_capacity_keeps_local_graph_and_request_budget(monkeypatch, pcp, graph, mega_moe):
    monkeypatch.setattr(
        afc, "get_ascend_config", lambda: SimpleNamespace(enable_prefill_mc2=False, enable_fused_mc2=int(mega_moe))
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: mega_moe)
    monkeypatch.setattr(afc, "_is_decode_only_node", lambda _: True)
    config = _make_vllm_config(
        tensor_parallel_size=8,
        max_num_batched_tokens=16384,
        cudagraph_capture_sizes=[257] if graph else [],
        max_cudagraph_capture_size=257 if graph else 0,
    )
    config.parallel_config.prefill_context_parallel_size = pcp
    afc.set_mc2_tokens_capacity(config, 200, 3)
    assert afc.get_mc2_tokens_capacity() == (264 if graph else 600)


@pytest.mark.parametrize("fused,mega_moe,expected", [(0, False, 1024), (1, False, 1024), (1, True, 32768)])
def test_prefill_mc2_capacity_keeps_per_rank_operator_limit(monkeypatch, fused, mega_moe, expected):
    monkeypatch.setattr(
        afc, "get_ascend_config", lambda: SimpleNamespace(enable_prefill_mc2=True, enable_fused_mc2=fused)
    )
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: mega_moe)
    config = _make_vllm_config(tensor_parallel_size=2, max_num_batched_tokens=524288)
    config.parallel_config.prefill_context_parallel_size = 8
    config.speculative_config = None
    afc.set_mc2_tokens_capacity(config, 500, 1)
    assert afc.get_mc2_tokens_capacity() == expected


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (None, MoECommType.FUSED_MC2),
        (192, MoECommType.FUSED_MC2),
        (384, MoECommType.FUSED_MC2),
        (385, MoECommType.ALLTOALL),
        (512, MoECommType.ALLTOALL),
        (4096, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_megamoe_respects_mc2_tokens_capacity(
    monkeypatch,
    num_tokens,
    expected,
):
    # AscendConfig normalizes enable_fused_mc2=2/3 to 1 alongside
    # is_mega_moe_supported()=True; verify oversized steps do not fall through
    # into the legacy enable_fused_mc2 == 1 branch.
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType.A3,
        capacity=384,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    vllm_config = _make_vllm_config(world_size=8, num_experts=384, quant_type="w4a8")

    assert afc.select_moe_comm_method(num_tokens, vllm_config) == expected


@pytest.mark.parametrize(
    ("is_draft_model", "draft_moe_quant_type", "num_tokens", "expected"),
    [
        (False, QuantType.NONE, 48, MoECommType.FUSED_MC2),
        (True, QuantType.NONE, 48, MoECommType.MC2),
        (True, QuantType.NONE, 512, MoECommType.ALLTOALL),
        (True, QuantType.W8A8, 48, MoECommType.FUSED_MC2),
        (True, QuantType.W8A8, 512, MoECommType.ALLTOALL),
    ],
)
def test_select_moe_comm_method_a3_megamoe_isolates_unquantized_draft_model(
    monkeypatch,
    is_draft_model,
    draft_moe_quant_type,
    num_tokens,
    expected,
):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType.A3,
        capacity=384,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    vllm_config = _make_vllm_config(world_size=8, num_experts=384, quant_type="w4a8")

    assert (
        afc.select_moe_comm_method(
            num_tokens,
            vllm_config,
            is_draft_model=is_draft_model,
            draft_moe_quant_type=draft_moe_quant_type,
        )
        == expected
    )


def test_sync_draft_moe_comm_context_restores_target_mrv2_state(monkeypatch):
    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType.A3,
        capacity=384,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", True)
    vllm_config = _make_vllm_config(world_size=8, num_experts=384, quant_type="w4a8")
    vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(draft_moe_quant_type=QuantType.NONE),
        target_model_config=vllm_config.model_config,
    )

    target_comm = object()
    draft_comm = object()
    moe_mod_name = "vllm_ascend.ops.fused_moe.moe_comm_method"
    comm_table = {MoECommType.FUSED_MC2: target_comm, MoECommType.MC2: draft_comm}
    if moe_mod_name in sys.modules:
        monkeypatch.setattr(sys.modules[moe_mod_name], "get_moe_comm_method", lambda t: comm_table[t])
    else:
        monkeypatch.setitem(sys.modules, moe_mod_name, SimpleNamespace(get_moe_comm_method=lambda t: comm_table[t]))

    forward_context = SimpleNamespace(
        vllm_config=vllm_config,
        additional_kwargs={
            "is_draft_model": False,
            "is_draft_model_prefill": False,
            "draft_moe_quant_type": QuantType.NONE,
            "moe_comm_type": MoECommType.FUSED_MC2,
            "moe_comm_method": target_comm,
            "use_mega_moe": True,
            "max_tokens_across_dp": 24,
            "num_tokens": 24,
        },
    )
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)

    with afc.sync_draft_moe_comm_context(vllm_config, is_draft_model_prefill=True):
        assert afc._EXTRA_CTX.is_draft_model is True
        assert afc._EXTRA_CTX.is_draft_model_prefill is True
        assert afc._EXTRA_CTX.moe_comm_type is MoECommType.MC2
        assert afc._EXTRA_CTX.moe_comm_method is draft_comm
        assert afc._EXTRA_CTX.use_mega_moe is False

    assert afc._EXTRA_CTX.is_draft_model is False
    assert afc._EXTRA_CTX.is_draft_model_prefill is False
    assert afc._EXTRA_CTX.moe_comm_type is MoECommType.FUSED_MC2
    assert afc._EXTRA_CTX.moe_comm_method is target_comm
    assert afc._EXTRA_CTX.use_mega_moe is True


def test_bind_draft_moe_comm_context_eager_exception_and_dense_draft(monkeypatch):
    import torch.nn as nn

    from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext, model_capture_wrapper
    from vllm_ascend.worker.v2.spec_decode import _wrap_speculator_draft_model_context

    _patch_select_moe_comm_method_deps(
        monkeypatch,
        device_type=AscendDeviceType.A3,
        capacity=384,
        ep_world_size=8,
        enable_fused_mc2=1,
    )
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "_USE_V2_EXTRA_KWARGS", True)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False, raising=False)

    target_comm = object()
    draft_comm = object()
    moe_mod_name = "vllm_ascend.ops.fused_moe.moe_comm_method"
    comm_table = {MoECommType.FUSED_MC2: target_comm, MoECommType.MC2: draft_comm, None: None}
    if moe_mod_name in sys.modules:
        monkeypatch.setattr(sys.modules[moe_mod_name], "get_moe_comm_method", lambda t: comm_table[t])
    else:
        monkeypatch.setitem(sys.modules, moe_mod_name, SimpleNamespace(get_moe_comm_method=lambda t: comm_table[t]))

    draft_vllm_config = _make_vllm_config(world_size=8, num_experts=384, quant_type="w4a8")
    draft_vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(draft_moe_quant_type=QuantType.NONE),
        target_model_config=SimpleNamespace(),
    )

    observed: list[tuple[object, ...]] = []

    class DummyDraftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_comm_methods = {MoECommType.MC2: draft_comm}
            self.should_fail = False

        def forward(self, x):
            try:
                observed.append(
                    (
                        afc._EXTRA_CTX.is_draft_model,
                        afc._EXTRA_CTX.is_draft_model_prefill,
                        afc._EXTRA_CTX.moe_comm_type,
                        afc._EXTRA_CTX.moe_comm_method,
                        afc._EXTRA_CTX.use_mega_moe,
                    )
                )
            except AssertionError:
                observed.append(("no_ctx",))
            if self.should_fail:
                raise RuntimeError("boom")
            return x + 1

    class DummySpeculator:
        def __init__(self):
            self.draft_vllm_config = draft_vllm_config
            self.model = None

        def load_model(self):
            self.model = DummyDraftModel()

    speculator = _wrap_speculator_draft_model_context(DummySpeculator(), draft_vllm_config)
    speculator.load_model()
    assert isinstance(speculator.model, DummyDraftModel)

    # Outside forward_context (get_forward_context raises AssertionError): safe no-op.
    def _raise_no_ctx():
        raise AssertionError("Forward context is not set")

    monkeypatch.setattr(afc, "get_forward_context", _raise_no_ctx)
    assert torch.equal(speculator.model(torch.zeros(2)), torch.ones(2))
    observed.clear()

    # ForwardContext intentionally has NO .vllm_config attribute (upstream vLLM ForwardContext).
    from vllm.config import CUDAGraphMode

    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        additional_kwargs={
            "is_draft_model": False,
            "is_draft_model_prefill": False,
            "draft_moe_quant_type": QuantType.NONE,
            "moe_comm_type": MoECommType.FUSED_MC2,
            "moe_comm_method": target_comm,
            "use_mega_moe": True,
            "max_tokens_across_dp": 24,
            "num_tokens": 24,
        },
    )
    monkeypatch.setattr(afc, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr("vllm_ascend.worker.v2.aclgraph_utils.get_forward_context", lambda: forward_context)

    # 1. Eager / PIECEWISE direct speculator.model(...) call isolates draft comm & restores target.
    out = speculator.model(torch.zeros(2))
    assert torch.equal(out, torch.ones(2))
    assert observed[-1] == (True, False, MoECommType.MC2, draft_comm, False)
    assert afc._EXTRA_CTX.moe_comm_type is MoECommType.FUSED_MC2
    assert afc._EXTRA_CTX.use_mega_moe is True

    # 2. Exception inside draft forward still restores target state cleanly.
    speculator.model.should_fail = True
    with pytest.raises(RuntimeError, match="boom"):
        speculator.model(torch.zeros(2))
    speculator.model.should_fail = False
    assert afc._EXTRA_CTX.is_draft_model is False
    assert afc._EXTRA_CTX.moe_comm_type is MoECommType.FUSED_MC2
    assert afc._EXTRA_CTX.moe_comm_method is target_comm
    assert afc._EXTRA_CTX.use_mega_moe is True

    # 3. Capture wrapper (ModelWithContext) preserves is_draft_model_prefill=True without forward_context.vllm_config.
    with model_capture_wrapper(speculator, is_draft_model_prefill=True):
        assert isinstance(speculator.model, ModelWithContext)
        speculator.model(torch.zeros(2))
    assert observed[-1] == (True, True, MoECommType.MC2, draft_comm, False)
    assert afc._EXTRA_CTX.is_draft_model is False
    assert afc._EXTRA_CTX.is_draft_model_prefill is False
    assert afc._EXTRA_CTX.moe_comm_type is MoECommType.FUSED_MC2

    # 4. Dense draft model (is_moe_model=False -> moe_comm_type=None) with model_comm_methods={} never raises KeyError.
    dense_draft_cfg = _make_vllm_config(world_size=8, num_experts=0, quant_type=None)
    monkeypatch.setattr(afc, "is_moe_model", lambda cfg: cfg is not dense_draft_cfg)
    speculator.model.moe_comm_methods = {}
    afc.bind_draft_moe_comm_context(speculator.model, dense_draft_cfg)
    speculator.model(torch.zeros(2))
    assert observed[-1] == (True, False, None, None, False)
    assert afc._EXTRA_CTX.moe_comm_type is MoECommType.FUSED_MC2


def test_megamoe_routed_scaling_factor_never_double_scales_router_scaled_weights():
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl, TokenDispatcherWithMC2

    captured: dict[str, torch.Tensor] = {}

    def fake_mega_moe(_x, _topk_ids, topk_weights, *_args, **_kwargs):
        captured["topk_weights"] = topk_weights.clone()
        return torch.zeros(2, 8), torch.zeros(2, dtype=torch.int32)

    comm_impl = object.__new__(FusedMC2CommImpl)
    comm_impl.mega_moe = fake_mega_moe
    comm_impl._mega_moe_supported_kwargs = frozenset(
        {
            "shared_l1_weights",
            "shared_l2_weights",
            "shared_l1_weights_sf",
            "shared_l2_weights_sf",
            "shared_expert_quant_out_dtype",
            "shared_weight1_type",
            "shared_weight2_type",
        }
    )
    comm_impl.mega_moe_activation_kwargs = {}
    comm_impl.mega_moe_symm_buffer = MagicMock()
    comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm_impl.token_dispatcher.max_num_tokens_per_rank = 64
    comm_impl.token_dispatcher.global_bs = 0

    weights = MoEWeights(
        w1=[torch.zeros(8, 16, dtype=torch.int8)],
        w2=[torch.zeros(8, 8, dtype=torch.int8)],
        w1_scale=[torch.ones(16)],
        w2_scale=[torch.ones(8)],
        shared_w1=[torch.zeros(8, 16, dtype=torch.int8)],
        shared_w2=[torch.zeros(8, 8, dtype=torch.int8)],
        shared_w1_scale=[torch.ones(16)],
        shared_w2_scale=[torch.ones(8)],
    )

    for fold_scale in (1.0, None):
        inp = MagicMock()
        inp.hidden_states = torch.zeros(2, 8, dtype=torch.bfloat16)
        inp.topk_ids = torch.zeros(2, 2, dtype=torch.int32)
        inp.topk_weights = torch.full((2, 2), 1.5, dtype=torch.float32)
        inp.quant.quant_type = QuantType.W8A8
        inp.routing.mc2_mask = torch.ones(2, dtype=torch.bool)
        inp.layer = SimpleNamespace(
            routed_scaling_factor=1.5,
            apply_routed_scale_to_output=False,
            ascend_megamoe_fold_scale=fold_scale,
            mega_moe_activation_kwargs={},
            ascend_megamoe_shared_fusion=SimpleNamespace(quant_out_dtype=258, weight_type=258),
        )
        comm_impl._apply_cann_mega_moe(inp, weights, is_decode_only_node=True)
        torch.testing.assert_close(captured["topk_weights"], torch.full((2, 2), 1.5))

    inp_out_scaled = MagicMock()
    inp_out_scaled.hidden_states = torch.zeros(2, 8, dtype=torch.bfloat16)
    inp_out_scaled.topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    inp_out_scaled.topk_weights = torch.ones(2, 2, dtype=torch.float32)
    inp_out_scaled.quant.quant_type = QuantType.W8A8
    inp_out_scaled.routing.mc2_mask = torch.ones(2, dtype=torch.bool)
    inp_out_scaled.layer = SimpleNamespace(
        routed_scaling_factor=1.0,
        apply_routed_scale_to_output=True,
        ascend_megamoe_fold_scale=1.5,
        mega_moe_activation_kwargs={},
        ascend_megamoe_shared_fusion=SimpleNamespace(quant_out_dtype=258, weight_type=258),
    )
    comm_impl._apply_cann_mega_moe(inp_out_scaled, weights, is_decode_only_node=True)
    torch.testing.assert_close(captured["topk_weights"], torch.full((2, 2), 1.5))


def test_resolve_use_mega_moe_requires_fused_mc2(monkeypatch):
    vllm_config = MagicMock()
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda cfg: cfg is vllm_config)
    assert afc.resolve_use_mega_moe(vllm_config, MoECommType.FUSED_MC2) is True
    assert afc.resolve_use_mega_moe(vllm_config, MoECommType.ALLTOALL) is False
    assert afc.resolve_use_mega_moe(vllm_config, MoECommType.MC2) is False
    assert afc.resolve_use_mega_moe(None, MoECommType.FUSED_MC2) is False


def test_moe_config_key_isolates_distinct_shapes():
    from vllm_ascend.ops.fused_moe.moe_comm_method import _moe_config_key

    target_cfg = SimpleNamespace(
        num_experts=64,
        num_local_experts=8,
        hidden_dim=5120,
        intermediate_size_per_partition=1536,
        experts_per_token=6,
    )
    draft_cfg = SimpleNamespace(
        num_experts=64,
        num_local_experts=8,
        hidden_dim=2048,
        intermediate_size_per_partition=1024,
        experts_per_token=4,
    )
    assert _moe_config_key(MoECommType.FUSED_MC2, target_cfg) != _moe_config_key(MoECommType.FUSED_MC2, draft_cfg)


def test_unquantized_moe_draft_stays_on_3d_nz_and_maybe_transpose_2d_list(monkeypatch):
    import vllm_ascend.ops.fused_moe.routed_experts as routed_mod
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    method.dynamic_eplb = False
    method._maybe_pad_weight = lambda w: w

    # 2D per-expert weight list must transpose (0, 1) without IndexError.
    t_list = [torch.randn(8, 16), torch.randn(8, 16)]
    transposed = method._maybe_transpose(t_list, need_trans=True)
    assert [tuple(t.shape) for t in transposed] == [(16, 8), (16, 8)]

    # Draft unquantized layer must bypass MegaMoe ND list split and keep 3D NZ weights.
    layer = torch.nn.Module()
    layer.register_parameter("w13_weight", torch.nn.Parameter(torch.zeros(2, 16, 8), requires_grad=False))
    layer.register_parameter("w2_weight", torch.nn.Parameter(torch.zeros(2, 8, 8), requires_grad=False))
    format_calls = []

    def fake_format_cast(tensor, fmt):
        format_calls.append(fmt)
        return tensor

    draft_cfg = SimpleNamespace(model_config=SimpleNamespace(runner_type="draft"), speculative_config=None)
    monkeypatch.setattr(routed_mod, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    monkeypatch.setattr(routed_mod, "get_current_vllm_config", lambda: draft_cfg)
    monkeypatch.setattr(routed_mod, "use_cann_megamoe", lambda _cfg: True)
    monkeypatch.setattr(routed_mod, "is_draft_vllm_config", lambda _cfg: True)
    monkeypatch.setattr(routed_mod.torch_npu, "npu_format_cast", fake_format_cast, raising=False)

    method.process_weights_after_loading(layer)
    assert hasattr(layer, "w13_weight")
    assert hasattr(layer, "w2_weight")
    assert not hasattr(layer, "w13_weight_list")
    assert format_calls == [routed_mod.ACL_FORMAT_FRACTAL_NZ, routed_mod.ACL_FORMAT_FRACTAL_NZ]
