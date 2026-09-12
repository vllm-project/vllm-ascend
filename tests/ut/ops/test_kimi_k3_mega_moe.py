from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

import vllm_ascend.ascend_config as ascend_config_module
import vllm_ascend.ascend_forward_context as afc
import vllm_ascend.ops.fused_moe.moe_comm_method as moe_comm_module
from vllm_ascend.ascend_config import AscendConfig, clear_ascend_config, init_ascend_config, is_mega_moe_supported
from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType


def test_enable_fused_mc2_mode_2_normalizes_to_internal_mode_1(monkeypatch):
    """Mode 2 is user-facing; runtime keeps using mode 1 plus the MegaMoe flag."""
    clear_ascend_config()
    monkeypatch.setattr(ascend_config_module, "_MEGA_MOE_SUPPORTED", None)
    monkeypatch.setattr(
        ascend_config_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "cann_ops_transformer" else None,
    )
    monkeypatch.setattr(
        AscendConfig,
        "_is_megamoe_supported_by_config",
        staticmethod(lambda _vllm_config: True),
    )
    monkeypatch.setattr(
        "vllm_ascend.platform.NPUPlatform.check_and_update_config",
        lambda *args, **kwargs: None,
    )

    vllm_config = VllmConfig()
    vllm_config.additional_config = {"enable_fused_mc2": 2}

    config = init_ascend_config(vllm_config)

    assert config.enable_fused_mc2 == 1
    assert is_mega_moe_supported() is True
    clear_ascend_config()


def test_enable_fused_mc2_mode_1_keeps_megamoe_rolled_back(monkeypatch):
    clear_ascend_config()
    monkeypatch.setattr(ascend_config_module, "_MEGA_MOE_SUPPORTED", True)
    monkeypatch.setattr(
        AscendConfig,
        "_is_megamoe_supported_by_config",
        staticmethod(lambda _vllm_config: True),
    )
    monkeypatch.setattr(
        "vllm_ascend.platform.NPUPlatform.check_and_update_config",
        lambda *args, **kwargs: None,
    )

    vllm_config = VllmConfig()
    vllm_config.additional_config = {"enable_fused_mc2": 1}

    config = init_ascend_config(vllm_config)

    assert config.enable_fused_mc2 == 1
    assert is_mega_moe_supported() is False
    clear_ascend_config()


def test_kimi_megamoe_capability_uses_routed_expert_dimensions():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=7168,
                routed_expert_hidden_size=3584,
                routed_expert_intermediate_size=3072,
                moe_quantize="w4a8",
            )
        )
    )

    assert AscendConfig._is_megamoe_supported_by_config(vllm_config)


@pytest.mark.parametrize("ep_world_size", [2, 4, 8, 16, 32, 64, 128])
def test_cann_megamoe_accepts_supported_ep_sizes(monkeypatch, ep_world_size):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    monkeypatch.setattr(afc, "is_moe_model", lambda _vllm_config: True)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
        lora_config=None,
    )

    assert afc.use_cann_megamoe(vllm_config) is True


@pytest.mark.parametrize("ep_world_size", [3, 56, 129])
def test_cann_megamoe_rejects_unsupported_ep_sizes(monkeypatch, ep_world_size):
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A3)
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    monkeypatch.setattr(afc, "is_moe_model", lambda _vllm_config: True)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=ep_world_size))
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
        lora_config=None,
    )

    assert afc.use_cann_megamoe(vllm_config) is False


def test_kimi_megamoe_symm_buffer_uses_routed_dimensions(monkeypatch):
    comm_impl = object.__new__(FusedMC2CommImpl)
    comm_impl.moe_config = SimpleNamespace(
        num_experts=896,
        experts_per_token=16,
        hidden_dim=3584,
        intermediate_size_per_partition=3072,
    )
    comm_impl.token_dispatcher = MagicMock(spec=TokenDispatcherWithMC2)
    comm_impl.token_dispatcher.global_bs = 4096
    comm_impl.token_dispatcher.ep_world_size = 64
    comm_impl.token_dispatcher.ep_rank_id = 0
    comm_impl.get_symm_buffer_for_mega_moe = MagicMock(return_value=object())

    monkeypatch.setattr(
        moe_comm_module,
        "get_mc2_group",
        lambda: SimpleNamespace(device_group=object()),
    )
    monkeypatch.setattr(
        moe_comm_module,
        "get_ascend_config",
        lambda: SimpleNamespace(mega_moe_max_tokens=65536),
    )

    comm_impl._init_mega_moe_symm_buffer(
        dispatch_quant_mode=2,
        dispatch_quant_out_dtype=torch.int8,
        is_decode_only_node=False,
    )

    call = comm_impl.get_symm_buffer_for_mega_moe.call_args
    assert call.kwargs["hidden"] == 3584
    assert call.kwargs["intermediate_hidden"] == 6144
    assert call.kwargs["max_recv_token_num"] == 65536


def test_cann_mega_moe_maps_kimi_situ_activation():
    activation, params = moe_utils.get_cann_mega_moe_activation_settings(
        MoEActivation.SITU,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    assert activation == "situglu"
    assert params == {"beta": 4.0, "linear_beta": 25.0}


def test_cann_mega_moe_detects_situ_api():
    def mega_moe(*args, activation="swiglu", activation_params=None):
        return activation, activation_params

    def legacy_mega_moe(*args, activation_clamp=None):
        return activation_clamp

    assert moe_utils.cann_mega_moe_supports_situ(mega_moe)
    assert not moe_utils.cann_mega_moe_supports_situ(legacy_mega_moe)


def _build_comm_impl_for_operator_test():
    comm_impl = object.__new__(FusedMC2CommImpl)
    comm_impl.moe_config = SimpleNamespace(
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm_impl.token_dispatcher.global_bs = 1
    comm_impl.token_dispatcher.max_num_tokens_per_rank = 8
    comm_impl.mega_moe_symm_buffer = SimpleNamespace(
        dispatch_quant_mode=0,
        dispatch_quant_out_dtype=None,
    )
    comm_impl.swiglu_limit = 0.0
    return comm_impl


def test_cann_mega_moe_forwards_kimi_situ_to_operator():
    comm_impl = _build_comm_impl_for_operator_test()
    comm_impl._mega_moe_supports_situ = True
    expected = torch.randn(2, 4)
    expert_tokens = torch.ones(2, dtype=torch.int32)
    comm_impl.mega_moe = MagicMock(return_value=(expected, expert_tokens))
    fused_input = build_fused_experts_input(
        hidden_states=torch.randn(2, 4),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        topk_weights=torch.ones(2, 1),
        w1=torch.randn(2, 4),
        w2=torch.randn(2, 4),
        quant_type=QuantType.NONE,
        dynamic_eplb=False,
        activation=MoEActivation.SITU,
    )

    result, result_expert_tokens = comm_impl._apply_cann_mega_moe(
        fused_input,
        is_decode_only_node=False,
    )

    assert result is expected
    assert result_expert_tokens is expert_tokens
    call = comm_impl.mega_moe.call_args
    assert call.kwargs["activation"] == "situglu"
    assert call.kwargs["activation_params"] == {"beta": 4.0, "linear_beta": 25.0}
    assert call.kwargs["l1_weights_sf"] is None
    assert call.kwargs["l2_weights_sf"] is None


def _run_scale_normalization_case(scale_payload):
    comm_impl = _build_comm_impl_for_operator_test()
    comm_impl._mega_moe_supports_situ = False
    output = torch.randn(2, 4)
    expert_tokens = torch.ones(1, dtype=torch.int32)
    comm_impl.mega_moe = MagicMock(return_value=(output, expert_tokens))

    fused_input = build_fused_experts_input(
        hidden_states=torch.randn(2, 4),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        w1=[torch.randn(4, 8)],
        w2=[torch.randn(8, 4)],
        quant_type=QuantType.W4A8,
        dynamic_eplb=False,
        w1_scale=scale_payload,
        w2_scale=scale_payload,
    )

    comm_impl._apply_cann_mega_moe(fused_input, is_decode_only_node=False)
    call = comm_impl.mega_moe.call_args
    return call.kwargs["l1_weights_sf"], call.kwargs["l2_weights_sf"]


@pytest.mark.parametrize("as_list", [False, True])
def test_megamoe_scale_normalization_squeezes_leading_singleton(as_list):
    scale = torch.ones(1, 8)
    payload = [scale] if as_list else scale

    l1_scales, l2_scales = _run_scale_normalization_case(payload)

    assert len(l1_scales) == 1
    assert len(l2_scales) == 1
    assert l1_scales[0].shape == (8,)
    assert l2_scales[0].shape == (8,)


@pytest.mark.parametrize("as_list", [False, True])
def test_megamoe_scale_normalization_preserves_group_scales(as_list):
    scale = torch.ones(2, 8)
    payload = [scale] if as_list else scale

    l1_scales, l2_scales = _run_scale_normalization_case(payload)

    assert len(l1_scales) == 1
    assert len(l2_scales) == 1
    assert l1_scales[0].shape == (2, 8)
    assert l2_scales[0].shape == (2, 8)
