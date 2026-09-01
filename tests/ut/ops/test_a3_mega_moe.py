from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.fused_moe import a3_mega_moe as a3_moe
from vllm_ascend.ops.fused_moe import moe_comm_method
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType


def _make_vllm_config(*, quantize=None, ep_size=8):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=4096,
                moe_intermediate_size=2048,
                quantize=quantize,
            ),
        ),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            world_size_across_dp=ep_size,
            pipeline_parallel_size=1,
        ),
    )


def _make_ascend_config(vllm_config):
    return SimpleNamespace(
        enable_fused_mc2=1,
        vllm_config=vllm_config,
        mix_placement=False,
        eplb_config=SimpleNamespace(
            dynamic_eplb=False,
            num_redundant_experts=0,
        ),
    )


@pytest.mark.parametrize("quantize", [None, "w8a8_dynamic", "w4a8_dynamic"])
def test_a3_mega_moe_capability_accepts_mainline_configs(monkeypatch, quantize):
    vllm_config = _make_vllm_config(quantize=quantize)
    monkeypatch.setattr(a3_moe, "get_ascend_device_type", lambda: AscendDeviceType.A3)
    monkeypatch.setattr(a3_moe, "is_a3_mega_moe_package_available", lambda: True)
    monkeypatch.setattr(a3_moe, "get_ascend_config", lambda: _make_ascend_config(vllm_config))

    assert a3_moe.is_a3_mega_moe_enabled(vllm_config)


@pytest.mark.parametrize(
    ("device_type", "ep_size", "dynamic_eplb", "expected"),
    [
        (AscendDeviceType.A5, 8, False, False),
        (AscendDeviceType.A3, 65, False, False),
        (AscendDeviceType.A3, 8, True, False),
        (AscendDeviceType.A3, 8, False, True),
    ],
)
def test_a3_mega_moe_capability_is_hardware_and_config_scoped(
    monkeypatch,
    device_type,
    ep_size,
    dynamic_eplb,
    expected,
):
    vllm_config = _make_vllm_config(ep_size=ep_size)
    ascend_config = _make_ascend_config(vllm_config)
    ascend_config.eplb_config.dynamic_eplb = dynamic_eplb
    monkeypatch.setattr(a3_moe, "get_ascend_device_type", lambda: device_type)
    monkeypatch.setattr(a3_moe, "is_a3_mega_moe_package_available", lambda: True)
    monkeypatch.setattr(a3_moe, "get_ascend_config", lambda: ascend_config)

    assert a3_moe.is_a3_mega_moe_enabled(vllm_config) is expected


def test_setup_moe_comm_method_keeps_a5_on_existing_fused_backend(monkeypatch):
    moe_config = SimpleNamespace(ep_size=8)
    legacy_backend = object()
    a3_backend = object()
    monkeypatch.setattr(moe_comm_method, "AlltoAllCommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_method, "AllGatherCommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_method, "MC2CommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_method, "FusedMC2CommImpl", MagicMock(return_value=legacy_backend))
    monkeypatch.setattr(moe_comm_method, "A3MegaMoECommImpl", MagicMock(return_value=a3_backend))
    monkeypatch.setattr(moe_comm_method, "is_a3_mega_moe_enabled", lambda: False)

    moe_comm_method.setup_moe_comm_method(moe_config)

    assert moe_comm_method.get_moe_comm_method(moe_comm_method.MoECommType.FUSED_MC2) is legacy_backend


def test_setup_moe_comm_method_selects_a3_backend_only_when_enabled(monkeypatch):
    moe_config = SimpleNamespace(ep_size=8)
    legacy_backend = object()
    a3_backend = object()
    monkeypatch.setattr(moe_comm_method, "AlltoAllCommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_method, "AllGatherCommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_method, "MC2CommImpl", MagicMock())
    monkeypatch.setattr(moe_comm_method, "FusedMC2CommImpl", MagicMock(return_value=legacy_backend))
    monkeypatch.setattr(moe_comm_method, "A3MegaMoECommImpl", MagicMock(return_value=a3_backend))
    monkeypatch.setattr(moe_comm_method, "is_a3_mega_moe_enabled", lambda: True)

    moe_comm_method.setup_moe_comm_method(moe_config)

    assert moe_comm_method.get_moe_comm_method(moe_comm_method.MoECommType.FUSED_MC2) is a3_backend


def test_a3_mega_moe_backend_uses_rank_invariant_capacity_and_active_mask(monkeypatch):
    symmetric_buffer = SimpleNamespace(
        dispatch_quant_mode=None,
        dispatch_quant_out_dtype=None,
    )
    get_symmetric_buffer = MagicMock(return_value=symmetric_buffer)
    mega_moe = MagicMock(return_value=(torch.randn(3, 8), torch.tensor([1, 2], dtype=torch.int32)))
    monkeypatch.setattr(a3_moe, "_get_a3_mega_moe_ops", lambda: (get_symmetric_buffer, mega_moe))
    device_group = object()
    monkeypatch.setattr(
        "vllm_ascend.distributed.parallel_state.get_mc2_group",
        lambda: SimpleNamespace(device_group=device_group),
    )
    moe_config = SimpleNamespace(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=8,
        intermediate_size_per_partition=4,
    )
    dispatcher = SimpleNamespace(
        ep_world_size=4,
        ep_rank_id=1,
        max_num_tokens_per_rank=16,
        global_bs=0,
    )
    backend = a3_moe.A3MegaMoEBackend(moe_config, dispatcher)
    w1 = [torch.randn(8, 8) for _ in range(2)]
    w2 = [torch.randn(8, 4) for _ in range(2)]
    fused_input = build_fused_experts_input(
        hidden_states=torch.randn(3, 8),
        topk_weights=torch.randn(3, 2),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int64),
        w1=w1,
        w2=w2,
        quant_type=QuantType.NONE,
        dynamic_eplb=False,
        mc2_mask=torch.tensor([True, True, False]),
    )

    output, expert_tokens = backend.fused_experts(fused_input)

    assert output.shape == (3, 8)
    assert expert_tokens.tolist() == [1, 2]
    get_symmetric_buffer.assert_called_once_with(
        device_group,
        8,
        16,
        2,
        hidden=8,
        intermediate_hidden=8,
        max_recv_token_num=128,
        dispatch_quant_mode=0,
        dispatch_quant_out_dtype=None,
    )
    call = mega_moe.call_args
    assert call.args[3] is w1
    assert call.args[4] is w2
    assert call.kwargs["x_active_mask"].dtype == torch.int8
    assert call.kwargs["l1_weights_sf"] is None
    assert call.kwargs["weight1_type"] is None


@pytest.mark.parametrize(
    ("quant_type", "expected_settings"),
    [
        (QuantType.NONE, (0, None, None)),
        (QuantType.W8A8, (2, 258, 258)),
        (QuantType.W4A8, (2, 258, 285)),
    ],
)
def test_a3_mega_moe_quant_settings(quant_type, expected_settings):
    assert a3_moe._get_a3_mega_moe_quant_settings(quant_type) == expected_settings
