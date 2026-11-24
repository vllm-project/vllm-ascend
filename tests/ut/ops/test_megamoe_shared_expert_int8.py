# SPDX-License-Identifier: Apache-2.0
"""Guards for the A3 INT8 shared-expert fusion path."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe import moe_comm_method, moe_utils
from vllm_ascend.ops.fused_moe import shared_experts as shared_experts_module
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MegamoeSharedExpertFusion, MegamoeSharedWeights
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.w4a8 import w4a8
from vllm_ascend.quantization.quant_type import QuantType

_MODULE = shared_experts_module.__name__


def _make_runner(quant_type: QuantType) -> AscendSharedExperts:
    """Build AscendSharedExperts without __init__; only the fusion state is used."""
    runner = object.__new__(AscendSharedExperts)
    runner.quant_type = quant_type
    runner.moe_config = SimpleNamespace(hidden_dim=4096, intermediate_size_per_partition=2048)
    runner.megamoe_shared_weights_ready = False
    runner._megamoe_shared_fusion_disabled = False
    runner._megamoe_shared_parts = {}
    runner._megamoe_shared_quant_type = QuantType.NONE
    runner._megamoe_shared_fusion = shared_experts_module.MegamoeSharedExpertFusion()
    # Mirror the remaining __init__ fields so this helper cannot drift again.
    runner._megamoe_shared_weight_dtype = None
    runner._megamoe_shared_fusion_platform = "A3/INT8"
    return runner


def _shared_linear(weight_dtype: torch.dtype, shared_quant_type: QuantType) -> SimpleNamespace:
    return SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(4096, 4096, dtype=weight_dtype)),
        quant_method=SimpleNamespace(quant_type=shared_quant_type),
    )


@pytest.fixture
def cann_mega_moe_abi(monkeypatch, mixed_int_abi):
    def mixed_mega_moe(
        *,
        shared_l1_weights=None,
        shared_l2_weights=None,
        shared_l1_weights_sf=None,
        shared_l2_weights_sf=None,
        shared_expert_quant_out_dtype=None,
        shared_weight1_type=None,
        shared_weight2_type=None,
    ):
        pass

    def legacy_mega_moe(
        *,
        shared_l1_weights=None,
        shared_l2_weights=None,
        shared_l1_weights_sf=None,
        shared_l2_weights_sf=None,
        shared_expert_quant_out_dtype=None,
        shared_weight1_type=None,
    ):
        pass

    ops = ModuleType("cann_ops_transformer.ops")
    monkeypatch.setattr(ops, "mega_moe", mixed_mega_moe if mixed_int_abi else legacy_mega_moe, raising=False)
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    detect_abi = moe_utils.supports_cann_mega_moe_mixed_int_shared
    detect_abi.cache_clear()
    yield
    detect_abi.cache_clear()


@pytest.mark.parametrize(
    "quant_type,mixed_int_abi,expected",
    [
        pytest.param(QuantType.W8A8, False, True, id="w8a8-legacy-abi"),
        pytest.param(QuantType.W4A8, False, False, id="w4a8-legacy-abi"),
        pytest.param(QuantType.W4A8, True, True, id="w4a8-mixed-int-abi"),
    ],
)
@pytest.mark.usefixtures("cann_mega_moe_abi")
def test_setup_a3_shared_expert_fusion(quant_type, expected) -> None:
    runner = _make_runner(quant_type)
    runner.layer = SimpleNamespace(gate_up_proj=SimpleNamespace(), down_proj=SimpleNamespace())
    runner.moe_config.tp_group = SimpleNamespace(world_size=1)
    runner._routed_megamoe_activation_signature = Mock(return_value=("silu", ()))
    runner._shared_megamoe_activation_signature = Mock(return_value=("silu", ()))
    runner._wrap_linear_for_megamoe_fusion = Mock()
    with (
        patch(f"{_MODULE}.use_cann_megamoe", return_value=True),
        patch(f"{_MODULE}.get_current_vllm_config", return_value=object()),
        patch(
            f"{_MODULE}.get_ascend_config",
            return_value=SimpleNamespace(megamoe_shared_expert_fusion=True),
        ),
        patch(
            f"{_MODULE}.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A3),
        ),
    ):
        runner._setup_megamoe_shared_fusion()

    assert runner._megamoe_shared_fusion_disabled is not expected
    assert runner._wrap_linear_for_megamoe_fusion.call_count == (2 if expected else 0)


def test_collect_rejects_packed_int32_shared_weight() -> None:
    """A W4A8 shared linear materializes int32 packed weights: keep the separate forward."""
    runner = _make_runner(QuantType.W8A8)
    runner._collect_megamoe_shared_weight("w1", _shared_linear(torch.int32, QuantType.W4A8))

    assert runner._megamoe_shared_fusion_disabled is True
    assert runner._megamoe_shared_parts == {}


def test_collect_rejects_w4a8_shared_quant_type() -> None:
    """An int8 buffer that declares W4A8 does not match: reject it."""
    runner = _make_runner(QuantType.W8A8)
    runner._collect_megamoe_shared_weight("w1", _shared_linear(torch.int8, QuantType.W4A8))

    assert runner._megamoe_shared_fusion_disabled is True
    assert runner._megamoe_shared_parts == {}


def test_collect_rejects_int8_shared_on_a5_mxfp() -> None:
    """An int8 shared linear cannot ride the A5 MXFP-only MegaMoe payload family."""
    runner = _make_runner(QuantType.W8A8MXFP)
    runner._megamoe_shared_fusion_platform = "A5/MXFP"
    runner._collect_megamoe_shared_weight("w1", _shared_linear(torch.int8, QuantType.W8A8))

    assert runner._megamoe_shared_fusion_disabled is True
    assert runner._megamoe_shared_parts == {}


def test_collect_rejects_mxfp_shared_on_a3_int8() -> None:
    """An MXFP shared linear cannot ride the arch22 INT8 MegaMoe payload family."""
    runner = _make_runner(QuantType.W8A8)
    runner._collect_megamoe_shared_weight("w1", _shared_linear(torch.float8_e4m3fn, QuantType.W8A8MXFP))

    assert runner._megamoe_shared_fusion_disabled is True
    assert runner._megamoe_shared_parts == {}


def test_collect_int8_payload_matches_routed_scale_dtype() -> None:
    """The A3 INT8 payload must carry the same int64 scale container as routed."""
    runner = _make_runner(QuantType.W8A8)
    linear = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(4096, 4096, dtype=torch.int8)),
        weight_scale=SimpleNamespace(data=torch.full((4096,), 0.0078125)),
        quant_method=SimpleNamespace(quant_type=QuantType.W8A8),
    )
    runner._collect_megamoe_shared_weight("w1", linear)

    scale = runner._megamoe_shared_parts["w1_scale"]
    assert scale.dtype is torch.int64
    assert int(scale[0]) == 1006632960
    assert tuple(runner._megamoe_shared_parts["w1"].shape) == (4096, 4096)


_INTEGER_SHARED_ABI = frozenset(
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


@pytest.mark.parametrize(
    "use_expert_weight_list,use_mega_moe,shared_state",
    [
        pytest.param(False, True, "ready", id="tensor-ready-megamoe"),
        pytest.param(False, False, "ready", id="tensor-ready-non-megamoe"),
        pytest.param(True, True, "ready", id="list-ready-megamoe"),
        pytest.param(True, False, "ready", id="list-ready-non-megamoe"),
        pytest.param(False, True, "missing", id="megamoe-fusion-missing"),
        pytest.param(False, True, "pending", id="megamoe-weights-pending"),
    ],
)
def test_w4a8_shared_weights_reach_megamoe(monkeypatch, use_expert_weight_list, use_mega_moe, shared_state):
    context = SimpleNamespace(use_mega_moe=use_mega_moe, is_decode_only_node=False)
    monkeypatch.setattr(w4a8, "_EXTRA_CTX", context)
    monkeypatch.setattr(moe_comm_method, "_EXTRA_CTX", context)
    monkeypatch.setattr(
        moe_comm_method, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3)
    )
    method = w4a8.AscendW4A8DynamicFusedMoEMethod.__new__(w4a8.AscendW4A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = use_expert_weight_list
    layer = SimpleNamespace()
    for name in ("w13", "w2"):
        weight = torch.zeros(4, 4, dtype=torch.int8)
        scale = torch.ones(4)
        bias = torch.zeros(4)
        for suffix, tensor in (("weight", weight), ("weight_scale", scale), ("scale_bias", bias)):
            setattr(layer, f"{name}_{suffix}", tensor)
            setattr(layer, f"{name}_{suffix}_list", [tensor])
            setattr(layer, f"cann_mega_moe_{name}_{suffix}_list", [tensor])
    shared = MegamoeSharedWeights(
        w1=[torch.ones(4, 4, dtype=torch.int8)],
        w2=[torch.ones(2, 4, dtype=torch.int8)],
        w1_scale=[torch.ones(4, dtype=torch.int64)],
        w2_scale=[torch.ones(4, dtype=torch.int64)],
    )
    _, quant_out_dtype, weight_type = moe_utils._get_cann_mega_moe_quant_settings(QuantType.W8A8)
    if shared_state != "missing":
        layer.ascend_megamoe_shared_fusion = MegamoeSharedExpertFusion(
            weights=shared if shared_state == "ready" else None,
            quant_out_dtype=quant_out_dtype,
            weight_type=weight_type,
        )
    weights = method.get_fused_mc2_weights(layer)
    active = use_mega_moe and shared_state == "ready"
    for name in ("w1", "w2", "w1_scale", "w2_scale"):
        assert getattr(weights, f"shared_{name}") is (getattr(shared, name) if active else None)
    if not use_mega_moe:
        return

    # Execute the real communicator and W4A8 getter. Only the native kernel is
    # mocked, so a dropped payload cannot be hidden by a mocked getter.
    comm = FusedMC2CommImpl.__new__(FusedMC2CommImpl)
    comm.token_dispatcher = Mock(spec=TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = 4
    comm.token_dispatcher.global_bs = 0
    comm.enable_fused_mc2 = 1
    comm.mega_moe_symm_buffer = SimpleNamespace()
    comm._mega_moe_supported_kwargs = _INTEGER_SHARED_ABI
    comm.mega_moe = Mock(return_value=(torch.zeros(2, 4), torch.zeros(1, dtype=torch.int64)))
    inputs = SimpleNamespace(
        hidden_states=torch.ones(2, 4),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
        topk_weights=torch.ones(2, 1),
        routing=SimpleNamespace(mc2_mask=None),
        quant=SimpleNamespace(quant_type=QuantType.W4A8),
        layer=layer,
    )
    comm.fused_experts(inputs, quant_method=method)
    kwargs = comm.mega_moe.call_args.kwargs
    if active:
        assert kwargs["shared_l1_weights"] is shared.w1
        assert kwargs["shared_l2_weights"] is shared.w2
        assert kwargs["shared_l1_weights_sf"] is shared.w1_scale
        assert kwargs["shared_l2_weights_sf"] is shared.w2_scale
        assert kwargs["shared_expert_quant_out_dtype"] == quant_out_dtype
        assert kwargs["shared_weight1_type"] == kwargs["shared_weight2_type"] == weight_type
    else:
        assert not (_INTEGER_SHARED_ABI & kwargs.keys())
