# SPDX-License-Identifier: Apache-2.0
"""A3 shared-expert format admission, ABI gating, and scale encoding."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe import moe_comm_method, moe_utils
from vllm_ascend.ops.fused_moe import shared_experts as shared_mod
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MegamoeSharedExpertFusion, MegamoeSharedWeights
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.w4a8 import w4a8
from vllm_ascend.quantization.quant_type import QuantType

_HIDDEN = 1024
_SHARED_INTERMEDIATE = 512

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


def _projection(in_dim, out_dim, dtype, quant_type, *, packed_scale=False):
    linear = nn.Module()
    linear.bias = None
    linear.weight = nn.Parameter(torch.zeros((in_dim, out_dim), dtype=dtype), requires_grad=False)
    scale = torch.full((out_dim,), 0.0078125)
    if packed_scale:
        # Set nonzero upper bits to check the low-32-bit decoding contract.
        scale = scale.view(torch.int32).to(torch.int64) | (1 << 40)
    linear.weight_scale = nn.Parameter(scale, requires_grad=False)
    linear.weight_scale_fp32 = torch.full((out_dim,), 99.0)
    linear.quant_method = SimpleNamespace(quant_method=SimpleNamespace(quant_type=quant_type))
    return linear


def _make_executor(
    monkeypatch,
    routed,
    shared,
    weight_dtype,
    profile,
    *,
    packed_scale=False,
    patch_mixed_abi: bool = True,
):
    monkeypatch.setattr(shared_mod, "get_current_vllm_config", SimpleNamespace)
    monkeypatch.setattr(shared_mod, "use_cann_megamoe", lambda _: True)
    monkeypatch.setattr(shared_mod, "shared_expert_tp_enable", lambda: False)
    monkeypatch.setattr(shared_mod, "get_current_hardware_profile", lambda: get_hardware_profile(profile))
    if patch_mixed_abi:
        monkeypatch.setattr(moe_utils, "supports_cann_mega_moe_mixed_int_shared", lambda: True)
    monkeypatch.setattr(
        shared_mod,
        "get_ascend_config",
        lambda: SimpleNamespace(
            megamoe_shared_expert_fusion=True,
            multistream_overlap_shared_expert=False,
            enable_shared_expert_dp=False,
        ),
    )
    layer = SimpleNamespace(
        gate_up_proj=_projection(_HIDDEN, 2 * _SHARED_INTERMEDIATE, weight_dtype, shared, packed_scale=packed_scale),
        down_proj=_projection(_SHARED_INTERMEDIATE, _HIDDEN, weight_dtype, shared, packed_scale=packed_scale),
        act_fn=SiluAndMul.__new__(SiluAndMul),
    )
    config = SimpleNamespace(
        hidden_dim=_HIDDEN,
        intermediate_size_per_partition=_SHARED_INTERMEDIATE,
        in_dtype=torch.bfloat16,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        is_sequence_parallel=False,
        tp_group=SimpleNamespace(world_size=1),
    )
    return AscendSharedExperts(layer, config, routed, None)


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
def test_setup_a3_shared_expert_fusion(monkeypatch, quant_type, expected) -> None:
    executor = _make_executor(
        monkeypatch,
        quant_type,
        QuantType.W8A8,
        torch.int8,
        AscendDeviceType.A3,
        patch_mixed_abi=False,
    )
    assert (executor._megamoe_formats is not None) is expected
    executor.refresh_megamoe_shared_weights()
    assert executor.megamoe_shared_weights_ready is expected


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


@pytest.mark.parametrize("routed", [QuantType.W4A8, QuantType.W8A8])
@pytest.mark.parametrize("packed_scale", [False, True])
def test_a3_w8a8_shared_payload(monkeypatch, routed, packed_scale):
    executor = _make_executor(
        monkeypatch,
        routed,
        QuantType.W8A8,
        torch.int8,
        AscendDeviceType.A3,
        packed_scale=packed_scale,
    )
    executor.refresh_megamoe_shared_weights()

    fusion = executor.megamoe_shared_fusion()
    weights = fusion.weights
    assert weights is not None
    assert fusion.weight_type == 258
    assert tuple(weights.w1[0].shape) == (_HIDDEN, 2 * _SHARED_INTERMEDIATE)
    assert tuple(weights.w2[0].shape) == (_SHARED_INTERMEDIATE, _HIDDEN)
    for scale in (weights.w1_scale[0], weights.w2_scale[0]):
        assert scale.dtype == torch.int64
        torch.testing.assert_close(scale, torch.full_like(scale, 1006632960))


@pytest.mark.parametrize(
    ("routed", "shared", "carrier", "profile"),
    [
        (QuantType.W4A8, QuantType.W4A8, torch.int32, AscendDeviceType.A3),
        (QuantType.W8A8, QuantType.W4A8, torch.int8, AscendDeviceType.A3),
        (QuantType.W8A8MXFP, QuantType.W8A8, torch.int8, AscendDeviceType.A5),
        (QuantType.W8A8, QuantType.W8A8MXFP, torch.float8_e4m3fn, AscendDeviceType.A3),
    ],
)
def test_incompatible_shared_format_stays_standalone(monkeypatch, routed, shared, carrier, profile):
    executor = _make_executor(monkeypatch, routed, shared, carrier, profile)
    executor.refresh_megamoe_shared_weights()

    assert not executor.megamoe_shared_weights_ready
    assert executor.megamoe_shared_fusion().weights is None


@pytest.mark.parametrize("invalid", ["asymmetric", "grouped_scale"])
def test_a3_int8_rejects_incompatible_scale_contract(monkeypatch, invalid):
    executor = _make_executor(monkeypatch, QuantType.W4A8, QuantType.W8A8, torch.int8, AscendDeviceType.A3)
    down = executor.layer.down_proj
    if invalid == "asymmetric":
        down.weight_offset = nn.Parameter(torch.ones(_HIDDEN), requires_grad=False)
    else:
        down.weight_scale = nn.Parameter(torch.ones(_HIDDEN, 2), requires_grad=False)
    executor.refresh_megamoe_shared_weights()

    assert not executor.megamoe_shared_weights_ready


def test_a3_bf16_shared_payload_uses_dense_transposed_weights(monkeypatch):
    executor = _make_executor(monkeypatch, QuantType.NONE, QuantType.NONE, torch.bfloat16, AscendDeviceType.A3)
    for projection in (executor.layer.gate_up_proj, executor.layer.down_proj):
        dense_weight = (projection.weight.T + torch.arange(projection.weight.size(0))).contiguous()
        projection.weight = nn.Parameter(dense_weight.bfloat16(), requires_grad=False)
        del projection.weight_scale
    executor.refresh_megamoe_shared_weights()

    weights = executor.megamoe_shared_fusion().weights
    assert weights is not None
    assert weights.w1_scale is None and weights.w2_scale is None
    torch.testing.assert_close(weights.w1[0], executor.layer.gate_up_proj.weight.T)
    torch.testing.assert_close(weights.w2[0], executor.layer.down_proj.weight.T)


@pytest.mark.parametrize(
    "shared,carrier", [(QuantType.W8A8MXFP, torch.float8_e4m3fn), (QuantType.W4A8MXFP, torch.uint8)]
)
def test_a5_shared_mxfp_refills_e8m0_scale_storage(monkeypatch, shared, carrier):
    executor = _make_executor(monkeypatch, QuantType.W8A8MXFP, shared, carrier, AscendDeviceType.A5)
    projections = executor.layer.gate_up_proj, executor.layer.down_proj
    gen = torch.Generator().manual_seed(17)
    for projection in projections:
        in_dim, out_dim = projection.weight.shape
        if carrier == torch.uint8:
            projection.weight = nn.Parameter(projection.weight[: in_dim // 2].clone(), requires_grad=False)
        projection.weight_scale = nn.Parameter(
            torch.randint(1, 200, (in_dim // 64, out_dim, 2), dtype=torch.uint8, generator=gen),
            requires_grad=False,
        )
    executor.refresh_megamoe_shared_weights()
    weights = executor.megamoe_shared_fusion().weights
    assert weights is not None
    assert weights.w1_scale[0].dtype == weights.w2_scale[0].dtype == torch.float8_e8m0fnu
    pointers = tuple(operand.data_ptr() for operand in weights.operands())
    gen_new = torch.Generator().manual_seed(42)
    expected_scales = []
    for projection in projections:
        updated = torch.randint(1, 200, projection.weight_scale.shape, dtype=torch.uint8, generator=gen_new)
        projection.weight_scale.copy_(updated)
        expected_scales.append(updated.transpose(0, 1).contiguous())
    executor.refresh_megamoe_shared_weights()

    assert executor.megamoe_shared_fusion().weights is weights
    assert pointers == tuple(operand.data_ptr() for operand in weights.operands())
    for scale, expected in zip((weights.w1_scale[0], weights.w2_scale[0]), expected_scales, strict=True):
        torch.testing.assert_close(scale.view(torch.uint8), expected, rtol=0, atol=0)
