# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method, moe_utils, routed_experts
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.moe_comm_method import ZercMoECommImpl
from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("available", [True, False])
def test_wheel_availability_is_lazy_and_cached(monkeypatch, available):
    load = MagicMock(return_value=object(), side_effect=None if available else ImportError("missing wheel"))
    monkeypatch.setattr(moe_utils, "load_zercmoe_ops", load)
    moe_utils.zercmoe_lib_available.cache_clear()
    try:
        load.assert_not_called()
        assert moe_utils.zercmoe_lib_available() is available
        assert moe_utils.zercmoe_lib_available() is available
        load.assert_called_once_with()
    finally:
        moe_utils.zercmoe_lib_available.cache_clear()


def test_weight_processing_preserves_decode_weights_and_packs_prefill_once(monkeypatch):
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4, dtype=torch.bfloat16), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 4, 4, dtype=torch.bfloat16), requires_grad=False)
    w13_nd = layer.w13_weight.detach().transpose(1, 2).contiguous()
    w2_nd = layer.w2_weight.detach().transpose(1, 2).contiguous()
    packed_w13, packed_w2 = w13_nd.flatten().clone(), w2_nd.flatten().clone()
    pack = MagicMock(return_value=(packed_w13, packed_w2))
    monkeypatch.setattr(moe_utils, "load_zercmoe_ops", lambda: SimpleNamespace(pack_weights=pack))
    monkeypatch.setattr(routed_experts, "use_cann_zercmoe", lambda config: True)
    monkeypatch.setattr(routed_experts, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(routed_experts, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    nz_cast = MagicMock(side_effect=AssertionError("ZercMoE must retain ND weights"))
    monkeypatch.setattr(routed_experts.torch_npu, "npu_format_cast", nz_cast)
    monkeypatch.setattr(routed_experts, "maybe_trans_nz", nz_cast)
    method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
    method.dynamic_eplb = False
    method.moe = SimpleNamespace(num_local_experts=2, hidden_dim=4, intermediate_size_per_partition=4)
    method._maybe_pad_weight = lambda weight: weight
    monkeypatch.setattr(
        AscendUnquantizedFusedMoEMethod.__mro__[2],
        "process_weights_after_loading",
        lambda self, layer: None,
        raising=False,
    )

    method.process_weights_after_loading(layer)

    pack.assert_called_once()
    torch.testing.assert_close(pack.call_args.args[0], w13_nd)
    torch.testing.assert_close(pack.call_args.args[1], w2_nd)
    decode_w13, decode_w2 = method.get_mlp_weights(layer)
    torch.testing.assert_close(decode_w13, w13_nd)
    torch.testing.assert_close(decode_w2, w2_nd)
    packed = method.get_zercmoe_weights(layer)
    assert packed.w1 is packed_w13
    assert packed.w2 is packed_w2
    assert packed.w1_scale is None and packed.w2_scale is None
    nz_cast.assert_not_called()


def _make_layer_contract():
    layer = SimpleNamespace(
        activation="silu",
        w13_weight=torch.empty(2, 16, 64, dtype=torch.bfloat16),
        w2_weight=torch.empty(2, 32, 16, dtype=torch.bfloat16),
    )
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_dim=16,
        intermediate_size_per_partition=32,
        activation="silu",
        has_bias=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
    )
    return layer, config


@pytest.mark.parametrize(
    "owner,field,value",
    [
        ("layer", "activation", "gelu"),
        ("config", "activation", "swigluoai"),
        ("layer", "swigluoai_uninterleave", True),
        ("config", "swiglu_limit", 7.0),
        ("layer", "swiglu_alpha", 1.702),
        ("config", "swiglu_beta", 1.0),
        ("config", "has_bias", True),
        ("layer", "w13_bias", torch.zeros(2, 64)),
        ("layer", "w2_bias", torch.zeros(2, 16)),
        ("layer", "apply_router_weight_on_input", True),
        ("config", "apply_router_weight_on_input", True),
        ("layer", "w13_weight", torch.empty(2, 16, 64, dtype=torch.float16)),
        ("layer", "w2_weight", torch.empty(2, 32, 16, dtype=torch.float32)),
        ("layer", "w13_weight", torch.empty(2, 64, 16, dtype=torch.bfloat16)),
        ("layer", "w2_weight", torch.empty(2, 16, 32, dtype=torch.bfloat16)),
    ],
)
def test_layer_contract_rejects_unsupported_features(owner, field, value):
    layer, config = _make_layer_contract()
    setattr(layer if owner == "layer" else config, field, value)

    with pytest.raises(ValueError, match="Disable additional_config.enable_zerc_moe"):
        moe_utils.validate_zercmoe_layer(layer, config)


def test_layer_contract_accepts_enum_activation_and_default_scalars():
    layer, config = _make_layer_contract()
    layer.activation = SimpleNamespace(value="silu")
    layer.swiglu_limit = 0.0
    config.swiglu_alpha = 1.0
    config.swiglu_beta = 0.0

    moe_utils.validate_zercmoe_layer(layer, config)


def _make_comm(monkeypatch, *, ipport=None, shmem_mb=None):
    comm = object.__new__(ZercMoECommImpl)
    comm._zercmoe = None
    comm.zerc_symm_buffer = None
    comm.moe_config = SimpleNamespace(
        num_experts=8,
        num_local_experts=2,
        experts_per_token=2,
        hidden_dim=16,
        intermediate_size_per_partition=32,
    )
    comm.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm.token_dispatcher.max_num_tokens_per_rank = 8
    comm.token_dispatcher.ep_world_size = 4
    comm.token_dispatcher.ep_rank_id = 1
    symm_buffer = SimpleNamespace(moe_expert_num=8, ep_world_size=4, ccl_buffer_size=4096)
    wheel = SimpleNamespace(get_symm_buffer=MagicMock(return_value=symm_buffer), mega_moe=MagicMock())
    load = MagicMock(return_value=wheel)
    monkeypatch.setattr(moe_utils, "load_zercmoe_ops", load)
    monkeypatch.setattr(
        moe_comm_method,
        "get_ascend_config",
        lambda: SimpleNamespace(zercmoe_ipport=ipport, zercmoe_shmem_mb=shmem_mb),
    )
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    return comm, wheel, load


def _make_input(num_tokens=2, *, w1=None, w2=None, activation="silu", quant_type=QuantType.NONE):
    return build_fused_experts_input(
        hidden_states=torch.zeros(num_tokens, 16, dtype=torch.bfloat16),
        topk_ids=torch.zeros(num_tokens, 2, dtype=torch.int64),
        topk_weights=torch.ones(num_tokens, 2, dtype=torch.bfloat16),
        w1=torch.empty(256, dtype=torch.bfloat16) if w1 is None else w1,
        w2=torch.empty(128, dtype=torch.bfloat16) if w2 is None else w2,
        quant_type=quant_type,
        dynamic_eplb=False,
        activation=activation,
    )


def test_fused_experts_calls_direct_op_without_expert_count_sync_and_reuses_heap(monkeypatch):
    comm, wheel, load = _make_comm(monkeypatch)
    output = torch.ones(2, 16, dtype=torch.bfloat16)
    op = MagicMock(return_value=(output, None))
    monkeypatch.setattr(torch.ops.npu, "zerc_moe", op, raising=False)
    inp = _make_input()
    weights = SimpleNamespace(w1=inp.weights.w1, w2=inp.weights.w2)
    quant_method = SimpleNamespace(get_zercmoe_weights=MagicMock(return_value=weights))

    result = comm.fused_experts(inp, quant_method=quant_method)
    comm.fused_experts(_make_input(num_tokens=4))

    assert result.routed_out is output
    assert result.before_dispatch_evt is not None
    assert result.expert_tokens is None
    quant_method.get_zercmoe_weights.assert_called_once_with(inp.layer)
    load.assert_called_once_with()
    wheel.get_symm_buffer.assert_called_once()
    assert wheel.get_symm_buffer.call_args.kwargs["num_max_tokens_per_rank"] == 8
    wheel.mega_moe.assert_not_called()
    first_call = op.call_args_list[0].args
    assert first_call[0] is inp.hidden_states
    assert first_call[1].dtype == torch.int32
    assert first_call[2].dtype == torch.float32
    assert first_call[3][0] is inp.weights.w1
    assert first_call[4][0] is inp.weights.w2
    assert first_call[5:] == (8, 4, 4096, 0, 0)


@pytest.mark.parametrize(
    "inp,error,match",
    [
        (_make_input(num_tokens=9), ValueError, "at most 8"),
        (_make_input(w1=torch.empty(2, 8)), AssertionError, "pre-packed zN weights"),
    ],
)
def test_invalid_input_fails_before_collective_bootstrap(monkeypatch, inp, error, match):
    comm, wheel, load = _make_comm(monkeypatch)

    with pytest.raises(error, match=match):
        comm.fused_experts(inp)

    load.assert_not_called()
    wheel.get_symm_buffer.assert_not_called()


@pytest.mark.parametrize(
    "inp",
    [_make_input(activation="gelu"), _make_input(quant_type=QuantType.W8A8)],
)
def test_fused_experts_defends_operator_contract(monkeypatch, inp):
    comm, _, load = _make_comm(monkeypatch)

    with pytest.raises(ValueError, match="Disable additional_config.enable_zerc_moe"):
        comm.fused_experts(inp)

    load.assert_not_called()


@pytest.mark.parametrize(
    "ipport,shmem_mb,expected_ipport,expected_mb",
    [(None, None, "tcp://127.0.0.1:28765", 0), ("tcp://10.0.0.1:29000", 512, "tcp://10.0.0.1:29000", 512)],
)
def test_shmem_uses_rank_invariant_config(monkeypatch, ipport, shmem_mb, expected_ipport, expected_mb):
    comm, wheel, load = _make_comm(monkeypatch, ipport=ipport, shmem_mb=shmem_mb)
    comm._ensure_zercmoe_loaded()

    comm._init_zerc_symm_buffer()

    assert moe_comm_method.os.environ["RANK"] == "1"
    assert moe_comm_method.os.environ["WORLD_SIZE"] == "4"
    wheel.get_symm_buffer.assert_called_once_with(
        moe_expert_num=8,
        ep_world_size=4,
        num_max_tokens_per_rank=8,
        num_topk=2,
        hidden=16,
        intermediate_hidden=32,
        ipport=expected_ipport,
        device_id=torch.npu.current_device(),
        mem_size_mb=expected_mb,
    )
    load.assert_called_once_with()


def test_shmem_overrides_inherited_launcher_rank(monkeypatch):
    comm, wheel, _ = _make_comm(monkeypatch)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    observed_ranks = []
    wheel.get_symm_buffer.side_effect = lambda **kwargs: observed_ranks.append(
        (moe_comm_method.os.environ["RANK"], moe_comm_method.os.environ["WORLD_SIZE"])
    )
    comm._ensure_zercmoe_loaded()

    comm._init_zerc_symm_buffer()

    assert observed_ranks == [("1", "4")]


@pytest.mark.parametrize("field", ["experts_per_token", "hidden_dim", "intermediate_size_per_partition"])
def test_comm_cache_keeps_different_shmem_shapes_separate(field):
    config1 = SimpleNamespace(
        num_experts=8, num_local_experts=2, experts_per_token=2, hidden_dim=16, intermediate_size_per_partition=32
    )
    config2 = SimpleNamespace(**vars(config1))
    setattr(config2, field, getattr(config2, field) * 2)

    assert moe_comm_method._moe_config_key(MoECommType.ZERC_MOE, config1) != moe_comm_method._moe_config_key(
        MoECommType.ZERC_MOE, config2
    )


def test_comm_construction_does_not_load_or_bootstrap_shmem(monkeypatch):
    load = MagicMock()
    monkeypatch.setattr(moe_utils, "load_zercmoe_ops", load)
    monkeypatch.setattr(ZercMoECommImpl, "_get_token_dispatcher", lambda self: object())
    monkeypatch.setattr(ZercMoECommImpl, "_get_prepare_finalize", lambda self: object())

    comm = ZercMoECommImpl(SimpleNamespace())

    assert comm.zerc_symm_buffer is None
    load.assert_not_called()
