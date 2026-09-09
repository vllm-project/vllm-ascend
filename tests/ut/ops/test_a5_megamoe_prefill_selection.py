from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend import ascend_config as config_module
from vllm_ascend import ascend_forward_context as afc
from vllm_ascend import utils
from vllm_ascend.ops.fused_moe import moe_comm_method as comm
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.mega_moe_adapter import (
    CannMegaMoeActivation,
    CannMegaMoeLayerCapability,
)
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize(
    ("tokens", "supported", "switch", "enabled", "expected"),
    [
        (1, True, 1, True, afc.MoECommType.FUSED_MC2),
        (8, True, 1, True, afc.MoECommType.FUSED_MC2),
        (8192, True, 1, True, afc.MoECommType.FUSED_MC2),
        (1, False, 1, True, afc.MoECommType.MC2),
        (8192, False, 1, True, afc.MoECommType.ALLTOALL),
        (8192, True, 0, False, afc.MoECommType.ALLTOALL),
        (8192, True, 1, False, afc.MoECommType.ALLTOALL),
        (8192, True, 0, True, afc.MoECommType.ALLTOALL),
    ],
)
def test_a5_capability_selection_without_capacity_gate(monkeypatch, tokens, supported, switch, enabled, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(top_k_experts=1)),
        parallel_config=SimpleNamespace(enable_expert_parallel=True, world_size_across_dp=8),
        lora_config=None,
    )
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 8)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A5)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=switch))
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: enabled)
    assert afc.select_moe_comm_method(tokens, config, cann_mega_moe_supported=supported) == expected


@pytest.mark.parametrize("counts", [(2, 0), None, (0, 1), (1, 1), (0, 0)])
def test_a5_dp_ranks_keep_same_comm_method_with_different_metadata(monkeypatch, counts):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(top_k_experts=8)),
        parallel_config=SimpleNamespace(enable_expert_parallel=True, world_size_across_dp=32),
        lora_config=None,
    )
    context = SimpleNamespace(dp_metadata=None)
    capability = CannMegaMoeLayerCapability(True, "", QuantType.W4A8MXFP)
    monkeypatch.setattr(afc, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(afc, "set_forward_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(afc, "get_forward_context", lambda: context)
    monkeypatch.setattr(afc, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(afc, "get_dp_group", lambda: SimpleNamespace(world_size=4))
    monkeypatch.setattr(afc, "has_layer_idx", lambda _: False)
    monkeypatch.setattr(afc, "get_mc2_mask", lambda: None)
    monkeypatch.setattr(afc, "get_model_cann_mega_moe_capability", lambda _: capability)
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 128)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A5)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=32))
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    monkeypatch.setattr(afc, "is_mega_moe_supported", lambda: True)
    monkeypatch.setattr(comm, "get_moe_comm_method", lambda kind: kind)
    metadata = None if counts is None else {"layer": SimpleNamespace(num_prefills=counts[0], num_decodes=counts[1])}
    with afc.set_ascend_forward_context(metadata, config, num_tokens=16128):
        assert context.moe_comm_type == afc.MoECommType.FUSED_MC2


def _request():
    return build_fused_experts_input(
        hidden_states=torch.ones(2, 4),
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.tensor([[0], [1]]),
        w1=torch.ones(2, 4, 8).transpose(-2, -1),
        w2=torch.ones(2, 4, 8).transpose(-2, -1),
        w1_scale=torch.ones(2, 4, 2, 2).transpose(-3, -2),
        w2_scale=torch.ones(2, 4, 2, 2).transpose(-3, -2),
        quant_type=QuantType.W4A8MXFP,
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch.float4_e2m1fn_x2,
        mxfp_scale_dtype=torch.float8_e8m0fnu,
        mxfp_per_token_scale_dtype=torch.float8_e8m0fnu,
        dynamic_eplb=False,
        activation=MoEActivation.SITU,
    )


def _implementation():
    impl = comm.FusedMC2CommImpl.__new__(comm.FusedMC2CommImpl)
    impl.token_dispatcher = MagicMock(spec=comm.TokenDispatcherWithMC2)
    impl.token_dispatcher.max_num_tokens_per_rank = 512
    impl.token_dispatcher.global_bs = 0
    impl.moe_config = SimpleNamespace(activation_situ_beta=4.0, activation_situ_linear_beta=25.0)
    impl.cann_mega_moe_capability = CannMegaMoeLayerCapability(
        True, "", QuantType.W4A8MXFP, CannMegaMoeActivation("situglu", alpha=25.0, beta=4.0)
    )
    impl.mega_moe = MagicMock(return_value=(torch.ones(2, 4), torch.ones(2, dtype=torch.int32)))
    impl.mega_moe_symm_buffer = SimpleNamespace(dispatch_quant_mode=4, dispatch_quant_out_dtype=torch.float8_e4m3fn)
    impl.swiglu_limit = 0.0
    return impl


@pytest.mark.parametrize("chunk_size", [4096, 8192, 16384])
@pytest.mark.parametrize("decode_only", [False, True])
def test_a5_receive_capacity_is_fixed_on_all_dp_ranks(monkeypatch, chunk_size, decode_only):
    monkeypatch.setattr(comm, "get_mc2_group", lambda: SimpleNamespace(device_group=object()))
    for ep_rank in (0, 8, 16, 24):
        impl = _implementation()
        impl.token_dispatcher.a5_need_extra_args = True
        impl.token_dispatcher.ep_world_size = 32
        impl.token_dispatcher.ep_rank_id = ep_rank
        impl.token_dispatcher.max_num_tokens_per_rank = chunk_size // 8
        impl.moe_config = SimpleNamespace(
            experts_per_token=16,
            num_experts=896,
            hidden_dim=3584,
            intermediate_size_per_partition=3072,
        )
        impl.get_symm_buffer_for_mega_moe = MagicMock()
        result = impl._init_mega_moe_symm_buffer(is_decode_only_node=decode_only)
        assert result is impl.get_symm_buffer_for_mega_moe.return_value
        assert impl.get_symm_buffer_for_mega_moe.call_args.kwargs["max_recv_token_num"] == 65536


@pytest.mark.parametrize("decode_only, expected", [(False, 65536), (True, 1048576)])
def test_non_a5_receive_capacity_policy_is_preserved(monkeypatch, decode_only, expected):
    monkeypatch.setattr(comm, "get_mc2_group", lambda: SimpleNamespace(device_group=object()))
    monkeypatch.setattr(comm, "get_ascend_config", lambda: SimpleNamespace(mega_moe_max_tokens=65536))
    impl = _implementation()
    impl.token_dispatcher.a5_need_extra_args = False
    impl.token_dispatcher.ep_world_size = 32
    impl.token_dispatcher.max_num_tokens_per_rank = 2048
    impl.moe_config = SimpleNamespace(
        experts_per_token=16,
        num_experts=896,
        hidden_dim=3584,
        intermediate_size_per_partition=3072,
    )
    impl.get_symm_buffer_for_mega_moe = MagicMock()
    impl._init_mega_moe_symm_buffer(is_decode_only_node=decode_only)
    assert impl.get_symm_buffer_for_mega_moe.call_args.kwargs["max_recv_token_num"] == expected


def test_fused_mc2_routes_supported_situ_to_mega_moe():
    impl = _implementation()
    request = _request()
    with (
        patch.object(comm, "get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=1)),
        patch.object(comm, "is_mega_moe_supported", return_value=True),
        patch.object(comm, "_EXTRA_CTX", SimpleNamespace(is_decode_only_node=False)),
    ):
        result = impl.fused_experts(request)
    impl.mega_moe.assert_called_once()
    args, kwargs = impl.mega_moe.call_args
    assert torch.equal(args[1], request.topk_ids.to(torch.int32))
    assert kwargs["activation"] == "situglu"
    assert kwargs["activation_params"] == {"beta": 4.0, "linear_beta": 25.0}
    assert args[3][0].data_ptr() == request.weights.w1.data_ptr()
    assert kwargs["l1_weights_sf"][0].data_ptr() == request.weights.w1_scale.data_ptr()
    assert args[3][0].is_contiguous()
    assert kwargs["l1_weights_sf"][0].is_contiguous()
    assert result.routed_out is impl.mega_moe.return_value[0]


@pytest.mark.parametrize("value, normalized, enabled", [(0, 0, False), (1, 1, False), (2, 1, True)])
def test_fused_mc2_input_controls_mega_moe(monkeypatch, value, normalized, enabled):
    monkeypatch.setattr(config_module, "_MEGA_MOE_SUPPORTED", None)
    monkeypatch.setattr(config_module.importlib.util, "find_spec", lambda _: object())
    config = SimpleNamespace(weight_nz_mode=1, enable_mc2_hierarchy_comm=False, enable_fused_mc2=value)
    config_module.AscendConfig._validate_user_input_ranges(config)
    assert config.enable_fused_mc2 == normalized
    assert config_module.is_mega_moe_supported() is enabled


@pytest.mark.parametrize("switch, enabled, loads", [(0, False, False), (1, False, False), (1, True, True)])
def test_fused_mc2_load_respects_mega_moe_switch(monkeypatch, switch, enabled, loads):
    def initialize_base(self, config):
        self.moe_config = config
        self.token_dispatcher = SimpleNamespace(a5_need_extra_args=True)

    monkeypatch.setattr(comm.MoECommMethod, "__init__", initialize_base)
    monkeypatch.setattr(comm, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=switch))
    monkeypatch.setattr(comm, "is_mega_moe_supported", lambda: enabled)
    monkeypatch.setattr(comm.torch, "zeros", lambda *args, **kwargs: None)
    loader = MagicMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(comm.moe_utils, "load_cann_mega_moe_ops", loader)
    config = SimpleNamespace(num_local_experts=2, swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None)
    capability = CannMegaMoeLayerCapability(True, "", QuantType.W4A8MXFP)
    instance = comm.FusedMC2CommImpl(config, cann_mega_moe_capability=capability)
    assert loader.called is loads
    assert (instance.mega_moe is not None) is loads


@pytest.mark.parametrize("switch, enabled, skip", [(0, False, True), (1, False, True), (1, True, False)])
def test_disabled_capability_does_not_force_dp_allreduce(monkeypatch, switch, enabled, skip):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(separate_routine=lambda: False)),
    )
    ascend = SimpleNamespace(
        enable_fused_mc2=switch,
        get_mc2_comm_alg=lambda: "",
        scheduler_config=SimpleNamespace(recompute_scheduler_enable=True),
    )
    monkeypatch.setattr(utils, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr(utils, "is_mega_moe_supported", lambda: enabled)
    monkeypatch.setattr(utils, "is_moe_model", lambda _: True)
    monkeypatch.setattr(utils, "get_potential_max_tokens", lambda: 8)
    monkeypatch.setattr(afc, "use_cann_megamoe", lambda _: False)
    monkeypatch.setattr(afc, "select_moe_comm_method", lambda *args, **kwargs: afc.MoECommType.MC2)
    assert utils.should_skip_allreduce_across_dp_group(config, cann_mega_moe_supported=True) is skip


@pytest.mark.parametrize("mask", [[True, True], [True, False], [False, False]])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8])
def test_a5_mxfp_padding_does_not_route_or_propagate_stale_output(mask, dtype):
    impl = _implementation()
    request = _request()
    valid = torch.tensor(mask)
    request = replace(request, routing=replace(request.routing, mc2_mask=valid.to(dtype)))
    original_ids = request.topk_ids.clone()
    impl.mega_moe.return_value[0][~valid] = float("nan")
    with (
        patch.object(comm, "get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=1)),
        patch.object(comm, "is_mega_moe_supported", return_value=True),
        patch.object(comm, "_EXTRA_CTX", SimpleNamespace(is_decode_only_node=False)),
    ):
        result = impl.fused_experts(request)
    routed_ids = impl.mega_moe.call_args.args[1]
    assert torch.equal(routed_ids[valid], original_ids[valid].int())
    assert (routed_ids[~valid] == -1).all()
    assert torch.equal(request.topk_ids, original_ids)
    assert (result.routed_out[~valid] == 0).all()
    assert (result.routed_out[valid] == 1).all()
    assert impl.mega_moe.call_args.kwargs["x_active_mask"] is None
