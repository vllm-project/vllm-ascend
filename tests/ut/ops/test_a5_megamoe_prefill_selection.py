from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend import ascend_forward_context as afc
from vllm_ascend.ops.fused_moe import moe_comm_method as comm
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.mega_moe_adapter import (
    CannMegaMoeActivation,
    CannMegaMoeLayerCapability,
)
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize(
    ("tokens", "pure_prefill", "supported", "expected"),
    [
        (1, True, True, afc.MoECommType.FUSED_MC2),
        (8192, True, True, afc.MoECommType.FUSED_MC2),
        (1, False, True, afc.MoECommType.MC2),
        (8, False, True, afc.MoECommType.MC2),
        (8192, False, True, afc.MoECommType.ALLTOALL),
        (1, True, False, afc.MoECommType.MC2),
    ],
)
def test_a5_pure_prefill_only_without_capacity_gate(monkeypatch, tokens, pure_prefill, supported, expected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(top_k_experts=1)),
        parallel_config=SimpleNamespace(enable_expert_parallel=True, world_size_across_dp=8),
        lora_config=None,
    )
    monkeypatch.setattr(afc, "is_moe_model", lambda _: True)
    monkeypatch.setattr(afc, "get_mc2_tokens_capacity", lambda: 8)
    monkeypatch.setattr(afc, "get_ascend_device_type", lambda: afc.AscendDeviceType.A5)
    monkeypatch.setattr(afc, "get_ep_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(afc, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    assert (
        afc.select_moe_comm_method(tokens, config, is_pure_prefill=pure_prefill, cann_mega_moe_supported=supported)
        == expected
    )


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
