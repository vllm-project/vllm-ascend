from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.device.mxfp_compat import FLOAT4_E2M1FN_X2_DTYPE
from vllm_ascend.ops.activation import SituActivationConfig
from vllm_ascend.ops.fused_moe.mega_moe import MegaMoEBackend, _view_mxfp_scales_as_e8m0
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMegaMoE
from vllm_ascend.quantization.quant_type import QuantType


def _make_moe_config(*, num_experts=8, num_local_experts=2, top_k=2):
    moe_config = MagicMock(spec=FusedMoEConfig)
    moe_config.num_experts = num_experts
    moe_config.num_local_experts = num_local_experts
    moe_config.experts_per_token = top_k
    return moe_config


def _make_ascend_config():
    return SimpleNamespace(
        mega_moe_max_tokens=32,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        ),
    )


def _make_group(device_group=None):
    return SimpleNamespace(
        device_group=object() if device_group is None else device_group,
        world_size=4,
        ranks=[0, 1, 2, 3],
    )


def _make_fused_input(
    *,
    activation=None,
    group_size=32,
    quant_type=QuantType.W4A8MXFP,
    dynamic_eplb=False,
    global_redundant_expert_num=0,
    lora_context=None,
    w1=None,
):
    return build_fused_experts_input(
        hidden_states=torch.randn(4, 128),
        topk_weights=torch.randn(4, 2),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int64),
        w1=(torch.randint(0, 255, (2, 128, 64), dtype=torch.uint8) if w1 is None else w1),
        w2=torch.randint(0, 255, (2, 128, 32), dtype=torch.uint8),
        quant_type=quant_type,
        dynamic_eplb=dynamic_eplb,
        global_redundant_expert_num=global_redundant_expert_num,
        activation=activation or SituActivationConfig(beta=4.0, linear_beta=25.0),
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=FLOAT4_E2M1FN_X2_DTYPE,
        mxfp_scale_dtype=torch.uint8,
        mxfp_per_token_scale_dtype=torch.uint8,
        mxfp_group_size=group_size,
        w1_scale=torch.ones(2, 128, 2, 2, dtype=torch.uint8),
        w2_scale=torch.ones(2, 128, 1, 2, dtype=torch.uint8),
        lora_context=lora_context,
    )


def _make_backend(*, get_symm_buffer=None, mega_moe=None, moe_config=None):
    get_symm_buffer = get_symm_buffer or MagicMock(return_value=object())
    mega_moe = mega_moe or MagicMock(return_value=(torch.randn(4, 128), torch.tensor([1, 3], dtype=torch.int32)))
    return MegaMoEBackend(
        moe_config or _make_moe_config(),
        ops=(get_symm_buffer, mega_moe),
    )


def test_kimi_situ_operator_contract_and_buffer_layout():
    group = _make_group()
    sym_buffer = object()
    get_symm_buffer = MagicMock(return_value=sym_buffer)
    mega_moe = MagicMock(return_value=(torch.randn(4, 128), torch.tensor([1, 3], dtype=torch.int32)))
    backend = _make_backend(get_symm_buffer=get_symm_buffer, mega_moe=mega_moe)
    fused_input = _make_fused_input()

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group", return_value=group),
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config", return_value=_make_ascend_config()),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe._EXTRA_CTX", SimpleNamespace(capturing=False)),
    ):
        backend.fused_experts(fused_input)

    get_symm_buffer.assert_called_once_with(
        group.device_group,
        num_experts=8,
        num_max_tokens_per_rank=8,
        num_topk=2,
        hidden=128,
        intermediate_hidden=128,
        dispatch_quant_mode=4,
        dispatch_quant_out_dtype=torch.float8_e4m3fn,
    )
    kwargs = mega_moe.call_args.kwargs
    assert kwargs["sym_buffer"] is sym_buffer
    assert kwargs["activation"] == "situglu"
    assert kwargs["activation_params"] == {"beta": 4.0, "linear_beta": 25.0}
    assert kwargs["activation_clamp"] is None
    assert kwargs["weight1_type"] == FLOAT4_E2M1FN_X2_DTYPE
    assert kwargs["topk_ids"].dtype == torch.int32
    assert kwargs["topk_ids"].is_contiguous()
    assert kwargs["l1_weights_sf"][0].dtype == torch.float8_e8m0fnu
    assert kwargs["l2_weights_sf"][0].dtype == torch.float8_e8m0fnu


def test_situ_without_linear_beta_is_forwarded_as_none():
    assert MegaMoEBackend._resolve_activation(SituActivationConfig(beta=4.0), 7.0) == (
        "situglu",
        {"beta": 4.0, "linear_beta": None},
        None,
    )


def test_swiglu_does_not_pass_activation_params():
    group = _make_group()
    mega_moe = MagicMock(return_value=(torch.randn(4, 128), None))
    backend = _make_backend(mega_moe=mega_moe)

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group", return_value=group),
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config", return_value=_make_ascend_config()),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe._EXTRA_CTX", SimpleNamespace(capturing=False)),
    ):
        backend.fused_experts(_make_fused_input(activation="silu"))

    assert mega_moe.call_args.kwargs["activation"] == "swiglu"
    assert "activation_params" not in mega_moe.call_args.kwargs


def test_uint8_scales_are_reinterpreted_without_copy():
    scale = torch.arange(64, dtype=torch.uint8).reshape(2, 16, 1, 2)

    normalized = _view_mxfp_scales_as_e8m0([scale], "w1_scale")[0]

    assert normalized.dtype == torch.float8_e8m0fnu
    assert normalized.shape == scale.shape
    assert normalized.stride() == scale.stride()
    assert normalized.data_ptr() == scale.data_ptr()


def test_kimi_k3_latent_dimensions_produce_6144_intermediate_hidden():
    backend = _make_backend(moe_config=_make_moe_config(num_experts=896, num_local_experts=1, top_k=16))
    fused_input = build_fused_experts_input(
        hidden_states=torch.empty(1, 3584, device="meta"),
        topk_weights=torch.empty(1, 16, device="meta"),
        topk_ids=torch.empty(1, 16, dtype=torch.int64, device="meta"),
        w1=torch.empty(1, 6144, 1792, dtype=torch.uint8, device="meta"),
        w2=torch.empty(1, 3584, 1536, dtype=torch.uint8, device="meta"),
        quant_type=QuantType.W4A8MXFP,
        dynamic_eplb=False,
        activation=SituActivationConfig(beta=4.0, linear_beta=25.0),
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=FLOAT4_E2M1FN_X2_DTYPE,
        mxfp_scale_dtype=torch.uint8,
        mxfp_per_token_scale_dtype=torch.uint8,
        mxfp_group_size=32,
        w1_scale=torch.empty(1, 6144, 56, 2, dtype=torch.float8_e8m0fnu, device="meta"),
        w2_scale=torch.empty(1, 3584, 48, 2, dtype=torch.float8_e8m0fnu, device="meta"),
    )

    projected_hidden = backend._validate_stacked_mxfp_layout(
        fused_input,
        [fused_input.weights.w1],
        [fused_input.weights.w2],
        [fused_input.weights.w1_scale],
        [fused_input.weights.w2_scale],
    )

    assert projected_hidden == 6144


@pytest.mark.parametrize(
    ("input_kwargs", "message"),
    [
        ({"group_size": 64}, "group_size=32"),
        ({"quant_type": QuantType.W8A8MXFP}, "only W4A8MXFP"),
        ({"dynamic_eplb": True}, "dynamic EPLB"),
        ({"global_redundant_expert_num": 1}, "redundant physical experts"),
        ({"lora_context": object()}, "MoE LoRA"),
    ],
)
def test_backend_rejects_unsupported_runtime_contracts(input_kwargs, message):
    backend = _make_backend()

    with pytest.raises(RuntimeError, match=message):
        backend.fused_experts(_make_fused_input(**input_kwargs))


def test_backend_rejects_gmm_oriented_noncontiguous_weight():
    gmm_weight = torch.randint(0, 255, (2, 64, 128), dtype=torch.uint8).transpose(1, 2)
    backend = _make_backend()

    with pytest.raises(ValueError, match="requires contiguous w1"):
        backend.fused_experts(_make_fused_input(w1=gmm_weight))


def test_backend_rejects_nonpacked_weight_storage():
    backend = _make_backend()

    with pytest.raises(ValueError, match="one-byte storage"):
        backend.fused_experts(_make_fused_input(w1=torch.randn(2, 128, 64)))


def test_process_wide_buffer_is_reused_across_backends():
    group = _make_group()
    sym_buffer = object()
    get_symm_buffer = MagicMock(return_value=sym_buffer)
    mega_moe = MagicMock(return_value=(torch.randn(4, 128), None))

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group", return_value=group),
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config", return_value=_make_ascend_config()),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe._EXTRA_CTX", SimpleNamespace(capturing=False)),
    ):
        _make_backend(get_symm_buffer=get_symm_buffer, mega_moe=mega_moe).fused_experts(_make_fused_input())
        _make_backend(get_symm_buffer=get_symm_buffer, mega_moe=mega_moe).fused_experts(_make_fused_input())

    get_symm_buffer.assert_called_once()
    assert all(call.kwargs["sym_buffer"] is sym_buffer for call in mega_moe.call_args_list)


def test_buffer_key_change_fails_instead_of_reallocating():
    group = _make_group()
    get_symm_buffer = MagicMock(return_value=object())
    first = _make_backend(get_symm_buffer=get_symm_buffer)
    second = _make_backend(get_symm_buffer=get_symm_buffer, moe_config=_make_moe_config(num_experts=16))
    fused_input = _make_fused_input()

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group", return_value=group),
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config", return_value=_make_ascend_config()),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe._EXTRA_CTX", SimpleNamespace(capturing=False)),
    ):
        first.fused_experts(fused_input)
        with pytest.raises(RuntimeError, match="process-wide and immutable"):
            second.fused_experts(fused_input)

    get_symm_buffer.assert_called_once()


def test_buffer_creation_is_rejected_during_aclgraph_capture():
    backend = _make_backend()

    with (
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group", return_value=_make_group()),
        patch("vllm_ascend.ops.fused_moe.mega_moe.get_ascend_config", return_value=_make_ascend_config()),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch("vllm_ascend.ops.fused_moe.mega_moe._EXTRA_CTX", SimpleNamespace(capturing=True)),
        pytest.raises(RuntimeError, match="before ACLGraph capture"),
    ):
        backend.fused_experts(_make_fused_input())


def test_prepare_pads_only_to_active_dp_max_and_finalize_unpads():
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)
    hidden_states = torch.randn(4, 128)
    router_logits = torch.randn(4, 8)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX",
            SimpleNamespace(max_tokens_across_dp=6),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_ascend_config",
            return_value=_make_ascend_config(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
    ):
        prepared = prepare_finalize.prepare(hidden_states, router_logits)

    assert prepared.hidden_states.shape == (6, 128)
    assert prepared.router_logits.shape == (6, 8)
    assert torch.count_nonzero(prepared.hidden_states[4:]) == 0
    output = torch.randn(6, 128)
    torch.testing.assert_close(prepare_finalize.finalize(output, False), output[:4])


def test_prepare_rejects_active_dp_max_over_buffer_capacity():
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX",
            SimpleNamespace(max_tokens_across_dp=9),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_ascend_config",
            return_value=_make_ascend_config(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        pytest.raises(ValueError, match="exceeds the symmetric buffer token capacity"),
    ):
        prepare_finalize.prepare(torch.randn(4, 128), torch.randn(4, 8))
