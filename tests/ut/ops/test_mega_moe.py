from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.mega_moe import MegaMoEBackend
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMegaMoE
from vllm_ascend.quantization.quant_type import QuantType


def _make_moe_config(*, activation=MoEActivation.SITU):
    return SimpleNamespace(
        num_experts=8,
        num_local_experts=2,
        experts_per_token=2,
        activation=activation,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        swiglu_limit=None,
    )


def _make_fused_input(
    *,
    activation=MoEActivation.SITU,
    group_size=32,
    act_quant_type=torch.float8_e4m3fn,
    weight_quant_type=torch_npu.float4_e2m1fn_x2,
    dynamic_eplb=False,
    global_redundant_expert_num=0,
    lora_context=None,
):
    return build_fused_experts_input(
        hidden_states=torch.randn(4, 128),
        topk_weights=torch.randn(4, 2, dtype=torch.bfloat16),
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int64),
        w1=torch.randint(0, 255, (2, 128, 64), dtype=torch.uint8),
        w2=torch.randint(0, 255, (2, 128, 32), dtype=torch.uint8),
        quant_type=QuantType.W4A8MXFP,
        dynamic_eplb=dynamic_eplb,
        global_redundant_expert_num=global_redundant_expert_num,
        activation=activation,
        mxfp_act_quant_type=act_quant_type,
        mxfp_weight_quant_type=weight_quant_type,
        mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_group_size=group_size,
        w1_scale=torch.ones(2, 128, 2, 2, dtype=torch.uint8),
        w2_scale=torch.ones(2, 128, 1, 2, dtype=torch.uint8),
        lora_context=lora_context,
    )


def _run_backend(mega_moe, *, activation=MoEActivation.SITU):
    group = SimpleNamespace(device_group=object())
    get_symm_buffer = MagicMock(return_value=object())
    vllm_config = SimpleNamespace()
    with (
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe._get_mega_moe_ops",
            return_value=(get_symm_buffer, mega_moe),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_current_vllm_config",
            return_value=vllm_config,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
        patch(
            "vllm_ascend.ops.fused_moe.mega_moe.get_mega_moe_group",
            return_value=group,
        ),
    ):
        MegaMoEBackend(_make_moe_config(activation=activation)).fused_experts(_make_fused_input(activation=activation))
    return get_symm_buffer


def test_kimi_situ_uses_cann_activation_contract_and_projected_buffer_width():
    mega_moe = MagicMock(return_value=(torch.randn(4, 128), None))

    get_symm_buffer = _run_backend(mega_moe)

    assert get_symm_buffer.call_args.kwargs["intermediate_hidden"] == 128
    assert "max_recv_token_num" not in get_symm_buffer.call_args.kwargs
    kwargs = mega_moe.call_args.kwargs
    assert kwargs["activation"] == "situglu"
    assert kwargs["activation_params"] == {"beta": 4.0, "linear_beta": 25.0}
    assert kwargs["activation_clamp"] is None
    assert kwargs["topk_ids"].dtype == torch.int32
    assert kwargs["topk_ids"].is_contiguous()
    assert kwargs["topk_weights"].dtype == torch.float32
    assert kwargs["topk_weights"].is_contiguous()


def test_kimi_k3_latent_layout_uses_6144_buffer_width():
    fused_input = build_fused_experts_input(
        hidden_states=torch.empty(1, 3584, device="meta"),
        topk_weights=torch.empty(1, 16, device="meta"),
        topk_ids=torch.empty(1, 16, dtype=torch.int64, device="meta"),
        w1=torch.empty(1, 6144, 1792, dtype=torch.uint8, device="meta"),
        w2=torch.empty(1, 3584, 1536, dtype=torch.uint8, device="meta"),
        quant_type=QuantType.W4A8MXFP,
        dynamic_eplb=False,
        activation=MoEActivation.SITU,
        mxfp_act_quant_type=torch.float8_e4m3fn,
        mxfp_weight_quant_type=torch_npu.float4_e2m1fn_x2,
        mxfp_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        mxfp_group_size=32,
        w1_scale=torch.empty(1, 6144, 56, 2, dtype=torch.uint8, device="meta"),
        w2_scale=torch.empty(1, 3584, 48, 2, dtype=torch.uint8, device="meta"),
    )
    backend = MegaMoEBackend(SimpleNamespace(num_experts=896, num_local_experts=1, experts_per_token=16))

    projected_hidden = backend._validate_stacked_mxfp_layout(
        fused_input,
        [fused_input.weights.w1],
        [fused_input.weights.w2],
        [fused_input.weights.w1_scale],
        [fused_input.weights.w2_scale],
    )

    assert projected_hidden == 6144


def test_swiglu_keeps_existing_contract_without_situ_parameters():
    mega_moe = MagicMock(return_value=(torch.randn(4, 128), None))

    _run_backend(mega_moe, activation=MoEActivation.SILU)

    assert mega_moe.call_args.kwargs["activation"] == "swiglu"
    assert "activation_params" not in mega_moe.call_args.kwargs


@pytest.mark.parametrize("missing_attr", ["activation_situ_beta", "activation_situ_linear_beta"])
def test_kimi_situ_requires_both_activation_parameters(missing_attr):
    backend = MegaMoEBackend(_make_moe_config())
    setattr(backend.moe_config, missing_attr, None)

    with pytest.raises(ValueError, match="requires activation_situ_beta"):
        backend._resolve_activation(MoEActivation.SITU)


@pytest.mark.parametrize(
    ("input_kwargs", "message"),
    [
        ({"group_size": 64}, "group_size=32"),
        ({"act_quant_type": torch.float16}, "FP8 E4M3"),
        ({"weight_quant_type": torch.float8_e4m3fn}, "FP4 E2M1"),
        ({"dynamic_eplb": True}, "dynamic EPLB"),
        ({"global_redundant_expert_num": 1}, "redundant physical experts"),
        ({"lora_context": object()}, "MoE LoRA"),
    ],
)
def test_mega_moe_rejects_unsupported_runtime_contract(input_kwargs, message):
    backend = MegaMoEBackend(_make_moe_config())

    with pytest.raises(RuntimeError, match=message):
        backend.fused_experts(_make_fused_input(**input_kwargs))


def test_kimi_situ_rejects_old_cann_signature():
    mega_moe = MagicMock(side_effect=TypeError("unexpected keyword argument 'activation_params'"))

    with pytest.raises(RuntimeError, match="Kimi K3 SiTU activation_params"):
        _run_backend(mega_moe)


def test_mega_moe_prepare_aligns_dp_tokens_and_restores_local_batch():
    prepare_finalize = object.__new__(PrepareAndFinalizeWithMegaMoE)
    hidden_states = torch.randn(4, 128)
    router_logits = torch.randn(4, 8)
    input_ids = torch.tensor([1, 2, 3, 4])

    with (
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX",
            SimpleNamespace(max_tokens_across_dp=6),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_current_vllm_config",
            return_value=SimpleNamespace(),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.prepare_finalize.get_a5_mega_moe_buffer_tokens_per_rank",
            return_value=8,
        ),
    ):
        prepared = prepare_finalize.prepare(hidden_states, router_logits)

    assert prepared.hidden_states.shape == (6, 128)
    assert prepared.router_logits.shape == (6, 8)
    torch.testing.assert_close(prepare_finalize.pad_and_split_input_ids(input_ids), torch.tensor([1, 2, 3, 4, 0, 0]))

    output = torch.randn(6, 128)
    torch.testing.assert_close(
        prepare_finalize.finalize(output, reduce_results=False),
        output[:4],
    )
