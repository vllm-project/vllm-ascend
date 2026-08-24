from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ascend_forward_context import _MEGA_MOE_TOKENS_PER_RANK_LIMIT
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.methods.base import QuantType


def _make_comm_impl():
    comm_impl = object.__new__(FusedMC2CommImpl)
    comm_impl.token_dispatcher = object.__new__(TokenDispatcherWithMC2)
    comm_impl.token_dispatcher.global_bs = 1
    comm_impl.mega_moe_symm_buffer = SimpleNamespace(
        dispatch_quant_mode=None,
        dispatch_quant_out_dtype=None,
    )
    comm_impl.mega_moe = MagicMock()
    comm_impl.swiglu_limit = 7.0
    comm_impl.swiglu_alpha = 1.5
    comm_impl.swiglu_beta = 0.25
    return comm_impl


def _make_fused_experts_input(num_tokens, mc2_mask=None):
    return SimpleNamespace(
        hidden_states=torch.empty(num_tokens, 4),
        topk_ids=torch.zeros(num_tokens, 1, dtype=torch.int64),
        topk_weights=torch.ones(num_tokens, 1),
        activation="swigluoai_uninterleave",
        weights=SimpleNamespace(
            w1=[torch.empty(4, 8, dtype=torch.int8)],
            w2=[torch.empty(4, 4, dtype=torch.int8)],
            w1_scale=[torch.empty(8, dtype=torch.int64)],
            w2_scale=[torch.empty(4, dtype=torch.int64)],
            w1_scale_bias=None,
            w2_scale_bias=None,
        ),
        routing=SimpleNamespace(mc2_mask=mc2_mask),
        quant=SimpleNamespace(quant_type=QuantType.W8A8),
    )


def test_cann_mega_moe_receives_swigluoai_parameters():
    comm_impl = _make_comm_impl()
    comm_impl.mega_moe.return_value = (torch.empty(1, 4), torch.empty(2, dtype=torch.int32))

    comm_impl._apply_cann_mega_moe(_make_fused_experts_input(1))

    call_kwargs = comm_impl.mega_moe.call_args.kwargs
    assert call_kwargs["activation"] == "swigluoai"
    assert call_kwargs["activation_clamp"] == 7.0
    assert call_kwargs["activation_params"] == {"alpha": 1.5, "beta": 0.25}


def test_cann_mega_moe_symm_buffer_uses_chunk_capacity():
    comm_impl = _make_comm_impl()
    comm_impl.token_dispatcher.ep_world_size = 2
    comm_impl.token_dispatcher.ep_rank_id = 0
    comm_impl.token_dispatcher.max_num_tokens_per_rank = 40
    comm_impl.moe_config = SimpleNamespace(
        experts_per_token=2,
        num_experts=8,
        hidden_dim=4096,
        intermediate_size_per_partition=1536,
    )
    comm_impl.get_symm_buffer_for_mega_moe = MagicMock()
    mc2_group = SimpleNamespace(device_group=object())

    with patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_mc2_group", return_value=mc2_group):
        comm_impl._init_mega_moe_symm_buffer()

    call = comm_impl.get_symm_buffer_for_mega_moe.call_args
    assert call.args[2] == _MEGA_MOE_TOKENS_PER_RANK_LIMIT


def test_cann_mega_moe_splits_batches_above_operator_limit():
    comm_impl = _make_comm_impl()
    comm_impl.token_dispatcher.global_bs = 0
    first_out = torch.ones(_MEGA_MOE_TOKENS_PER_RANK_LIMIT, 4)
    second_out = torch.full((1, 4), 2.0)
    comm_impl.mega_moe.side_effect = [
        (first_out, torch.tensor([1, 2], dtype=torch.int32)),
        (second_out, torch.tensor([3, 4], dtype=torch.int32)),
    ]
    num_tokens = _MEGA_MOE_TOKENS_PER_RANK_LIMIT + 1

    with patch("vllm_ascend.ops.fused_moe.moe_comm_method.torch.cat") as mock_cat:
        out, expert_tokens = comm_impl._apply_cann_mega_moe(
            _make_fused_experts_input(num_tokens, mc2_mask=torch.ones(num_tokens, dtype=torch.bool))
        )

    mock_cat.assert_not_called()
    assert [call.args[0].shape[0] for call in comm_impl.mega_moe.call_args_list] == [
        _MEGA_MOE_TOKENS_PER_RANK_LIMIT,
        1,
    ]
    assert [call.kwargs["x_active_mask"].shape[0] for call in comm_impl.mega_moe.call_args_list] == [
        _MEGA_MOE_TOKENS_PER_RANK_LIMIT,
        1,
    ]
    torch.testing.assert_close(out, torch.cat([first_out, second_out], dim=0))
    torch.testing.assert_close(expert_tokens, torch.tensor([4, 6], dtype=torch.int32))


def test_minimax_m3_uses_intermediate_size_for_megamoe_support():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["MiniMaxM3ForCausalLM"],
            hf_text_config=SimpleNamespace(
                hidden_size=4096,
                intermediate_size=1536,
                moe_quantize="w8a8",
            ),
        )
    )

    assert AscendConfig._is_megamoe_supported_by_config(vllm_config)
