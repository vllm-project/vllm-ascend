# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.patch.worker import patch_qwen3_5


def test_qwen3_5_text_attention_uses_standard_rope():
    attention = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        rotary_emb=SimpleNamespace(),
    )
    assert not patch_qwen3_5._uses_multimodal_rope(attention)


def test_qwen3_5_multimodal_attention_uses_mrope():
    attention = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe"),
        rotary_emb=SimpleNamespace(mrope_section=[11, 11, 10]),
    )
    assert patch_qwen3_5._uses_multimodal_rope(attention)


@pytest.mark.parametrize("positions_ndim", [1, 2])
def test_qwen3_5_attention_supplies_three_rope_planes(positions_ndim):
    num_tokens, head_size, rope_dim = 4, 256, 64
    positions = torch.tensor([3, 7, 11, 15])
    if positions_ndim == 2:
        positions = torch.stack([positions, positions + 20, positions + 40])
    cache = torch.arange(64 * rope_dim, dtype=torch.float32).reshape(64, rope_dim)
    q = torch.zeros(num_tokens, 2 * head_size)
    k = v = torch.zeros(num_tokens, head_size)
    attention = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe_text", rms_norm_eps=1e-6),
        rotary_emb=SimpleNamespace(
            cos_sin_cache=cache,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
            rotary_dim=rope_dim,
        ),
        qkv_proj=MagicMock(return_value=(torch.zeros(num_tokens, 4 * head_size), None)),
        q_norm=SimpleNamespace(weight=torch.zeros(head_size)),
        k_norm=SimpleNamespace(weight=torch.zeros(head_size)),
        num_heads=2,
        num_kv_heads=1,
        head_dim=head_size,
        attn_output_gate=False,
        attn=MagicMock(return_value=q),
        o_proj=MagicMock(return_value=(q, None)),
    )
    with patch.object(
        torch.ops.vllm,
        "triton_split_qkv_rmsnorm_mrope",
        return_value=(q, k, v, None),
        create=True,
    ) as fused:
        patch_qwen3_5.AscendQwen3NextAttention.forward(attention, positions, q)

    # The Triton kernel uses raw offsets for contiguous T/H/W planes.
    cos_sin = fused.call_args.kwargs["cos_sin"]
    assert cos_sin.shape == (3, num_tokens, rope_dim)
    assert cos_sin.is_contiguous()
    expected_positions = positions if positions_ndim == 2 else positions.unsqueeze(0).expand(3, -1)
    torch.testing.assert_close(cos_sin, cache[expected_positions])


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None,
    reason="Qwen3.5 MTP model is not available in this vLLM version.",
)
def test_qwen3_5_mtp_forward_uses_local_inputs_on_last_pp_rank():
    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 2
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(2, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x + 1)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x + 2)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4] + x[:, 4:])
    layer0 = MagicMock(return_value=(torch.full((2, 4), 3.0), torch.full((2, 4), 4.0)))
    layer1 = MagicMock(return_value=(torch.full((2, 4), 5.0), torch.full((2, 4), 6.0)))
    layer1.use_attn_reduce_scatter_for_moe = False
    predictor.layers = [layer0, layer1]
    predictor.norm = MagicMock(return_value=(torch.full((2, 4), 7.0), None))

    with patch(
        "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        output = predictor.forward(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 4),
            intermediate_tensors=IntermediateTensors({"hidden_states": torch.full((2, 4), 99.0)}),
            spec_step_idx=3,
        )

    predictor.embed_input_ids.assert_called_once()
    layer1.assert_called_once()
    predictor.norm.assert_called_once()
    assert torch.equal(output, torch.full((2, 4), 7.0))


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None,
    reason="Qwen3.5 MTP model is not available in this vLLM version.",
)
def test_qwen3_5_mtp_forward_returns_intermediate_tensors_on_non_last_pp_rank():
    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 1
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(1, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4])
    predictor.layers = [MagicMock(return_value=(torch.full((1, 4), 3.0), torch.full((1, 4), 4.0)))]
    predictor.norm = MagicMock()

    with (
        patch(
            "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
            return_value=SimpleNamespace(is_last_rank=False),
        ),
        patch(
            "vllm.model_executor.models.utils.sequence_parallel_chunk",
            side_effect=lambda x: x,
        ),
        patch(
            "vllm_ascend.patch.worker.patch_qwen3_5.sequence_parallel_chunk",
            side_effect=lambda x: x,
            create=True,
        ),
    ):
        output = predictor.forward(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            hidden_states=torch.zeros(1, 4),
        )

    assert isinstance(output, IntermediateTensors)
    assert torch.equal(output["hidden_states"], torch.full((1, 4), 3.0))
    assert torch.equal(output["residual"], torch.full((1, 4), 4.0))
    predictor.norm.assert_not_called()
