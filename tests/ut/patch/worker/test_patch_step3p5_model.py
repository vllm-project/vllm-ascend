# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.patch.worker import patch_step3p5_model

_skip_no_step3p5 = pytest.mark.skipif(
    patch_step3p5_model.Step3p5DecoderLayer is None,
    reason="Step3p5DecoderLayer is not available in this vLLM version.",
)


def _make_layer(*, layer_idx=0, use_moe=False, attn_out=None, mlp_out=None):
    """Build a fake Step3p5DecoderLayer with identity layernorms."""
    layer = MagicMock()
    layer.layer_idx = layer_idx
    layer.use_moe = use_moe
    layer.input_layernorm = MagicMock(side_effect=lambda x: x)
    layer.self_attn = MagicMock(return_value=attn_out if attn_out is not None else torch.zeros(2, 8))
    layer.post_attention_layernorm = MagicMock(side_effect=lambda x: x)
    layer.mlp = MagicMock(return_value=mlp_out if mlp_out is not None else torch.zeros(2, 8))
    layer.moe = MagicMock()
    return layer


@_skip_no_step3p5
def test_layer0_residual_is_chunked_when_fc1_enabled():
    """Layer 0 of a VL model under FC1: the full-replicated residual must be
    pad+slice'd to [n_local, H] before being added to the reduce_scatter
    attention output, and the original forward must not run."""
    # Replicated residual [N=4, H=8]; tp=2, rank=0 -> slice rows [0:2].
    hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    attn_out = torch.full((2, 8), 10.0)  # FC1 o_proj reduce_scatter -> [2, 8]
    mlp_out = torch.full((2, 8), 100.0)
    layer = _make_layer(layer_idx=0, attn_out=attn_out, mlp_out=mlp_out)
    positions = torch.tensor([0, 1])

    ctx = SimpleNamespace(flash_comm_v1_enabled=True, is_draft_model=False)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)

    with (
        patch("vllm_ascend.ascend_forward_context._EXTRA_CTX", ctx),
        patch(
            "vllm_ascend.patch.worker.patch_step3p5_model.get_tp_group",
            return_value=tp_group,
        ),
        patch("vllm_ascend.patch.worker.patch_step3p5_model._original_layer_forward") as orig,
    ):
        out = patch_step3p5_model._patched_forward(layer, positions, hidden)

    # FC1 path taken: original forward must NOT run.
    orig.assert_not_called()
    layer.self_attn.assert_called_once()
    layer.mlp.assert_called_once()
    layer.moe.assert_not_called()

    # pad_needed = n_local * tp - N = 2 * 2 - 4 = 0, no padding
    # residual = hidden[0:2] (rank 0 slice)
    n_local = 2
    residual = hidden[0 * n_local : (0 + 1) * n_local].contiguous()
    expected = mlp_out + attn_out + residual
    assert out.shape == torch.Size([2, 8])
    assert torch.equal(out, expected)


@_skip_no_step3p5
def test_layer0_residual_pad_when_n_not_divisible_by_tp():
    """When N is not divisible by tp, residual is padded before slicing so
    every rank gets an equal-sized chunk."""
    # N=3, tp=2: FC1 scatters to n_local = ceil(3/2) = 2
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    attn_out = torch.full((2, 8), 10.0)
    mlp_out = torch.full((2, 8), 100.0)
    layer = _make_layer(layer_idx=0, attn_out=attn_out, mlp_out=mlp_out)
    positions = torch.tensor([0, 1])

    ctx = SimpleNamespace(flash_comm_v1_enabled=True, is_draft_model=False)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)

    with (
        patch("vllm_ascend.ascend_forward_context._EXTRA_CTX", ctx),
        patch(
            "vllm_ascend.patch.worker.patch_step3p5_model.get_tp_group",
            return_value=tp_group,
        ),
    ):
        out = patch_step3p5_model._patched_forward(layer, positions, hidden)

    # pad_needed = 2 * 2 - 3 = 1
    # residual padded: [3,8] -> [4,8], then slice [0:2]
    padded = torch.nn.functional.pad(hidden, (0, 0, 0, 1))
    residual = padded[0:2].contiguous()
    expected = mlp_out + attn_out + residual
    assert out.shape == torch.Size([2, 8])
    assert torch.equal(out, expected)


@_skip_no_step3p5
def test_layer0_residual_noop_when_same_shape():
    """Text model: input already SP-sharded [N/tp, H], same as FC1 output
    shape.  The pad+slice branch should NOT fire — just a plain add."""
    # Both residual and attn_out are [2, 8] -> shape match.
    hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    attn_out = torch.full((2, 8), 10.0)
    mlp_out = torch.full((2, 8), 100.0)
    layer = _make_layer(layer_idx=0, attn_out=attn_out, mlp_out=mlp_out)
    positions = torch.tensor([0, 1])

    ctx = SimpleNamespace(flash_comm_v1_enabled=True, is_draft_model=False)
    tp_group = SimpleNamespace(world_size=2, rank_in_group=0)

    with (
        patch("vllm_ascend.ascend_forward_context._EXTRA_CTX", ctx),
        patch(
            "vllm_ascend.patch.worker.patch_step3p5_model.get_tp_group",
            return_value=tp_group,
        ),
        patch("vllm_ascend.patch.worker.patch_step3p5_model._original_layer_forward") as orig,
    ):
        out = patch_step3p5_model._patched_forward(layer, positions, hidden)

    # residual.shape[0] == n_local, so pad+slice is skipped
    orig.assert_not_called()
    expected = mlp_out + attn_out + hidden
    assert torch.equal(out, expected)


@_skip_no_step3p5
@pytest.mark.parametrize(
    "layer_idx, flash_comm_v1_enabled, is_draft_model",
    [
        (1, True, False),  # not layer 0
        (0, False, False),  # FC1 disabled
        (0, True, True),  # MTP draft model
    ],
    ids=["non-layer0", "fc1-disabled", "draft-model"],
)
def test_delegates_to_original_when_layer0_fc1_condition_not_met(layer_idx, flash_comm_v1_enabled, is_draft_model):
    """Outside the (layer0 + FC1 + main model) case the patched forward must be
    a transparent passthrough to the original implementation."""
    hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    layer = _make_layer(layer_idx=layer_idx)
    positions = torch.tensor([0, 1])

    ctx = SimpleNamespace(
        flash_comm_v1_enabled=flash_comm_v1_enabled,
        is_draft_model=is_draft_model,
    )

    with (
        patch("vllm_ascend.ascend_forward_context._EXTRA_CTX", ctx),
        patch("vllm_ascend.patch.worker.patch_step3p5_model._original_layer_forward") as orig,
    ):
        patch_step3p5_model._patched_forward(layer, positions, hidden)

    orig.assert_called_once_with(layer, positions, hidden)
    layer.self_attn.assert_not_called()
