#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#


import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------
HEAD_SIZE = 64
ROTARY_DIM = 64
MAX_POS = 2048
BASE = 10000.0
DTYPE = torch.bfloa16
SEQ_LEN = 4
NUM_HEADS = 2


def _make_tensors(seq_len=SEQ_LEN, num_heads=NUM_HEADS, head_size=HEAD_SIZE):
    positions = torch.arange(seq_len, dtype=torch.long)
    query = torch.randn(seq_len, num_heads * head_size)
    key = torch.randn(seq_len, num_heads * head_size)
    return positions, query, key


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_init_side_effects():
    """
    Suppress all side-effects that fire during __init__ so every test starts
    from a clean, predictable state without needing real NPU ops or vLLM
    global config.
    """
    with (
        patch("your_module._record_cos_sin_cache"),
        patch("your_module._record_cos_and_sin_cache_interleaved"),
        patch("your_module.get_current_vllm_config") as mock_cfg,
    ):
        # Default: speculative_config is None → use_mtp = False
        mock_cfg.return_value.speculative_config = None
        yield mock_cfg


@pytest.fixture()
def make_embedding(patch_init_side_effects):
    """Factory that creates an AscendRotaryEmbedding with controllable use_mtp."""

    def _factory(use_mtp: bool = False, is_neox_style: bool = True):
        spec_cfg = MagicMock(method="mtp") if use_mtp else None
        patch_init_side_effects.return_value.speculative_config = spec_cfg

        with patch("your_module.RotaryEmbedding.__init__") as mock_parent_init:
            mock_parent_init.return_value = None
            from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding

            emb = AscendRotaryEmbedding.__new__(AscendRotaryEmbedding)
            # Manually set attrs that the real parent would set
            emb.head_size = HEAD_SIZE
            emb.rotary_dim = ROTARY_DIM
            emb.is_neox_style = is_neox_style
            emb.cos_sin_cache = torch.zeros(MAX_POS, ROTARY_DIM)
            # Call __init__ to exercise our code path
            AscendRotaryEmbedding.__init__(
                emb, HEAD_SIZE, ROTARY_DIM, MAX_POS, BASE, is_neox_style, DTYPE
            )
        return emb

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestForwardOot:

    @patch("your_module.torch.ops.vllm.npu_rotary_embedding")
    @patch("your_module._EXTRA_CTX")
    def test_basic_call_delegates_to_npu_op(self, mock_ctx, mock_npu_op, make_embedding):
        """forward_oot always calls npu_rotary_embedding and returns its result."""
        mock_ctx.is_draft_model = False
        mock_ctx.flash_comm_v1_enabled = False
        expected_output = (torch.randn(SEQ_LEN, NUM_HEADS * HEAD_SIZE),) * 2
        mock_npu_op.return_value = expected_output

        emb = make_embedding()
        positions, query, key = _make_tensors()

        result = emb.forward_oot(positions, query, key)

        mock_npu_op.assert_called_once_with(
            positions, query, key, emb.cos_sin_cache,
            HEAD_SIZE, ROTARY_DIM, emb.is_neox_style,
        )
        assert result is expected_output

    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context._EXTRA_CTX")
    def test_neox_style_override_true(self, mock_ctx, mock_npu_op, make_embedding):
        """is_neox_style_override=True wins over self.is_neox_style=False."""
        mock_ctx.is_draft_model = False
        mock_ctx.flash_comm_v1_enabled = False
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(is_neox_style=False)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key, is_neox_style_override=True)

        _, kwargs = mock_npu_op.call_args
        # Verify the override was forwarded correctly
        assert mock_npu_op.call_args[0][-1] is True  # last positional arg = is_neox_style

    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context._EXTRA_CTX")
    def test_neox_style_override_false(self, mock_ctx, mock_npu_op, make_embedding):
        """is_neox_style_override=False wins over self.is_neox_style=True."""
        mock_ctx.is_draft_model = False
        mock_ctx.flash_comm_v1_enabled = False
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(is_neox_style=True)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key, is_neox_style_override=False)

        assert mock_npu_op.call_args[0][-1] is False

    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context._EXTRA_CTX")
    def test_neox_style_override_none_uses_self(self, mock_ctx, mock_npu_op, make_embedding):
        """When override is None, self.is_neox_style is used unchanged."""
        mock_ctx.is_draft_model = False
        mock_ctx.flash_comm_v1_enabled = False
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(is_neox_style=True)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key, is_neox_style_override=None)

        assert mock_npu_op.call_args[0][-1] is True

    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context._EXTRA_CTX")
    def test_gather_unpad_called_when_all_conditions_met(
        self, mock_ctx, mock_npu_op, mock_gather, make_embedding
    ):
        """
        maybe_all_gather_and_maybe_unpad is called iff:
          is_draft_model=True AND use_mtp=True AND flash_comm_v1_enabled=True
        """
        mock_ctx.is_draft_model = True
        mock_ctx.flash_comm_v1_enabled = True
        gathered_positions = torch.arange(SEQ_LEN, dtype=torch.long)
        mock_gather.return_value = gathered_positions
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(use_mtp=True)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key)

        mock_gather.assert_called_once()
        # npu op should receive the gathered positions, not the originals
        assert mock_npu_op.call_args[0][0] is gathered_positions

    @pytest.mark.parametrize("is_draft_model,flash_comm,use_mtp", [
        (False, True,  True),   # not draft
        (True,  False, True),   # flash_comm disabled
        (True,  True,  False),  # use_mtp disabled
    ])
    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context._EXTRA_CTX")
    def test_gather_unpad_skipped_unless_all_conditions_met(
        self, mock_ctx, mock_npu_op, mock_gather,
        is_draft_model, flash_comm, use_mtp, make_embedding,
    ):
        """gather/unpad must NOT fire if any one of the three conditions is False."""
        mock_ctx.is_draft_model = is_draft_model
        mock_ctx.flash_comm_v1_enabled = flash_comm
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(use_mtp=use_mtp)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key)

        mock_gather.assert_not_called()
        # Original positions tensor is passed through untouched
        assert mock_npu_op.call_args[0][0] is positions

    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context._EXTRA_CTX")
    def test_offsets_parameter_accepted(self, mock_ctx, mock_npu_op, make_embedding):
        """forward_oot should accept an offsets tensor without raising."""
        mock_ctx.is_draft_model = False
        mock_ctx.flash_comm_v1_enabled = False
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding()
        positions, query, key = _make_tensors()
        offsets = torch.zeros(SEQ_LEN, dtype=torch.long)

        # Should not raise
        emb.forward_oot(positions, query, key, offsets=offsets)
        mock_npu_op.assert_called_once()