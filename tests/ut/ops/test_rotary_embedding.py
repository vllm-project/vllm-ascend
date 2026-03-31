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


import inspect
import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from vllm.model_executor.layers.rotary_embedding import (YaRNScalingRotaryEmbedding, RotaryEmbedding)
from vllm_ascend.ops.rotary_embedding import (AscendYaRNRotaryEmbedding, AscendRotaryEmbedding)


HEAD_SIZE = 64
ROTARY_DIM = 64
MAX_POS = 2048
BASE = 10000.0
DTYPE = torch.bfloat16
SEQ_LEN = 4
NUM_HEADS = 2


def _make_tensors(seq_len=SEQ_LEN, num_heads=NUM_HEADS, head_size=HEAD_SIZE):
    positions = torch.arange(seq_len, dtype=torch.long)
    query = torch.randn(seq_len, num_heads * head_size)
    key = torch.randn(seq_len, num_heads * head_size)
    return positions, query, key


def check_parent_init_signature_has_not_changed(parent_func, child_func):
    parent_sig = inspect.signature(parent_func)
    parent_params = set(parent_sig.parameters) - {"self"}

    child_sig = inspect.signature(child_func)
    child_params = set(child_sig.parameters) - {"self"}

    added   = parent_params - child_params
    removed = child_params - parent_params

    assert not added, (
        f"{parent_func.__name__} added new parameter(s): {added}. "
        f"Check whether {child_func.__name__} needs to forward them."
    )
    assert not removed, (
        f"{parent_func.__name__} removed parameter(s): {removed}. "
        f"Check whether {child_func.__name__} needs to forward them."
    )


@pytest.fixture(autouse=True)
def patch_init_side_effects():
    """
    Suppress all side-effects that fire during __init__ so every test starts
    from a clean, predictable state without needing real NPU ops or vLLM
    global config.
    """
    with (
        patch("vllm_ascend.ops.rotary_embedding._record_cos_sin_cache"),
        patch("vllm_ascend.ops.rotary_embedding._record_cos_and_sin_cache_interleaved"),
        patch("vllm_ascend.ops.rotary_embedding.get_current_vllm_config") as mock_cfg,
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

        with patch("vllm_ascend.ops.rotary_embedding.RotaryEmbedding.__init__") as mock_parent_init:
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


@pytest.fixture()
def make_yarn_embedding(patch_init_side_effects):
    """
    Factory for AscendYaRNRotaryEmbedding with parent __init__ suppressed.
    patch_init_side_effects is the same autouse fixture as before.
    """
    def _factory(is_neox_style: bool = True):
        with patch("vllm_ascend.ops.rotary_embedding.YaRNScalingRotaryEmbedding.__init__") as mock_parent_init:
            mock_parent_init.return_value = None
            from vllm_ascend.ops.rotary_embedding import AscendYaRNRotaryEmbedding

            emb = AscendYaRNRotaryEmbedding.__new__(AscendYaRNRotaryEmbedding)
            emb.head_size = HEAD_SIZE
            emb.rotary_dim = ROTARY_DIM
            emb.is_neox_style = is_neox_style
            emb.cos_sin_cache = torch.zeros(MAX_POS, ROTARY_DIM)
            AscendYaRNRotaryEmbedding.__init__(
                emb,
                head_size=HEAD_SIZE,
                rotary_dim=ROTARY_DIM,
                max_position_embeddings=MAX_POS,
                base=BASE,
                is_neox_style=is_neox_style,
                scaling_factor=1.0,
                dtype=DTYPE,
            )
        return emb

    return _factory


class TestAscendEmbeddingForwardOOT:

    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_basic_call_delegates_to_npu_op(self, mock_get_forward_context, mock_npu_op, make_embedding):
        """forward_oot always calls npu_rotary_embedding and returns its result."""
        mock_get_forward_context.return_value = MagicMock()
        mock_get_forward_context.return_value.is_draft_model = False
        mock_get_forward_context.return_value.flash_comm_v1_enabled = False
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
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_neox_style_override_true(self, mock_get_forward_context, mock_npu_op, make_embedding):
        """is_neox_style_override=True wins over self.is_neox_style=False."""
        mock_get_forward_context.return_value = MagicMock()
        mock_get_forward_context.return_value.is_draft_model = False
        mock_get_forward_context.return_value.flash_comm_v1_enabled = False
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(is_neox_style=False)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key, is_neox_style_override=True)

        _, kwargs = mock_npu_op.call_args
        # Verify the override was forwarded correctly
        assert mock_npu_op.call_args[0][-1] is True  # last positional arg = is_neox_style

    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_neox_style_override_false(self, mock_get_forward_context, mock_npu_op, make_embedding):
        """is_neox_style_override=False wins over self.is_neox_style=True."""
        mock_get_forward_context.return_value = MagicMock()
        mock_get_forward_context.return_value.is_draft_model = False
        mock_get_forward_context.return_value.flash_comm_v1_enabled = False
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(is_neox_style=True)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key, is_neox_style_override=False)

        assert mock_npu_op.call_args[0][-1] is False

    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_neox_style_override_none_uses_self(self, mock_get_forward_context, mock_npu_op, make_embedding):
        """When override is None, self.is_neox_style is used unchanged."""
        mock_get_forward_context.return_value = MagicMock()
        mock_get_forward_context.return_value.is_draft_model = False
        mock_get_forward_context.return_value.flash_comm_v1_enabled = False
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(is_neox_style=True)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key, is_neox_style_override=None)

        assert mock_npu_op.call_args[0][-1] is True

    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_gather_unpad_called_when_all_conditions_met(
        self, mock_get_forward_context, mock_npu_op, mock_gather, make_embedding
    ):
        """
        maybe_all_gather_and_maybe_unpad is called iff:
          is_draft_model=True AND use_mtp=True AND flash_comm_v1_enabled=True
        """
        mock_get_forward_context.return_value = MagicMock()
        mock_get_forward_context.return_value.is_draft_model = True
        mock_get_forward_context.return_value.flash_comm_v1_enabled = True
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
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_gather_unpad_skipped_unless_all_conditions_met(
        self, mock_get_forward_context, mock_npu_op, mock_gather,
        is_draft_model, flash_comm, use_mtp, make_embedding,
    ):
        """gather/unpad must NOT fire if any one of the three conditions is False."""
        mock_get_forward_context.return_value = MagicMock()
        mock_get_forward_context.return_value.is_draft_model = is_draft_model
        mock_get_forward_context.return_value.flash_comm_v1_enabled = flash_comm
        mock_npu_op.return_value = MagicMock()

        emb = make_embedding(use_mtp=use_mtp)
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key)

        mock_gather.assert_not_called()
        # Original positions tensor is passed through untouched
        assert mock_npu_op.call_args[0][0] is positions
    
    def test_parent_init_signature_has_not_changed(self):
        """
        Fail loudly if RotaryEmbedding.__init__ adds, removes, or
        renames parameters, so a developer knows to update AscendRotaryEmbedding
        accordingly.
        """
        check_parent_init_signature_has_not_changed(
            RotaryEmbedding.__init__,
            AscendRotaryEmbedding.__init__
        )


class TestAscendYaRNRotaryEmbeddingForwardOOT:

    @patch("vllm_ascend.ops.rotary_embedding.AscendRotaryEmbedding.forward_oot")
    def test_delegates_to_ascend_rotary_forward_oot(self, mock_delegate, make_yarn_embedding):
        """forward_oot must delegate to AscendRotaryEmbedding.forward_oot."""
        expected = MagicMock()
        mock_delegate.return_value = expected

        emb = make_yarn_embedding()
        positions, query, key = _make_tensors()

        result = emb.forward_oot(positions, query, key)

        mock_delegate.assert_called_once_with(emb, positions, query, key, None, None)
        assert result is expected

    @patch("vllm_ascend.ops.rotary_embedding.AscendRotaryEmbedding.forward_oot")
    def test_return_value_passed_through(self, mock_delegate, make_yarn_embedding):
        """Return value from the delegate is returned unchanged."""
        sentinel = (torch.randn(SEQ_LEN, HEAD_SIZE), torch.randn(SEQ_LEN, HEAD_SIZE))
        mock_delegate.return_value = sentinel

        emb = make_yarn_embedding()
        positions, query, key = _make_tensors()

        result = emb.forward_oot(positions, query, key)

        assert result is sentinel

    @pytest.mark.parametrize("override", [True, False])
    @patch("vllm_ascend.ops.rotary_embedding.AscendRotaryEmbedding.forward_oot")
    def test_is_neox_style_override_forwarded(self, mock_delegate, override, make_yarn_embedding):
        """is_neox_style_override must be forwarded verbatim, both True and False."""
        mock_delegate.return_value = MagicMock()

        emb = make_yarn_embedding()
        positions, query, key = _make_tensors()

        emb.forward_oot(positions, query, key, is_neox_style_override=override)

        _, call_args, _ = mock_delegate.mock_calls[0]
        assert call_args[5] is override  # 6th positional arg

    @patch("vllm_ascend.ops.rotary_embedding.AscendRotaryEmbedding.forward_oot")
    def test_all_args_forwarded_together(self, mock_delegate, make_yarn_embedding):
        """Smoke test: all args passed simultaneously are all forwarded correctly."""
        mock_delegate.return_value = MagicMock()

        emb = make_yarn_embedding()
        positions, query, key = _make_tensors()
        offsets = torch.ones(SEQ_LEN, dtype=torch.long)

        emb.forward_oot(positions, query, key, offsets=offsets, is_neox_style_override=False)

        mock_delegate.assert_called_once_with(emb, positions, query, key, offsets, False)

    def test_parent_init_signature_has_not_changed(self):
        """
        Fail loudly if YaRNScalingRotaryEmbedding.__init__ adds, removes, or
        renames parameters, so a developer knows to update AscendYaRNRotaryEmbedding
        accordingly.
        """
        check_parent_init_signature_has_not_changed(
            YaRNScalingRotaryEmbedding.__init__,
            AscendYaRNRotaryEmbedding.__init__
        )