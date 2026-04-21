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

import importlib
import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, YaRNScalingRotaryEmbedding

from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding, AscendYaRNRotaryEmbedding

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

    added = parent_params - child_params
    removed = child_params - parent_params

    assert not added, (
        f"{parent_func.__name__} added new parameter(s): {added}. "
        f"Check whether {child_func.__name__} needs to forward them."
    )
    assert not removed, (
        f"{parent_func.__name__} removed parameter(s): {removed}. "
        f"Check whether {child_func.__name__} needs to forward them."
    )


@pytest.fixture()
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


@pytest.fixture(autouse=True)
def reset_rotary_embedding_globals():
    rotary_embedding = importlib.import_module("vllm_ascend.ops.rotary_embedding")
    for name in (
        "_cos_mla",
        "_sin_mla",
        "_cos_cache",
        "_sin_cache",
        "_cos_sin_cache",
        "_cos",
        "_sin",
        "_cos_slice",
        "_sin_slice",
    ):
        setattr(rotary_embedding, name, None)
    yield
    for name in (
        "_cos_mla",
        "_sin_mla",
        "_cos_cache",
        "_sin_cache",
        "_cos_sin_cache",
        "_cos",
        "_sin",
        "_cos_slice",
        "_sin_slice",
    ):
        setattr(rotary_embedding, name, None)


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
            AscendRotaryEmbedding.__init__(emb, HEAD_SIZE, ROTARY_DIM, MAX_POS, BASE, is_neox_style, DTYPE)
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
            positions,
            query,
            key,
            emb.cos_sin_cache,
            HEAD_SIZE,
            ROTARY_DIM,
            emb.is_neox_style,
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

    @pytest.mark.parametrize(
        "is_draft_model,flash_comm,use_mtp",
        [
            (False, True, True),  # not draft
            (True, False, True),  # flash_comm disabled
            (True, True, False),  # use_mtp disabled
        ],
    )
    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    @patch("torch.ops.vllm.npu_rotary_embedding")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_gather_unpad_skipped_unless_all_conditions_met(
        self,
        mock_get_forward_context,
        mock_npu_op,
        mock_gather,
        is_draft_model,
        flash_comm,
        use_mtp,
        make_embedding,
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
        check_parent_init_signature_has_not_changed(RotaryEmbedding.__init__, AscendRotaryEmbedding.__init__)


class TestRotaryEmbeddingUtilities:
    def test_record_cos_sin_cache_records_once(self):
        from vllm_ascend.ops import rotary_embedding

        first = torch.randn(4, 8)
        second = torch.randn(4, 8)

        rotary_embedding._record_cos_sin_cache(first)
        rotary_embedding._record_cos_sin_cache(second)

        assert rotary_embedding._cos_sin_cache is first

    def test_record_cos_and_sin_cache_updates_both(self):
        from vllm_ascend.ops import rotary_embedding

        cos = torch.randn(3, 4)
        sin = torch.randn(3, 4)

        rotary_embedding._record_cos_and_sin_cache(cos, sin)

        assert rotary_embedding._cos_cache is cos
        assert rotary_embedding._sin_cache is sin

    def test_record_cos_and_sin_cache_interleaved_expands_once(self):
        from vllm_ascend.ops import rotary_embedding

        cache = torch.tensor(
            [
                [1.0, 2.0, 10.0, 20.0],
                [3.0, 4.0, 30.0, 40.0],
            ],
            dtype=torch.float32,
        )

        rotary_embedding._record_cos_and_sin_cache_interleaved(cache)

        expected_cos = torch.tensor([[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]])
        expected_sin = torch.tensor([[10.0, 20.0, 10.0, 20.0], [30.0, 40.0, 30.0, 40.0]])
        torch.testing.assert_close(rotary_embedding._cos_cache, expected_cos)
        torch.testing.assert_close(rotary_embedding._sin_cache, expected_sin)

        original_cos = rotary_embedding._cos_cache
        original_sin = rotary_embedding._sin_cache
        rotary_embedding._record_cos_and_sin_cache_interleaved(torch.zeros_like(cache))
        assert rotary_embedding._cos_cache is original_cos
        assert rotary_embedding._sin_cache is original_sin

    @patch("vllm_ascend.ops.rotary_embedding.is_vl_model", return_value=False)
    @patch("vllm_ascend.ops.rotary_embedding.has_rope", return_value=True)
    def test_set_cos_and_sin_uses_partial_rotary_factor(self, mock_has_rope, mock_is_vl):
        from vllm_ascend.ops import rotary_embedding

        config = MagicMock()
        config.model_config.use_mla = False
        config.model_config.get_head_size.return_value = 64
        config.model_config.hf_text_config = MagicMock()
        config.model_config.hf_text_config.partial_rotary_factor = 0.5
        config.scheduler_config.max_num_batched_tokens = 16

        rotary_embedding.set_cos_and_sin(
            config, max_num_reqs=2, decode_token_per_req=1, dtype=torch.float16, device="cpu"
        )

        assert rotary_embedding._cos.shape == (1, 16, 1, 32)
        assert rotary_embedding._sin.shape == (1, 16, 1, 32)

    @patch("vllm_ascend.ops.rotary_embedding.is_vl_model", return_value=False)
    @patch("vllm_ascend.ops.rotary_embedding.has_rope", return_value=True)
    def test_set_cos_and_sin_uses_rotary_dim_when_partial_factor_absent(self, mock_has_rope, mock_is_vl):
        from vllm_ascend.ops import rotary_embedding

        hf_text_config = MagicMock(spec=["rotary_dim"])
        hf_text_config.rotary_dim = 24

        config = MagicMock()
        config.model_config.use_mla = False
        config.model_config.get_head_size.return_value = 64
        config.model_config.hf_text_config = hf_text_config
        config.scheduler_config.max_num_batched_tokens = 8

        rotary_embedding.set_cos_and_sin(
            config, max_num_reqs=2, decode_token_per_req=1, dtype=torch.float32, device="cpu"
        )

        assert rotary_embedding._cos.shape == (1, 8, 1, 24)
        assert rotary_embedding._sin.shape == (1, 8, 1, 24)

    def test_get_cos_and_sin_mla_without_cache_returns_selected_positions(self):
        from vllm_ascend.ops import rotary_embedding

        rotary_embedding._cos_cache = torch.arange(12, dtype=torch.float32).view(3, 4)
        rotary_embedding._sin_cache = torch.arange(12, 24, dtype=torch.float32).view(3, 4)
        positions = torch.tensor([2, 0], dtype=torch.long)

        cos, sin = rotary_embedding.get_cos_and_sin_mla(positions, use_cache=False)

        torch.testing.assert_close(cos, rotary_embedding._cos_cache[positions].unsqueeze(1).unsqueeze(2))
        torch.testing.assert_close(sin, rotary_embedding._sin_cache[positions].unsqueeze(1).unsqueeze(2))

    def test_get_cos_and_sin_mla_with_cache_reuses_preallocated_buffers(self):
        from vllm_ascend.ops import rotary_embedding

        rotary_embedding._cos_cache = torch.arange(12, dtype=torch.float32).view(3, 4)
        rotary_embedding._sin_cache = torch.arange(12, 24, dtype=torch.float32).view(3, 4)
        rotary_embedding._cos_mla = torch.full((4, 1, 1, 4), -1.0)
        rotary_embedding._sin_mla = torch.full((4, 1, 1, 4), -1.0)
        positions = torch.tensor([1, 2], dtype=torch.long)

        cos, sin = rotary_embedding.get_cos_and_sin_mla(positions, use_cache=True)

        assert cos.data_ptr() == rotary_embedding._cos_mla[:2].data_ptr()
        assert sin.data_ptr() == rotary_embedding._sin_mla[:2].data_ptr()
        torch.testing.assert_close(cos, rotary_embedding._cos_cache[positions].unsqueeze(1).unsqueeze(2))
        torch.testing.assert_close(sin, rotary_embedding._sin_cache[positions].unsqueeze(1).unsqueeze(2))

    def test_update_cos_sin_populates_slice_cache(self):
        from vllm_ascend.ops import rotary_embedding

        rotary_embedding._cos_sin_cache = torch.tensor(
            [
                [1.0, 2.0, 10.0, 20.0],
                [3.0, 4.0, 30.0, 40.0],
                [5.0, 6.0, 50.0, 60.0],
            ],
            dtype=torch.float32,
        )
        rotary_embedding._cos = torch.zeros(1, 4, 1, 4, dtype=torch.float32)
        rotary_embedding._sin = torch.zeros(1, 4, 1, 4, dtype=torch.float32)
        positions = torch.tensor([2, 0], dtype=torch.long)

        rotary_embedding.update_cos_sin(positions)

        expected_cos = torch.tensor([[[[5.0, 6.0, 5.0, 6.0]], [[1.0, 2.0, 1.0, 2.0]]]])
        expected_sin = torch.tensor([[[[50.0, 60.0, 50.0, 60.0]], [[10.0, 20.0, 10.0, 20.0]]]])
        torch.testing.assert_close(rotary_embedding._cos_slice, expected_cos)
        torch.testing.assert_close(rotary_embedding._sin_slice, expected_sin)

    def test_update_cos_sin_returns_early_when_caches_missing(self):
        from vllm_ascend.ops import rotary_embedding

        rotary_embedding._cos_slice = "sentinel_cos"
        rotary_embedding._sin_slice = "sentinel_sin"

        rotary_embedding.update_cos_sin(torch.tensor([0, 1], dtype=torch.long))

        assert rotary_embedding._cos_slice == "sentinel_cos"
        assert rotary_embedding._sin_slice == "sentinel_sin"

    def test_get_cos_and_sin_slice_returns_current_views(self):
        from vllm_ascend.ops import rotary_embedding

        rotary_embedding._cos_slice = torch.ones(1, 2, 1, 4)
        rotary_embedding._sin_slice = torch.zeros(1, 2, 1, 4)

        cos, sin = rotary_embedding.get_cos_and_sin_slice()

        assert cos is rotary_embedding._cos_slice
        assert sin is rotary_embedding._sin_slice


class TestRopeForwardOOT:
    @patch("vllm_ascend.ops.rotary_embedding.HAS_TRITON", False)
    @patch("torch_npu._npu_rotary_embedding")
    def test_full_rotary_passes_head_size_to_npu_op(self, mock_rotary):
        from vllm_ascend.ops.rotary_embedding import rope_forward_oot

        def fake_rotary(positions, query, key, dim, cos_sin_cache, is_neox_style):
            query.add_(1.0)
            key.add_(2.0)

        mock_rotary.side_effect = fake_rotary
        positions, query, key = _make_tensors()
        cache = torch.randn(MAX_POS, ROTARY_DIM * 2)

        result_q, result_k = rope_forward_oot(
            positions,
            query.clone(),
            key.clone(),
            cache,
            head_size=HEAD_SIZE,
            rotary_dim=HEAD_SIZE,
            is_neox_style=True,
        )

        assert mock_rotary.call_args.args[3] == HEAD_SIZE
        torch.testing.assert_close(result_q, query + 1.0)
        torch.testing.assert_close(result_k, key + 2.0)

    @patch("vllm_ascend.ops.rotary_embedding.HAS_TRITON", False)
    @patch("torch_npu._npu_rotary_embedding")
    def test_partial_rotary_only_updates_rotary_slice(self, mock_rotary):
        from vllm_ascend.ops.rotary_embedding import rope_forward_oot

        rotary_dim = 32

        def fake_rotary(positions, query, key, dim, cos_sin_cache, is_neox_style):
            query.add_(3.0)
            key.add_(5.0)

        mock_rotary.side_effect = fake_rotary
        positions, query, key = _make_tensors()
        cache = torch.randn(MAX_POS, rotary_dim * 2)

        result_q, result_k = rope_forward_oot(
            positions,
            query.clone(),
            key.clone(),
            cache,
            head_size=HEAD_SIZE,
            rotary_dim=rotary_dim,
            is_neox_style=True,
        )

        expected_q = query.view(SEQ_LEN, NUM_HEADS, HEAD_SIZE).clone()
        expected_k = key.view(SEQ_LEN, NUM_HEADS, HEAD_SIZE).clone()
        expected_q[..., :rotary_dim] += 3.0
        expected_k[..., :rotary_dim] += 5.0
        torch.testing.assert_close(result_q, expected_q.reshape_as(query))
        torch.testing.assert_close(result_k, expected_k.reshape_as(key))
        assert mock_rotary.call_args.args[3] == rotary_dim

    def test_offsets_are_not_supported(self):
        from vllm_ascend.ops.rotary_embedding import rope_forward_oot

        positions, query, key = _make_tensors()
        cache = torch.randn(MAX_POS, ROTARY_DIM * 2)
        offsets = torch.arange(SEQ_LEN, dtype=torch.long)

        with pytest.raises(NotImplementedError, match="Batched rotary embedding"):
            rope_forward_oot(
                positions,
                query,
                key,
                cache,
                head_size=HEAD_SIZE,
                rotary_dim=ROTARY_DIM,
                is_neox_style=True,
                offsets=offsets,
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
            YaRNScalingRotaryEmbedding.__init__, AscendYaRNRotaryEmbedding.__init__
        )
