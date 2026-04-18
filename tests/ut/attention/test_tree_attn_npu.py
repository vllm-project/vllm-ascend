#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Unit tests for AscendTreeAttentionBackend."""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import torch

from vllm_ascend.attention.backends.tree_attn import (
    AscendTreeAttentionBackend,
    AscendTreeAttentionImpl,
    AscendTreeAttentionMetadata,
    AscendTreeAttentionMetadataBuilder,
    PAD_SIZE,
    _convert_tree_mask_for_npu,
    _get_depth_counts,
    _is_ancestor,
    _prepare_tree_attn_bias_gpu,
)


class TestTreeMaskConversion:
    """Tests for GPU to NPU mask conversion."""

    def test_is_ancestor(self):
        """Test ancestor relationship checking."""
        # (0,) is an ancestor of (0, 0)
        assert _is_ancestor((0,), (0, 0)) is True
        # (0, 0) is not an ancestor of (0,)
        assert _is_ancestor((0, 0), (0,)) is False
        # Same node is not its own ancestor
        assert _is_ancestor((0,), (0,)) is False
        # (0, 1) is not an ancestor of (0, 0)
        assert _is_ancestor((0, 1), (0, 0)) is False
        # (0,) is an ancestor of (0, 0, 0)
        assert _is_ancestor((0,), (0, 0, 0)) is True
        # (0, 0) is an ancestor of (0, 0, 0)
        assert _is_ancestor((0, 0), (0, 0, 0)) is True

    def test_get_depth_counts(self):
        """Test depth counting."""
        tree_choices = [(0,), (0, 0), (0, 1), (0, 0, 0), (0, 1, 0)]
        depth_counts = _get_depth_counts(tree_choices)
        assert depth_counts == [1, 2, 2]

    def test_prepare_tree_attn_bias_gpu(self):
        """Test GPU tree mask construction."""
        tree_choices = [(0,), (0, 0), (0, 1)]
        depth_counts = _get_depth_counts(tree_choices)
        mask = _prepare_tree_attn_bias_gpu(
            tree_choices, depth_counts, dtype=torch.float32, device=torch.device("cpu")
        )

        # tree_len = 4 (root + 3 drafts)
        assert mask.shape == (4, 4)

        # All tokens should attend to root (column 0)
        assert torch.all(mask[:, 0] == 0)

        # Each token should attend to itself
        for i in range(4):
            assert mask[i, i] == 0

        # (0, 0) should attend to (0,) - ancestor relationship (child -> parent)
        assert mask[2, 1] == 0

        # (0, 1) should not attend to (0, 0) - different branches
        assert mask[3, 2] == float('-inf')

    def test_convert_tree_mask_for_npu(self):
        """Test GPU to NPU mask conversion."""
        tree_choices = [(0,), (0, 0), (0, 1)]
        depth_counts = _get_depth_counts(tree_choices)
        gpu_mask = _prepare_tree_attn_bias_gpu(
            tree_choices, depth_counts, dtype=torch.float32, device=torch.device("cpu")
        )

        npu_mask = _convert_tree_mask_for_npu(gpu_mask, pad_size=PAD_SIZE)

        # Verify shape and dtype
        assert npu_mask.shape == (PAD_SIZE, PAD_SIZE)
        assert npu_mask.dtype == torch.int8

        # Verify conversion correctness
        tree_len = gpu_mask.shape[0]

        # Root column should be all 0 (attend)
        assert torch.all(npu_mask[:tree_len, 0] == 0)

        # Diagonal should be 0
        for i in range(tree_len):
            assert npu_mask[i, i] == 0

        # (0, 0) should attend to (0,) -> npu_mask[2, 1] == 0
        assert npu_mask[2, 1] == 0

        # (0, 1) should not attend to (0, 0) -> npu_mask[3, 2] == 1
        assert npu_mask[3, 2] == 1

        # Padding region should be 1 (block)
        assert torch.all(npu_mask[tree_len:, :] == 1)
        assert torch.all(npu_mask[:, tree_len:] == 1)


class TestAscendTreeAttentionBackend:
    """Tests for AscendTreeAttentionBackend."""

    def test_backend_name(self):
        """Test backend name."""
        assert AscendTreeAttentionBackend.get_name() == "TREE_ATTN"

    def test_backend_classes(self):
        """Test backend returns correct classes."""
        assert AscendTreeAttentionBackend.get_impl_cls() == AscendTreeAttentionImpl
        assert (
            AscendTreeAttentionBackend.get_builder_cls()
            == AscendTreeAttentionMetadataBuilder
        )

    def test_kv_cache_shape(self):
        """Test KV cache shape calculation."""
        shape = AscendTreeAttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=64,
        )
        assert shape == (2, 100, 16, 8, 64)

    def test_kv_cache_shape_invalid_block_size(self):
        """Test invalid block size raises error."""
        with pytest.raises(ValueError, match="Block size must be a multiple of 16"):
            AscendTreeAttentionBackend.get_kv_cache_shape(
                num_blocks=100,
                block_size=15,  # not a multiple of 16
                num_kv_heads=8,
                head_size=64,
            )


class TestAscendTreeAttentionMetadata:
    """Tests for AscendTreeAttentionMetadata."""

    def test_metadata_creation(self):
        """Test metadata creation."""
        metadata = AscendTreeAttentionMetadata(
            num_actual_tokens=10,
            num_decode_tokens=5,
            num_prefills=1,
            num_decodes=1,
            max_query_len=5,
            query_start_loc=torch.tensor([0, 5, 10]),
            max_seq_len=100,
            seq_lens=torch.tensor([50, 50]),
            block_tables=torch.zeros(2, 10, dtype=torch.int32),
            slot_mapping=torch.zeros(10, dtype=torch.int64),
            tree_attn_bias=torch.zeros(8, 8),
            tree_attn_mask=torch.zeros(PAD_SIZE, PAD_SIZE, dtype=torch.int8),
        )

        assert metadata.num_actual_tokens == 10
        assert metadata.num_decode_tokens == 5
        assert metadata.tree_attn_bias is not None
        assert metadata.tree_attn_mask is not None


class TestConvertFunction:
    """Tests for edge cases in mask conversion."""

    def test_binary_tree_depth_3(self):
        """Test binary tree with depth 3."""
        tree_choices = [
            (0,),
            (0, 0), (0, 1),
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        ]
        depth_counts = _get_depth_counts(tree_choices)
        gpu_mask = _prepare_tree_attn_bias_gpu(
            tree_choices, depth_counts, dtype=torch.float32, device=torch.device("cpu")
        )
        npu_mask = _convert_tree_mask_for_npu(gpu_mask)

        # Verify tree_len
        tree_len = 1 + len(tree_choices)
        assert gpu_mask.shape[0] == tree_len

        # Verify all ancestor relationships (child -> parent direction)
        for i, choice in enumerate(tree_choices):
            row = i + 1
            # Should attend to root
            assert npu_mask[row, 0] == 0
            # Should attend to self
            assert npu_mask[row, row] == 0

            # Verify ancestor chain (only child -> parent)
            for j, other_choice in enumerate(tree_choices):
                col = j + 1
                if _is_ancestor(other_choice, choice):
                    # other_choice is ancestor of choice, choice should attend
                    assert npu_mask[row, col] == 0, (
                        f"Node {row} ({choice}) should attend to ancestor {col} ({other_choice})"
                    )
                elif choice != other_choice and not _is_ancestor(choice, other_choice):
                    # Neither ancestor nor descendant, should not attend
                    assert npu_mask[row, col] == 1, (
                        f"Node {row} ({choice}) should NOT attend to {col} ({other_choice})"
                    )

    def test_single_token_tree(self):
        """Test tree with a single draft token."""
        tree_choices = [(0,)]
        depth_counts = _get_depth_counts(tree_choices)
        gpu_mask = _prepare_tree_attn_bias_gpu(
            tree_choices, depth_counts, dtype=torch.float32, device=torch.device("cpu")
        )
        npu_mask = _convert_tree_mask_for_npu(gpu_mask)

        # tree_len = 2 (root + 1 draft)
        assert gpu_mask.shape == (2, 2)
        # Root attends to itself (diagonal) and root (column 0)
        assert npu_mask[0, 0] == 0
        # Draft attends to root (column 0) and self (diagonal)
        assert npu_mask[1, 0] == 0
        assert npu_mask[1, 1] == 0


if __name__ == "__main__":
    if HAS_PYTEST:
        pytest.main([__file__, "-v"])
    else:
        print("=" * 60)
        print("Running Tree Attention Unit Tests")
        print("=" * 60)

        # Test 1: _is_ancestor
        print("\nTest 1: _is_ancestor")
        assert _is_ancestor((0,), (0, 0)) is True
        assert _is_ancestor((0, 0), (0,)) is False
        assert _is_ancestor((0,), (0,)) is False
        print("PASS: _is_ancestor")

        # Test 2: _get_depth_counts
        print("\nTest 2: _get_depth_counts")
        tree_choices = [(0,), (0, 0), (0, 1), (0, 0, 0), (0, 1, 0)]
        depth_counts = _get_depth_counts(tree_choices)
        assert depth_counts == [1, 2, 2]
        print("PASS: _get_depth_counts")

        # Test 3: _prepare_tree_attn_bias_gpu
        print("\nTest 3: _prepare_tree_attn_bias_gpu")
        tree_choices = [(0,), (0, 0), (0, 1)]
        depth_counts = _get_depth_counts(tree_choices)
        mask = _prepare_tree_attn_bias_gpu(
            tree_choices, depth_counts, dtype=torch.float32, device=torch.device("cpu")
        )
        assert mask.shape == (4, 4)
        assert torch.all(mask[:, 0] == 0)  # All attend to root
        for i in range(4):
            assert mask[i, i] == 0  # Diagonal
        assert mask[2, 1] == 0  # (0,0) attend (0,)
        assert mask[3, 2] == float('-inf')  # (0,1) NOT attend (0,0)
        print("PASS: _prepare_tree_attn_bias_gpu")

        # Test 4: _convert_tree_mask_for_npu
        print("\nTest 4: _convert_tree_mask_for_npu")
        npu_mask = _convert_tree_mask_for_npu(mask, pad_size=PAD_SIZE)
        assert npu_mask.shape == (PAD_SIZE, PAD_SIZE)
        assert npu_mask.dtype == torch.int8
        assert torch.all(npu_mask[:4, 0] == 0)  # Root attend
        print("PASS: _convert_tree_mask_for_npu")

        # Test 5: Backend name
        print("\nTest 5: AscendTreeAttentionBackend")
        assert AscendTreeAttentionBackend.get_name() == "TREE_ATTN"
        assert AscendTreeAttentionBackend.get_impl_cls() == AscendTreeAttentionImpl
        print("PASS: AscendTreeAttentionBackend")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
