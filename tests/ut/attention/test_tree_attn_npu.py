# SPDX-License-Identifier: Apache-2.0
# Unit tests for AscendTreeAttentionBackend

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
    """测试 GPU → NPU mask 转换。"""

    def test_is_ancestor(self):
        """测试祖先关系判断。"""
        # (0,) 是 (0, 0) 的祖先
        assert _is_ancestor((0,), (0, 0)) is True
        # (0, 0) 不是 (0,) 的祖先
        assert _is_ancestor((0, 0), (0,)) is False
        # 同一个节点不是自己的祖先
        assert _is_ancestor((0,), (0,)) is False
        # (0, 1) 不是 (0, 0) 的祖先
        assert _is_ancestor((0, 1), (0, 0)) is False
        # (0,) 是 (0, 0, 0) 的祖先
        assert _is_ancestor((0,), (0, 0, 0)) is True
        # (0, 0) 是 (0, 0, 0) 的祖先
        assert _is_ancestor((0, 0), (0, 0, 0)) is True

    def test_get_depth_counts(self):
        """测试深度计数。"""
        tree_choices = [(0,), (0, 0), (0, 1), (0, 0, 0), (0, 1, 0)]
        depth_counts = _get_depth_counts(tree_choices)
        assert depth_counts == [1, 2, 2]

    def test_prepare_tree_attn_bias_gpu(self):
        """测试 GPU tree mask 构建。"""
        tree_choices = [(0,), (0, 0), (0, 1)]
        depth_counts = _get_depth_counts(tree_choices)
        mask = _prepare_tree_attn_bias_gpu(
            tree_choices, depth_counts, dtype=torch.float32, device=torch.device("cpu")
        )

        # tree_len = 4 (root + 3 drafts)
        assert mask.shape == (4, 4)

        # 所有 token 应该 attend 到 root (column 0)
        assert torch.all(mask[:, 0] == 0)

        # 每个 token 应该 attend 到自己
        for i in range(4):
            assert mask[i, i] == 0

        # (0, 0) 应该 attend 到 (0,) - 祖先关系 (child → parent)
        assert mask[2, 1] == 0

        # (0, 1) 不应该 attend 到 (0, 0) - 不同分支
        assert mask[3, 2] == float('-inf')

    def test_convert_tree_mask_for_npu(self):
        """测试 GPU → NPU mask 转换。"""
        tree_choices = [(0,), (0, 0), (0, 1)]
        depth_counts = _get_depth_counts(tree_choices)
        gpu_mask = _prepare_tree_attn_bias_gpu(
            tree_choices, depth_counts, dtype=torch.float32, device=torch.device("cpu")
        )

        npu_mask = _convert_tree_mask_for_npu(gpu_mask, pad_size=PAD_SIZE)

        # 验证 shape 和 dtype
        assert npu_mask.shape == (PAD_SIZE, PAD_SIZE)
        assert npu_mask.dtype == torch.int8

        # 验证转换正确性
        tree_len = gpu_mask.shape[0]

        # Root 应该全是 0（attend）
        assert torch.all(npu_mask[:tree_len, 0] == 0)

        # 对角线应该是 0
        for i in range(tree_len):
            assert npu_mask[i, i] == 0

        # (0, 0) 应该 attend 到 (0,) → npu_mask[2, 1] == 0
        assert npu_mask[2, 1] == 0

        # (0, 1) 不应该 attend 到 (0, 0) → npu_mask[3, 2] == 1
        assert npu_mask[3, 2] == 1

        # padding 区域应该是 1（block）
        assert torch.all(npu_mask[tree_len:, :] == 1)
        assert torch.all(npu_mask[:, tree_len:] == 1)


class TestAscendTreeAttentionBackend:
    """测试 AscendTreeAttentionBackend。"""

    def test_backend_name(self):
        """测试 backend 名称。"""
        assert AscendTreeAttentionBackend.get_name() == "TREE_ATTN"

    def test_backend_classes(self):
        """测试 backend 返回的类。"""
        assert AscendTreeAttentionBackend.get_impl_cls() == AscendTreeAttentionImpl
        assert (
            AscendTreeAttentionBackend.get_builder_cls()
            == AscendTreeAttentionMetadataBuilder
        )

    def test_kv_cache_shape(self):
        """测试 KV cache shape 计算。"""
        shape = AscendTreeAttentionBackend.get_kv_cache_shape(
            num_blocks=100,
            block_size=16,
            num_kv_heads=8,
            head_size=64,
        )
        assert shape == (2, 100, 16, 8, 64)

    def test_kv_cache_shape_invalid_block_size(self):
        """测试无效的 block size。"""
        with pytest.raises(ValueError, match="Block size must be a multiple of 16"):
            AscendTreeAttentionBackend.get_kv_cache_shape(
                num_blocks=100,
                block_size=15,  # 不是 16 的倍数
                num_kv_heads=8,
                head_size=64,
            )


class TestAscendTreeAttentionMetadata:
    """测试 AscendTreeAttentionMetadata。"""

    def test_metadata_creation(self):
        """测试 metadata 创建。"""
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
    """测试转换函数的边界情况。"""

    def test_binary_tree_depth_3(self):
        """测试二叉树深度 3 的转换。"""
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

        # 验证 tree_len
        tree_len = 1 + len(tree_choices)
        assert gpu_mask.shape[0] == tree_len

        # 验证所有祖先关系 (child → parent 方向)
        for i, choice in enumerate(tree_choices):
            row = i + 1
            # 应该 attend 到 root
            assert npu_mask[row, 0] == 0
            # 应该 attend 到自己
            assert npu_mask[row, row] == 0

            # 验证祖先链 (只验证 child → parent)
            for j, other_choice in enumerate(tree_choices):
                col = j + 1
                if _is_ancestor(other_choice, choice):
                    # other_choice 是 choice 的祖先，choice 应该 attend other_choice
                    assert npu_mask[row, col] == 0, (
                        f"Node {row} ({choice}) should attend to ancestor {col} ({other_choice})"
                    )
                elif choice != other_choice and not _is_ancestor(choice, other_choice):
                    # 不是祖先也不是后代，不应该 attend
                    assert npu_mask[row, col] == 1, (
                        f"Node {row} ({choice}) should NOT attend to {col} ({other_choice})"
                    )

    def test_single_token_tree(self):
        """测试只有一个 draft token 的 tree。"""
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
        # 手动运行测试
        print("=" * 60)
        print("运行 Tree Attention 单元测试")
        print("=" * 60)

        # Test 1: _is_ancestor
        print("\n测试 1: _is_ancestor")
        assert _is_ancestor((0,), (0, 0)) is True
        assert _is_ancestor((0, 0), (0,)) is False
        assert _is_ancestor((0,), (0,)) is False
        print("✓ _is_ancestor 测试通过")

        # Test 2: _get_depth_counts
        print("\n测试 2: _get_depth_counts")
        tree_choices = [(0,), (0, 0), (0, 1), (0, 0, 0), (0, 1, 0)]
        depth_counts = _get_depth_counts(tree_choices)
        assert depth_counts == [1, 2, 2]
        print("✓ _get_depth_counts 测试通过")

        # Test 3: _prepare_tree_attn_bias_gpu
        print("\n测试 3: _prepare_tree_attn_bias_gpu")
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
        print("✓ _prepare_tree_attn_bias_gpu 测试通过")

        # Test 4: _convert_tree_mask_for_npu
        print("\n测试 4: _convert_tree_mask_for_npu")
        npu_mask = _convert_tree_mask_for_npu(mask, pad_size=PAD_SIZE)
        assert npu_mask.shape == (PAD_SIZE, PAD_SIZE)
        assert npu_mask.dtype == torch.int8
        assert torch.all(npu_mask[:4, 0] == 0)  # Root attend
        print("✓ _convert_tree_mask_for_npu 测试通过")

        # Test 5: Backend name
        print("\n测试 5: AscendTreeAttentionBackend")
        assert AscendTreeAttentionBackend.get_name() == "TREE_ATTN"
        assert AscendTreeAttentionBackend.get_impl_cls() == AscendTreeAttentionImpl
        print("✓ AscendTreeAttentionBackend 测试通过")

        print("\n" + "=" * 60)
        print("全部测试通过！")
        print("=" * 60)
