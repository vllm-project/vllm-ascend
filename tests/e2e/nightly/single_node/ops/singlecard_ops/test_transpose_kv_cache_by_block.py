import gc
import unittest

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

torch.set_printoptions(threshold=float("inf"))


def clone_kv_cache(k_caches, v_caches):
    new_k_caches = [cache.clone() for cache in k_caches]
    new_v_caches = [cache.clone() for cache in v_caches]
    return new_k_caches, new_v_caches


class TestTransposeKvCacheByBlock(unittest.TestCase):
    def compute_golden(
        self, k_caches, v_caches, block_ids_tensor, block_size, num_kv_head, head_dim, num_need_pulls, layers, dtype
    ):
        num_blocks = block_ids_tensor.shape[0]

        block_ids_tensor = block_ids_tensor.to(dtype=torch.int32)
        block_offsets = torch.arange(0, block_size, dtype=torch.int32).npu()
        slot_mapping = block_offsets.reshape((1, block_size)) + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        slot_mapping = slot_mapping.flatten()
        block_len = num_blocks * block_size
        block_len_tensor = torch.tensor([block_len], dtype=torch.int32).npu()

        block_table = block_ids_tensor.view(1, -1)
        seq_start_tensor = torch.tensor([0], dtype=torch.int32).npu()

        k = torch.empty(block_len, num_kv_head, head_dim, dtype=dtype).npu()
        v = torch.empty(block_len, num_kv_head, head_dim, dtype=dtype).npu()

        for layer in range(layers):
            k_cache_layer = k_caches[layer]
            v_cache_layer = v_caches[layer]

            torch_npu.npu_gather_pa_kv_cache(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_offset=seq_start_tensor,
                key=k,
                value=v,
            )

            k = k.view(num_blocks, num_need_pulls, block_size, -1)
            k.transpose_(1, 2)
            k = k.contiguous().view(block_len, num_kv_head, -1)

            v = v.view(num_blocks, num_need_pulls, block_size, -1)
            v.transpose_(1, 2)
            v = v.contiguous().view(block_len, num_kv_head, -1)

            torch_npu.npu_scatter_pa_kv_cache(
                key=k,
                value=v,
                key_cache=k_cache_layer,
                value_cache=v_cache_layer,
                slot_mapping=slot_mapping,
                cache_mode="Norm",
            )
        del k, v

    def test_transpose_kv_cache_by_block(self):
        # (layers, block_num, block_size, num_kv_head, head_dim, num_need_pulls)
        test_cases = [
            (16, 128, 128, 4, 128, 4),
            (16, 128, 128, 4, 128, 2),
            (16, 128, 128, 4, 128, 1),
            (16, 128, 128, 8, 128, 8),
            (16, 128, 128, 8, 128, 4),
            (16, 128, 128, 8, 128, 2),
        ]
        dtypes = [torch.float16, torch.bfloat16]
        for dtype in dtypes:
            for layers, block_num, block_size, num_kv_head, head_dim, num_need_pulls in test_cases:
                with self.subTest(
                    dtype=dtype,
                    shape=f"({layers}, {block_num}, {block_size}, {num_kv_head}, {head_dim}, {num_need_pulls})",
                ):
                    k_caches = []
                    v_caches = []
                    block_id_num = 33
                    block_ids_tensor = torch.randperm(block_num, dtype=torch.int64, device="npu")[:block_id_num]
                    for i in range(layers):
                        kcache = torch.randn(block_num, block_size, num_kv_head, head_dim, dtype=dtype, device="npu")
                        vcache = torch.randn(block_num, block_size, num_kv_head, head_dim, dtype=dtype, device="npu")
                        k_caches.append(kcache)
                        v_caches.append(vcache)

                    cloned_k_caches, cloned_v_caches = clone_kv_cache(k_caches, v_caches)
                    self.compute_golden(
                        cloned_k_caches,
                        cloned_v_caches,
                        block_ids_tensor,
                        block_size,
                        num_kv_head,
                        head_dim,
                        num_need_pulls,
                        layers,
                        dtype,
                    )
                    torch.ops._C_ascend.transpose_kv_cache_by_block(
                        k_caches, v_caches, block_ids_tensor, block_size, num_kv_head, head_dim, num_need_pulls, layers
                    )

                    for i in range(layers):
                        self.assert_tensors_almost_equal(k_caches[i], cloned_k_caches[i], dtype)
                        self.assert_tensors_almost_equal(v_caches[i], cloned_v_caches[i], dtype)
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    def test_block_strides_preserve_storage(self):
        # Cover full-load and split-block kernels, including a split tail.
        for dtype in (torch.float16, torch.bfloat16):
            for block_size, num_heads in ((16, 4), (128, 8), (127, 8)):
                for layouts in ((1, 1), (2, 2), (3, 3), (1, 2, 3)):
                    with self.subTest(dtype=dtype, block_size=block_size, num_heads=num_heads, layouts=layouts):
                        block_num, head_dim, split_num = 5, 128, 2
                        block_ids = [4, 1]
                        k_caches, v_caches, backings, expected_backings = [], [], [], []
                        expected_k, expected_v = [], []
                        for stride_factor in layouts:
                            # Guard blocks on both ends also exercise nonzero storage_offset.
                            if stride_factor == 2:
                                backing = torch.randn(
                                    block_num + 2, 2, block_size, num_heads, head_dim, dtype=dtype, device="npu"
                                )
                                expected = backing.cpu()
                                backings.append(backing)
                                expected_backings.append(expected)
                                k_caches.append(backing[1:-1, 0])
                                v_caches.append(backing[1:-1, 1])
                                expected_k.append(expected[1:-1, 0])
                                expected_v.append(expected[1:-1, 1])
                            else:
                                # K/V may have different physical strides; gaps must stay untouched.
                                for caches, expected_caches, factor in (
                                    (k_caches, expected_k, stride_factor),
                                    (v_caches, expected_v, 1 if stride_factor == 1 else 4),
                                ):
                                    backing = torch.randn(
                                        block_num + 2, factor, block_size, num_heads, head_dim, dtype=dtype, device="npu"
                                    )
                                    expected = backing.cpu()
                                    backings.append(backing)
                                    expected_backings.append(expected)
                                    caches.append(backing[1:-1, factor - 1])
                                    expected_caches.append(expected[1:-1, factor - 1])

                        for cache in expected_k + expected_v:
                            for block_id in block_ids:
                                selected = cache[block_id].clone()
                                cache[block_id].copy_(
                                    selected.reshape(split_num, block_size, -1).transpose(0, 1).reshape_as(selected)
                                )
                        views = k_caches + v_caches
                        metadata = [(cache.data_ptr(), cache.stride(), cache.storage_offset()) for cache in views]
                        ids = torch.tensor(block_ids, dtype=torch.int64, device="npu")
                        torch.ops._C_ascend.transpose_kv_cache_by_block(
                            k_caches, v_caches, ids, block_size, num_heads, head_dim, split_num, len(layouts)
                        )
                        torch.npu.synchronize()

                        # Compare whole allocations, including other K/V views, gaps and guard blocks.
                        for actual, expected in zip(backings, expected_backings):
                            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                        self.assertEqual(
                            metadata, [(cache.data_ptr(), cache.stride(), cache.storage_offset()) for cache in views]
                        )

    def test_rejects_noncontiguous_block_payload(self):
        backing = torch.randn(3, 16, 4, 256, dtype=torch.float16, device="npu")
        cache = backing[..., ::2]
        before = backing.cpu()
        ids = torch.tensor([1], dtype=torch.int64, device="npu")
        with self.assertRaisesRegex(RuntimeError, "contiguous within each block"):
            torch.ops._C_ascend.transpose_kv_cache_by_block([cache], [cache], ids, 16, 4, 128, 2, 1)
        torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)

    def test_empty_block_ids_preserve_cache(self):
        backing = torch.randn(3, 2, 16, 4, 128, dtype=torch.float16, device="npu")
        before = backing.cpu()
        ids = torch.empty(0, dtype=torch.int64, device="npu")
        torch.ops._C_ascend.transpose_kv_cache_by_block([backing[:, 0]], [backing[:, 1]], ids, 16, 4, 128, 2, 1)
        torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)

    def assert_tensors_almost_equal(self, actual, expected, dtype):
        """Check if two tensors are approximately equal (considering floating point errors)"""
        self.assertEqual(actual.shape, expected.shape, "Shape mismatch")

        # Check for NaN
        self.assertFalse(torch.isnan(actual).any(), "Actual result contains NaN")
        self.assertFalse(torch.isnan(expected).any(), "Expected result contains NaN")

        # Check for Inf
        self.assertFalse(torch.isinf(actual).any(), "Actual result contains Inf")
        self.assertFalse(torch.isinf(expected).any(), "Expected result contains Inf")

        # Set different tolerances based on data type
        if dtype == torch.float16:
            rtol, atol = 1e-5, 1e-5
        else:  # bfloat16
            rtol, atol = 1.5e-5, 1.5e-5

        # Compare values
        diff = torch.abs(actual - expected)
        max_diff = diff.max().item()
        max_expected = torch.abs(expected).max().item()

        # Check relative and absolute errors
        if max_expected > 0:
            relative_diff = max_diff / max_expected
            self.assertLessEqual(
                relative_diff,
                rtol,
                f"Relative error too large: {relative_diff} > {rtol}. Max difference: {max_diff}",
            )

        self.assertLessEqual(max_diff, atol, f"Absolute error too large: {max_diff} > {atol}")
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()
