import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.c8_k_cache_dtype = torch.int8
        runner.c8_k_scale_cache_dtype = torch.float16
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.ascend_config = MagicMock()
        runner.ascend_config.is_sparse_c8_layer.return_value = False
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))

    def test_sparse_mla_kv_cache_without_sparse_c8_uses_three_buffers(self):
        from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec

        runner = self._build_runner()
        runner.use_sparse = True
        runner.model_config.hf_text_config = MagicMock(index_head_dim=32)
        kv_cache_spec = AscendMLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=56,
            sparse_head_dim=(16, 8, 32),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
            cache_sparse_c8=False,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["sparse_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["sparse_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        self.assertEqual(len(kv_cache_raw_tensors["sparse_attn"]), 3)

        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["sparse_attn"],
            )
        ]
        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)

        self.assertEqual(len(kv_caches["sparse_attn"]), 3)
        self.assertEqual(kv_caches["sparse_attn"][2].dtype, torch.bfloat16)

    def test_sparse_mla_kv_cache_with_sparse_c8_uses_four_buffers(self):
        from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec

        runner = self._build_runner()
        runner.use_sparse = True
        runner.use_sparse_c8_indexer = True
        runner.model_config.hf_text_config = MagicMock(index_head_dim=32)
        kv_cache_spec = AscendMLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=56,
            sparse_head_dim=(16, 8, 32),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
            cache_sparse_c8=True,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["sparse_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["sparse_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        self.assertEqual(len(kv_cache_raw_tensors["sparse_attn"]), 4)

        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["sparse_attn"],
            )
        ]
        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)

        self.assertEqual(len(kv_caches["sparse_attn"]), 4)
        self.assertEqual(kv_caches["sparse_attn"][2].dtype, torch.int8)
        self.assertEqual(kv_caches["sparse_attn"][3].dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
