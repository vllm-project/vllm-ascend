import unittest

import torch
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    get_host_device_memory_usage_ratio,
)


class TestHostDeviceMemoryRatio(unittest.TestCase):
    def gen_kv_cache_specs(
        self,
        spec_type: type[KVCacheSpec],
        num: int,
        name_suffix: str,
        **kwargs,
    ):
        kv_cache_specs = {}
        for layer_id in range(num):
            name = f"model.layers.{layer_id}.{name_suffix}"
            spec = spec_type(**kwargs)
            kv_cache_specs[name] = spec
        return kv_cache_specs

    def test_normal_case(self):
        # same number of mla & indexer cache, no c8
        # gold = kv_dim / idx_dim
        #      = 576 / 128
        #      = 4.5
        hd_memory_ratio_gold = 4.5
        mla_specs = self.gen_kv_cache_specs(
            AscendMLAAttentionSpec,
            4,
            "self_attn.attn",
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            cache_sparse_sfa_c8=False,
            store_on_host=True,
        )
        indexer_specs = self.gen_kv_cache_specs(
            AscendSFAIndexerCacheSpec,
            4,
            "self_attn.indexer.k_cache",
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            scale_dim=0,
            scale_dtype=torch.int8,
            cache_sparse_li_c8=False,
        )
        all_specs = {}
        all_specs.update(mla_specs)
        all_specs.update(indexer_specs)
        self.assertEqual(get_host_device_memory_usage_ratio(all_specs), hd_memory_ratio_gold)

    def test_indexer_share(self):
        # multi layer share one indexer (GLM5.2), no c8
        # gold = (kv_num * kv_dim) / (idx_num * idx_dim)
        #      = (4 * 576) / (1 * 128)
        #      = 18
        hd_memory_ratio_gold = 18
        mla_specs = self.gen_kv_cache_specs(
            AscendMLAAttentionSpec,
            4,
            "self_attn.attn",
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            cache_sparse_sfa_c8=False,
            store_on_host=True,
        )
        indexer_specs = self.gen_kv_cache_specs(
            AscendSFAIndexerCacheSpec,
            1,
            "self_attn.indexer.k_cache",
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            scale_dim=0,
            scale_dtype=torch.int8,
            cache_sparse_li_c8=False,
        )
        all_specs = {}
        all_specs.update(mla_specs)
        all_specs.update(indexer_specs)
        self.assertEqual(get_host_device_memory_usage_ratio(all_specs), hd_memory_ratio_gold)

    def test_li_c8(self):
        # indxer c8 cache
        # gold = (kv_num * kv_dim * bf16) / (idx_num * (idx_dim * int8 + scale_dim * fp16))
        #      = (4 * 576 * 2) / (1 * (128 * 1 + 1 * 2))
        #      = 35.44615384615385
        hd_memory_ratio_gold = 35.44615384615385
        mla_specs = self.gen_kv_cache_specs(
            AscendMLAAttentionSpec,
            4,
            "self_attn.attn",
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            cache_sparse_sfa_c8=False,
            store_on_host=True,
        )
        indexer_specs = self.gen_kv_cache_specs(
            AscendSFAIndexerCacheSpec,
            1,
            "self_attn.indexer.k_cache",
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            cache_dtype_str="bfloat16",
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
        )
        all_specs = {}
        all_specs.update(mla_specs)
        all_specs.update(indexer_specs)
        self.assertEqual(get_host_device_memory_usage_ratio(all_specs), hd_memory_ratio_gold)

    def test_no_offload(self):
        # no store_on_host specs
        # gold = 0 (don't need host memory)
        hd_memory_ratio_gold = 0
        mla_specs = self.gen_kv_cache_specs(
            AscendMLAAttentionSpec,
            4,
            "self_attn.attn",
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            cache_sparse_sfa_c8=False,
            store_on_host=False,
        )
        indexer_specs = self.gen_kv_cache_specs(
            AscendSFAIndexerCacheSpec,
            1,
            "self_attn.indexer.k_cache",
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_dtype_str="bfloat16",
            scale_dim=0,
            scale_dtype=torch.int8,
            cache_sparse_li_c8=False,
        )
        all_specs = {}
        all_specs.update(mla_specs)
        all_specs.update(indexer_specs)
        self.assertEqual(get_host_device_memory_usage_ratio(all_specs), hd_memory_ratio_gold)


if __name__ == "__main__":
    unittest.main()
