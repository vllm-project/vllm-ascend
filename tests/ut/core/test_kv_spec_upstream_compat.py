"""Regression tests for upstream KV cache spec compatibility.

Upstream vLLM main declares storage_block_size as a dataclass field on
MLAAttentionSpec, drops the Ascend-only indexes_kv_by_block_stride flag and
renames KVCacheTensor.shared_by to layers; the Ascend spec classes and the
patched unify_kv_cache_spec_page_size must tolerate all three, and kpool
indexer/tail pages must pad instead of raising NotImplementedError.
"""

import dataclasses

import torch

from tests.ut.base import TestBase
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


class TestKVSpecUpstreamCompat(TestBase):
    def test_spec_constructs_with_storage_block_size_field(self):
        """The upstream dataclass __init__ assigns storage_block_size.

        Before the setter was added, constructing the spec raised
        'property has no setter'.
        """
        spec = AscendMLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            storage_block_size=128,
        )
        # Explicit override wins over the computed layout.
        self.assertEqual(spec.storage_block_size, 128)

    def test_spec_computed_layout_without_override(self):
        """Without an explicit value the computed Ascend layout is kept."""
        spec = AscendMLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        )
        self.assertEqual(spec.compress_ratio, 1)
        self.assertEqual(spec.storage_block_size, spec.block_size // spec.compress_ratio)

    def test_spec_replaces_with_indexes_kv_by_block_stride(self):
        """replace() round-trips the Ascend-only flag on our spec."""
        spec = AscendMLAAttentionSpec(
            block_size=512,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        )
        new = dataclasses.replace(spec, indexes_kv_by_block_stride=True)
        self.assertTrue(new.indexes_kv_by_block_stride)

    def test_shared_by_aliases_layers(self):
        """KVCacheTensor.shared_by reads the upstream 'layers' field."""
        from vllm.v1.kv_cache_interface import KVCacheTensor

        tensor = KVCacheTensor(size=1024, layers=["a.weight"], layer_stride=1, block_stride=1)
        self.assertEqual(tensor.shared_by, ["a.weight"])

    def test_unify_pads_indexer_pages(self):
        """kpool indexer (MLA-family) pages pad instead of raising.

        Sizes chosen so the indexer page does NOT divide the main page:
        96 * 2 bytes per state -> 768-byte pages vs a 524288-byte main page.
        """
        from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        main = MLAAttentionSpec(block_size=512, num_kv_heads=1, head_size=512, dtype=torch.bfloat16)
        indexer = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=96, dtype=torch.bfloat16)
        self.assertNotEqual(main.page_size_bytes % indexer.page_size_bytes, 0)
        unified = unify_kv_cache_spec_page_size(
            {"model.layers.0.self_attn.attn": main, "model.layers.0.indexer": indexer}
        )
        self.assertEqual(unified["model.layers.0.indexer"].page_size_bytes, main.page_size_bytes)
