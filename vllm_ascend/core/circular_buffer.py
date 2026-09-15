# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v0.27-compatible, single-page circular scratch ownership."""

from dataclasses import dataclass

from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, UniformTypeKVCacheSpecs


def prefix_cacheable(spec):
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return all(prefix_cacheable(member) for member in spec.kv_cache_specs.values())
    return bool(getattr(spec, "prefix_cacheable", True)) and bool(getattr(spec, "participates_in_prefix_caching", True))


# Upstream gained these properties after v0.27.1. Honor the existing GLM
# opt-out as well, without changing its manager or overwriting newer APIs.
if not hasattr(KVCacheSpec, "prefix_cacheable"):
    KVCacheSpec.prefix_cacheable = property(lambda self: getattr(self, "participates_in_prefix_caching", True))
if "prefix_cacheable" not in UniformTypeKVCacheSpecs.__dict__:
    UniformTypeKVCacheSpecs.prefix_cacheable = property(
        lambda self: all(prefix_cacheable(s) for s in self.kv_cache_specs.values())
    )


@dataclass(frozen=True, kw_only=True)
class AscendCircularBufferSpec(AttentionSpec):
    """A single packed plane, rather than AttentionSpec's default K+V pair."""

    @property
    def storage_block_size(self):
        return self.block_size

    @property
    def real_page_size_bytes(self):
        return self.block_size * self.num_kv_heads * self.head_size * self.dtype.itemsize

    @property
    def unpadded_page_size_bytes(self):
        """Expose the circular buffer's single-plane size to latest vLLM."""
        return self.real_page_size_bytes

    @property
    def prefix_cacheable(self):
        return False

    def max_memory_usage_bytes(self, vllm_config):
        return self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config, max_len):
        return 1

    def is_uniform_with_collection(self, specs):
        return all(type(s) is type(self) and s.block_size == self.block_size for s in specs.values())


def is_circular_spec(spec):
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return bool(spec.kv_cache_specs) and all(is_circular_spec(s) for s in spec.kv_cache_specs.values())
    return isinstance(spec, AscendCircularBufferSpec)


class AscendCircularBufferManager(FullAttentionManager):
    """One private page until free/preemption; no prefix hits or pruning."""

    supports_fine_grained_hash_lookup = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._record_new_block_ids = False

    def _claim_ring_block(self, request_id):
        blocks = self.req_to_blocks[request_id]
        if blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        blocks.extend(new_blocks)
        return new_blocks

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        return 0 if self.req_to_blocks.get(request_id) else 1

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        return self._claim_ring_block(request_id)

    def allocate_external_computed_blocks(self, request_id, num_local_computed_tokens, num_external_computed_tokens):
        self._claim_ring_block(request_id)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        return tuple([] for _ in kv_cache_group_ids), 0

    def cache_blocks(
        self,
        request,
        num_tokens,
        retention_interval=None,
        *,
        replay_boundary=None,
    ):
        pass

    def add_local_computed_blocks(
        self,
        request_id,
        new_computed_blocks,
        num_local_computed_tokens,
        num_external_computed_tokens,
    ):
        pass

    def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
        pass

    def get_num_common_prefix_blocks(self, running_request_id):
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens):
        return 0
