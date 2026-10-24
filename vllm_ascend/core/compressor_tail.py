# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed private state pages for the DeepSeek-V4 CYCLE compressor."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec


@dataclass(frozen=True, kw_only=True)
class AscendCompressorTailSpec(AscendSlidingWindowMLASpec):
    """Keep the current storage-page layout but bound it to a private ring."""

    tail_tokens: int
    ring_blocks_per_request: int

    def __post_init__(self) -> None:
        if self.dtype != torch.float32:
            raise ValueError("Compressor tails require FP32 state storage.")
        if self.storage_block_size <= 0 or self.tail_tokens <= 0:
            raise ValueError("Compressor tail and storage page must be positive.")
        if self.ring_blocks_per_request != cdiv(self.tail_tokens, self.storage_block_size):
            raise ValueError("Ring page count must cover exactly the persistent tail.")

    @property
    def is_circular(self) -> bool:
        return True

    @property
    def prefix_cacheable(self) -> bool:
        return False

    @property
    def uses_slot_mapping(self) -> bool:
        return False

    @property
    def block_table_token_alignment(self) -> int | None:
        return None

    def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
        return self.ring_blocks_per_request

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        return self.ring_blocks_per_request * self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        return self.ring_blocks_per_request

    def is_uniform_with_collection(self, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        # Main and indexer states share a block table, with different page
        # widths in the tuple. Uniformity concerns their allocation policy.
        return all(
            type(spec) is type(self)
            and spec.block_size == self.block_size
            and spec.storage_block_size == self.storage_block_size
            and spec.tail_tokens == self.tail_tokens
            and spec.ring_blocks_per_request == self.ring_blocks_per_request
            and spec.dtype == self.dtype
            and spec.model_version == self.model_version
            and spec.prefix_cacheable == self.prefix_cacheable
            and spec.uses_slot_mapping == self.uses_slot_mapping
            for spec in kv_cache_specs.values()
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        if not specs or not all(isinstance(spec, cls) and spec == specs[0] for spec in specs):
            raise ValueError("Compressor tails in one group must have identical layouts.")
        return specs[0]


class CompressorTailManager(CircularBufferManager):
    """Reuse current ring lifecycle and deferred-free support for multiple pages."""

    kv_cache_spec: AscendCompressorTailSpec

    @property
    def has_positionally_stable_blocks(self) -> bool:
        return False

    def _claim_ring_block(self, request_id: str) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        missing = self.kv_cache_spec.ring_blocks_per_request - len(req_blocks)
        assert missing >= 0
        if not missing:
            return []
        new_blocks = self.block_pool.get_new_blocks(missing)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        assert not new_computed_blocks, "Compressor state cannot adopt shared prefix pages."
        return self.kv_cache_spec.ring_blocks_per_request - len(self.req_to_blocks.get(request_id, ()))

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks and not num_external_computed_tokens
        self.num_cached_block[request_id] = 0

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not num_external_computed_tokens, "External compressor state restore is unsupported."
        self._claim_ring_block(request_id)
