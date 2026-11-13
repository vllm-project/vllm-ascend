# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import HiSparseResidentManager
from vllm.v1.kv_cache_interface import HiSparseResidentSpec
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_utils import KVCacheBlock


class AscendStoreHiSparseResidentManager(HiSparseResidentManager):
    """Use bounded Resident admission when AscendStore imports a prefix."""

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence["KVCacheBlock"],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        if apply_admission_cap and total_computed_tokens > num_local_computed_tokens:
            # Core constructs this manager with the same spec-derived cap it
            # uses for startup sizing. Bound the uncached suffix by that cap
            # and let Core account for the writable imported tail separately.
            imported_pages = cdiv(total_computed_tokens, self.block_size)
            existing_pages = len(self.req_to_blocks.get(request_id, ()))
            limit_pages = max(imported_pages, existing_pages) + self.max_admission_blocks_per_request
            num_tokens = min(num_tokens, limit_pages * self.block_size)

        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
            prefill_end=prefill_end,
        )


@register_kv_cache_spec(
    manager_class=AscendStoreHiSparseResidentManager,
    uniform_type_base_spec=HiSparseResidentSpec,
)
@dataclass(frozen=True)
class AscendStoreHiSparseResidentSpec(HiSparseResidentSpec):
    """Keep Core's Resident geometry and prefill-window cap."""

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        window_bytes = super().max_memory_usage_bytes(vllm_config)
        max_context_pages = cdiv(vllm_config.model_config.max_model_len, self.block_size)
        # The writable import tail can coexist with a full prefill window.
        return min(window_bytes + self.page_size_bytes, max_context_pages * self.page_size_bytes)
