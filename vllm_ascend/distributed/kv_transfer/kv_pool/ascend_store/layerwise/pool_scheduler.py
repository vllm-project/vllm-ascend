from collections.abc import Sequence
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_backend_class,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.dspark_prefix_cache import DSparkPrefixKeys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    build_layerwise_cache_layout,
    build_layerwise_reuse_layout,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    block_hash_to_str,
    get_block_hashes,
    get_group_block_size,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler


class LayerwiseKVPoolScheduler(KVPoolScheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        use_layerwise: bool,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        backend_name = str(extra_config.get("backend", "mooncake")).lower()
        self.layerwise_backend = get_backend_class(backend_name)
        self.layerwise_data_plane = (
            self.layerwise_backend.layerwise_data_plane if self.layerwise_backend is not None else None
        )
        self.use_block_key_layerwise = use_layerwise and self.layerwise_data_plane == "block_key"
        self.use_layerwise_transfer = use_layerwise and self.layerwise_data_plane == "gva"
        self.dspark_prefix_keys: DSparkPrefixKeys | None = None
        self.dspark_draft_only_group_ids = frozenset(getattr(kv_cache_config, "dspark_draft_only_group_ids", ()))
        super().__init__(vllm_config, use_layerwise, kv_cache_config)
        cacheable_block_sizes = [self.original_block_size[i] for i in self.cacheable_group_ids]
        if (
            self.use_layerwise
            and self.backend_name != "memcache"
            and not self.use_block_key_layerwise
            and len(cacheable_block_sizes) != len(self.original_block_size)
        ):
            raise ValueError("AscendStore private KV state requires Mooncake block-key or non-layerwise transfer")

    def _use_hybrid(self, vllm_config: VllmConfig, kv_cache_groups: list[Any] | None) -> bool:
        return super()._use_hybrid(vllm_config, kv_cache_groups) or (
            self.use_block_key_layerwise and kv_cache_groups is not None and len(kv_cache_groups) > 1
        )

    def _init_layerwise_runtime(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> None:
        self.block_key_hybrid = self.use_block_key_layerwise and self.use_hybrid
        self.layerwise_keys = (
            self.layerwise_backend.bind_layerwise_keys(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                model_name=vllm_config.model_config.model.split("/")[-1],
                use_hybrid=self.use_hybrid,
                grouped_block_size=self.grouped_block_size,
            )
            if self.layerwise_backend is not None and self.layerwise_data_plane is not None
            else None
        )
        if self.use_block_key_layerwise and self.tp_mismatch:
            raise ValueError("Mooncake layerwise does not yet support prefill/decode TP mismatch")
        self.layerwise_max_transfer_blocks = int(
            vllm_config.kv_transfer_config.kv_connector_extra_config.get("layerwise_max_transfer_blocks", 0)
        )

    def _init_layerwise_offload(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None) -> None:
        self.layerwise_offload = False
        if self.use_layerwise_transfer:
            extra_config = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
            if kv_cache_config is not None and extra_config is not None:
                reuse_layout = build_layerwise_reuse_layout(
                    get_layerwise_kv_cache_specs(kv_cache_config),
                    self.num_layers,
                    extra_config,
                )
                if reuse_layout.layer_cache_specs:
                    self.num_layers = max(
                        self.num_layers,
                        max(reuse_layout.layer_cache_specs) + 1,
                    )
                self.layerwise_offload = reuse_layout.has_layer_reuse
            else:
                self.layerwise_offload = build_layerwise_cache_layout(
                    self.num_layers,
                    vllm_config.kv_transfer_config.kv_connector_extra_config,
                ).has_layer_reuse

    def _lookup_external_hit_tokens(self, request: Request, num_computed_tokens: int) -> int | None:
        if not self.use_layerwise:
            return super()._lookup_external_hit_tokens(request, num_computed_tokens)
        prompt_token_len = len(request.prompt_token_ids)
        if self.use_block_key_layerwise:
            token_len = self._floor_to_cache_transfer_granularity(prompt_token_len)
            if token_len < self.cache_transfer_granularity:
                return None
            return self._get_block_key_layerwise_hit_tokens(request, token_len, num_computed_tokens)

        if self.use_layerwise_transfer:
            return self._get_layerwise_hit_tokens(request, prompt_token_len, num_computed_tokens)

        token_len = (
            self._floor_to_cache_transfer_granularity(prompt_token_len)
            if self._discard_partial_chunks
            else prompt_token_len
        )
        if token_len < self.cache_transfer_granularity:
            return None
        return self._get_store_lookup_hit_tokens(request, token_len, num_computed_tokens, include_layers=True)

    def _get_layerwise_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        self._get_or_create_request_tracker(request.request_id)
        if self.cache_coordinator is not None:
            return self._lookup_layerwise_with_coordinator(
                request, token_len, 0 if self.layerwise_offload else num_computed_tokens
            )
        return self._lookup_layerwise_contiguous(request, token_len, num_computed_tokens)

    def _get_block_key_layerwise_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        if self.block_key_hybrid:
            return self._lookup_layerwise_with_coordinator(request, token_len)
        return self._lookup_block_key_contiguous(request, token_len, num_computed_tokens)

    def _lookup_layerwise_with_coordinator(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int = 0,
    ) -> int:
        """Reachability-aware hit check for hybrid models.

        Only the blocks the KV cache managers consider reachable (sliding-
        window / compressor-state tails) are queried for reachability-limited
        groups — those groups are stored sparsely by the layerwise save path —
        and the hit length is derived by the coordinator lookup shared with
        the non-layerwise path, so it matches the store-side mask semantics.
        """
        coordinator = self.cache_coordinator
        assert coordinator is not None

        def query_group_hits(
            group_id: int,
            group_block_hashes: Sequence[BlockHash | str],
            lookup_mask: Sequence[bool] | None,
        ) -> list[BlockHash]:
            keys_by_block: list[list[str]] = []
            allowed_hashes: list[BlockHash] = []
            local_blocks = num_computed_tokens // self.grouped_block_size[group_id]
            for block_idx, block_hash in enumerate(group_block_hashes):
                if block_idx < local_blocks:
                    continue
                if lookup_mask is not None and not (block_idx < len(lookup_mask) and lookup_mask[block_idx]):
                    continue
                keys_by_block.append(self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(block_hash)))
                allowed_hashes.append(block_hash)
            block_hits = self._query_layerwise_block_hits(keys_by_block)
            return [block_hash for block_hash, hit in zip(allowed_hashes, block_hits) if hit]

        return coordinator.find_reachable_hit_tokens(
            request.block_hashes,
            token_len,
            query_group_hits,
            log_context=f"hit_check: req={request.request_id}",
            hbm_hit_tokens=num_computed_tokens,
        )

    def _lookup_layerwise_contiguous(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        # Layer reuse can evict earlier layers from HBM. Otherwise, only
        # probe blocks beyond the complete local prefix to avoid rewarming it.
        num_hash_blocks = token_len // self.hash_block_size
        block_hashes_to_check = request.block_hashes[:num_hash_blocks]
        hits_per_group: list[int] = []

        for group_id in range(len(self.grouped_block_size)):
            effective_block_size = get_group_block_size(self.grouped_block_size, group_id)
            group_block_hashes = get_block_hashes(block_hashes_to_check, effective_block_size, self.hash_block_size)
            query_start_block = (
                0
                if self.layerwise_offload
                else min(num_computed_tokens // effective_block_size, len(group_block_hashes))
            )
            group_block_hashes = group_block_hashes[query_start_block:]
            if not group_block_hashes:
                hits_per_group.append(query_start_block * effective_block_size)
                continue

            # Deliberately no cross-request hit cache: in a tiered (DRAM+SSD)
            # pool a cached hit can outlive an eviction and over-report
            # external hits, while a cached miss can hide a freshly saved
            # block. Every lookup re-probes backend readability instead.
            keys_by_block = [
                self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(bh)) for bh in group_block_hashes
            ]
            if not keys_by_block:
                continue
            block_hits = self._query_layerwise_block_hits(keys_by_block)
            num_hit_blocks = 0
            for hit in block_hits:
                if hit:
                    num_hit_blocks += 1
                else:
                    break

            hits_per_group.append((query_start_block + num_hit_blocks) * effective_block_size)

        if not hits_per_group:
            logger.debug(
                "hit_check: req=%s token_len=%d no participating groups (all skipped)",
                request.request_id,
                token_len,
            )
            return 0
        hit_tokens = min(hits_per_group)
        logger.debug(
            "hit_check: req=%s token_len=%d hits_per_group=%s hit_tokens=%d",
            request.request_id,
            token_len,
            hits_per_group,
            hit_tokens,
        )
        return hit_tokens

    def _lookup_block_key_contiguous(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        del num_computed_tokens
        num_hash_blocks = token_len // self.hash_block_size
        block_hashes = get_block_hashes(
            request.block_hashes[:num_hash_blocks],
            self._block_size,
            self.hash_block_size,
        )
        if not block_hashes:
            return 0
        keys_by_block = [
            self._make_layerwise_hit_check_keys(0, block_hash_to_str(block_hash)) for block_hash in block_hashes
        ]
        block_hits = self._query_layerwise_block_hits(keys_by_block)
        num_hit_blocks = 0
        for hit in block_hits:
            if not hit:
                break
            num_hit_blocks += 1
        logger.info(
            "Block-key layerwise hit check request=%s hit_blocks=%d/%d",
            request.request_id,
            num_hit_blocks,
            len(keys_by_block),
        )
        return num_hit_blocks * self._block_size

    def _make_layerwise_hit_check_keys(self, group_id: int, block_hash_hex: str) -> list[str]:
        """All-rank keys for scheduler-side hit check, built by the
        backend's protocol module.

        Single-group uses PR #11585 format; multi-group includes group_id.
        A block is a hit only when every PP stage has saved it, so the
        protocol helper enumerates all stages and head/TP ranks.
        """
        assert self.layerwise_keys is not None
        draft_keys = self.dspark_prefix_keys
        if draft_keys is not None and group_id in self.dspark_draft_only_group_ids:
            # Draft-only groups have no ordinary layerwise target object.
            return draft_keys.make_hit_check_keys(block_hash_hex)
        head_or_tp_ranks = (
            self.tp_size if group_id in self.num_speculative_blocks_by_group else self.tp_size // self.put_step
        )
        keys = self.layerwise_keys.make_hit_check_keys(group_id, block_hash_hex, head_or_tp_ranks)
        if draft_keys is not None:
            keys.extend(draft_keys.make_hit_check_keys(block_hash_hex))
        return keys

    def _query_layerwise_block_hits(self, keys_by_block: list[list[str]]) -> list[bool]:
        """Probe backend readability and fold physical keys into block hits."""
        all_keys = [key for block_keys in keys_by_block for key in block_keys]
        if not all_keys:
            return []

        keys_per_block = max(len(block_keys) for block_keys in keys_by_block)
        batch_size = (
            self.layerwise_max_transfer_blocks * keys_per_block
            if self.layerwise_max_transfer_blocks > 0
            else len(all_keys)
        )
        readable: list[bool] = []
        for start in range(0, len(all_keys), batch_size):
            batch = all_keys[start : start + batch_size]
            batch_readable = self.store_scheduler.batch_is_readable(batch)
            if len(batch_readable) != len(batch) or any(type(state) is not bool for state in batch_readable):
                raise RuntimeError("Layerwise readability probe returned invalid results")
            readable.extend(batch_readable)

        block_hits = []
        offset = 0
        for block_keys in keys_by_block:
            block_hits.append(all(readable[offset : offset + len(block_keys)]))
            offset += len(block_keys)
        return block_hits
