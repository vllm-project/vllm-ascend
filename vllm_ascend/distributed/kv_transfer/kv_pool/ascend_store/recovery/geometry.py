# SPDX-License-Identifier: Apache-2.0
"""One immutable, Core-resolved cache plan for all three Store layers."""

from dataclasses import dataclass, replace

from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec, resolve_kv_cache_block_sizes


@dataclass(frozen=True)
class StoreGeometry:
    scheduler_block_size: int
    hash_block_size: int
    hit_alignment: int | None
    groups: tuple
    group_ids: tuple[int, ...]
    use_eagle: bool
    retention_interval: int | None

    @property
    def partial_hits(self) -> bool:
        return self.hit_alignment is not None and self.hit_alignment < self.scheduler_block_size

    @classmethod
    def resolve(cls, config, cache_config, transfer_config=None):
        scheduler_size, hash_size = resolve_kv_cache_block_sizes(cache_config, config)
        dcp = config.parallel_config.decode_context_parallel_size
        transfer = transfer_config if transfer_config is not None else config.kv_transfer_config
        producer_only = bool(transfer and transfer.is_kv_producer and not transfer.is_kv_consumer)
        speculative = config.speculative_config
        return cls(
            scheduler_size,
            hash_size,
            cache_config.cache_hit_alignment_tokens,
            tuple(
                replace(
                    group,
                    kv_cache_spec=resolve_dcp_kv_cache_spec(group.kv_cache_spec, dcp),
                    is_eagle_group=False if producer_only else group.is_eagle_group,
                )
                for group in cache_config.prefix_cacheable_groups
            ),
            tuple(cache_config.prefix_cacheable_group_ids),
            bool(speculative and speculative.use_eagle_block_drop() and not producer_only),
            cache_config.prefix_cache_retention_interval,
        )
