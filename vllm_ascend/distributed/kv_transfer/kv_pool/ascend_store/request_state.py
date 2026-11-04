# SPDX-License-Identifier: Apache-2.0
"""Request tracking and side-effect-free save decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, cast

from vllm.utils.math_utils import cdiv


class SaveDecision(NamedTuple):
    target_token_len: int
    previous_saved_tokens: int
    save_end_token: int
    partial_block_index: int | None
    should_save_partial_block: bool
    skip_save: bool
    update_snapshot: bool
    snapshot_start_block: int | None
    snapshot_block_index: int | None


def normalize_block_ids_by_group(block_ids: tuple[list[int], ...] | list[int] | list[list[int]]) -> list[list[int]]:
    if isinstance(block_ids, tuple):
        return [group.copy() for group in block_ids]
    if isinstance(block_ids, list):
        if not block_ids:
            return [[]]
        if isinstance(block_ids[0], list):
            grouped_block_ids = cast(list[list[int]], block_ids)
            return [group.copy() for group in grouped_block_ids]
        flat_block_ids = cast(list[int], block_ids)
        return [flat_block_ids.copy()]
    raise ValueError(f"Unsupported block_ids type {type(block_ids)}")


@dataclass(init=False)
class RequestTracker:
    # Request id
    req_id: str

    token_len: int

    # The block ids that has been allocated so far, grouped by KV cache group.
    # NOTE: allocated blocks could be more than the number of tokens.
    allocated_block_ids_by_group: list[list[int]]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    # The token ids that has been scheduled so far
    # NOTE: This field will only be used when you enable kv-event
    token_ids: list[int] | None = None

    # Full prompt length before chunk truncation, used by sparse retention masks.
    num_prompt_tokens: int | None = None

    # End of the range being (re)prefilled. For a resumed request this also
    # includes previously generated tokens, so decode-only offload does not
    # publish intermediate recomputation state.
    prefill_end_tokens: int | None = None
    block_gvas: list[int] = field(default_factory=list)
    block_gvas_by_group: list[list[int]] = field(default_factory=list)
    gva_block_offset: int = 0
    last_block_gva: int | None = None

    # Identity of the latest request-scoped layerwise partial snapshot.
    last_layerwise_snapshot_end_token: int | None = None
    last_layerwise_snapshot_block_index: int | None = None
    last_layerwise_snapshot_start_block: int | None = None

    # Number of speculative scratch blocks for each Mamba cache group.
    num_speculative_blocks_by_group: dict[int, int] | None = None

    block_sizes: list[int] | None = None

    def __init__(
        self,
        req_id: str,
        token_len: int,
        allocated_block_ids_by_group: list[list[int]] | None = None,
        allocated_block_ids: list[int] | list[list[int]] | None = None,
        num_saved_tokens: int = 0,
        token_ids: list[int] | None = None,
        num_prompt_tokens: int | None = None,
        prefill_end_tokens: int | None = None,
        block_gvas: list[int] | None = None,
        block_gvas_by_group: list[list[int]] | None = None,
        gva_block_offset: int = 0,
        last_block_gva: int | None = None,
        num_speculative_blocks_by_group: dict[int, int] | None = None,
        block_sizes: list[int] | None = None,
        last_layerwise_snapshot_end_token: int | None = None,
        last_layerwise_snapshot_block_index: int | None = None,
        last_layerwise_snapshot_start_block: int | None = None,
    ) -> None:
        self.req_id = req_id
        self.token_len = token_len
        self.num_speculative_blocks_by_group = num_speculative_blocks_by_group
        block_ids = allocated_block_ids_by_group
        if block_ids is None:
            block_ids = normalize_block_ids_by_group(allocated_block_ids or [])
        self.allocated_block_ids_by_group = block_ids
        self.num_saved_tokens = num_saved_tokens
        self.token_ids = token_ids
        self.num_prompt_tokens = num_prompt_tokens
        self.prefill_end_tokens = prefill_end_tokens
        self.block_gvas = [] if block_gvas is None else block_gvas
        self.block_gvas_by_group = block_gvas_by_group if block_gvas_by_group is not None else []
        self.gva_block_offset = gva_block_offset
        self.last_block_gva = last_block_gva
        self.block_sizes = block_sizes
        self.last_layerwise_snapshot_end_token = last_layerwise_snapshot_end_token
        self.last_layerwise_snapshot_block_index = last_layerwise_snapshot_block_index
        self.last_layerwise_snapshot_start_block = last_layerwise_snapshot_start_block

    @property
    def allocated_block_ids(self) -> list[int]:
        return self.allocated_block_ids_by_group[0] if self.allocated_block_ids_by_group else []

    @allocated_block_ids.setter
    def allocated_block_ids(self, block_ids: list[int] | list[list[int]]) -> None:
        self.allocated_block_ids_by_group = normalize_block_ids_by_group(block_ids)

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[int],
        num_computed_tokens: int = 0,
    ) -> None:
        """Update the request tracker when a running request is scheduled again."""
        normalized = normalize_block_ids_by_group(new_block_ids)
        if len(normalized) > len(self.allocated_block_ids_by_group):
            self.allocated_block_ids_by_group.extend(
                [[] for _ in range(len(normalized) - len(self.allocated_block_ids_by_group))]
            )
        for group_id, ids in enumerate(normalized):
            self.update_mamba_spec_blocks(ids, group_id, num_computed_tokens)
            self.allocated_block_ids_by_group[group_id].extend(ids)

    def update_mamba_spec_blocks(self, block_ids: list[int], kv_cache_group_id: int, num_computed_tokens: int):
        """
        for mamba align groups, each step will:
            - Firstly, remove some previous blocks and append some necessary null blocks
            - Secondly, move the speculative blocks(maybe all or partially) to the last position for reuse
            - Finally, allocate a new block
        so, if a speculative block is moved to last position and replaced with null block,
        we also need to update the previous allocated_block_ids to 0.
        """
        if (
            self.num_speculative_blocks_by_group is not None
            and (num_speculative_blocks := self.num_speculative_blocks_by_group.get(kv_cache_group_id)) is not None
        ):
            assert self.block_sizes is not None and len(self.block_sizes) > kv_cache_group_id
            num_skipped_blocks = (
                max(num_computed_tokens - num_speculative_blocks - 1, 0) // self.block_sizes[kv_cache_group_id]
            )
            num_skipped_blocks = min(len(self.allocated_block_ids_by_group[kv_cache_group_id]), num_skipped_blocks)
            if num_skipped_blocks > 0:
                self.allocated_block_ids_by_group[kv_cache_group_id][:num_skipped_blocks] = [0] * num_skipped_blocks
            if not block_ids or num_speculative_blocks <= 0:
                return
            mask_spec_count = min(len(block_ids) - 1, num_speculative_blocks)
            group_block_ids = self.allocated_block_ids_by_group[kv_cache_group_id]
            if mask_spec_count >= num_speculative_blocks:
                group_block_ids[-num_speculative_blocks:] = [0] * num_speculative_blocks
            else:
                group_block_ids[-num_speculative_blocks : mask_spec_count - num_speculative_blocks] = [0] * (
                    mask_spec_count
                )

    def plan_save(
        self,
        cache_transfer_granularity: int,
        block_hash_count: int,
        *,
        can_load: bool = False,
        skip_save: bool | None = False,
        discard_partial_chunks: bool = True,
        save_partial_block: bool = False,
        hash_block_size: int | None = None,
    ) -> SaveDecision:
        """Compute one step's save boundaries without advancing request state."""
        target_token_len = self.token_len
        chunk_boundary = (
            cdiv(self.num_saved_tokens + 1, cache_transfer_granularity) * cache_transfer_granularity
            if discard_partial_chunks
            else 0
        )
        num_tokens_to_save = (
            target_token_len // cache_transfer_granularity * cache_transfer_granularity
            if discard_partial_chunks
            else target_token_len
        )
        hash_block_size = hash_block_size or cache_transfer_granularity
        assert cache_transfer_granularity % hash_block_size == 0
        hashes_per_transfer_block = cache_transfer_granularity // hash_block_size
        full_block_count = target_token_len // cache_transfer_granularity
        available_full_block_count = block_hash_count // hashes_per_transfer_block
        boundary_without_hash = (
            target_token_len > 0
            and target_token_len % cache_transfer_granularity == 0
            and full_block_count > available_full_block_count
        )
        # Draft tokens may cross a page before its request hash is committed.
        if full_block_count > available_full_block_count:
            num_tokens_to_save = available_full_block_count * cache_transfer_granularity
        snapshot_block_index = (
            full_block_count if target_token_len % cache_transfer_granularity != 0 else full_block_count - 1
        )
        partial_block_index = (
            snapshot_block_index
            if self.last_block_gva is not None
            and (target_token_len % cache_transfer_granularity != 0 or boundary_without_hash)
            else None
        )
        should_save_partial_block = save_partial_block and (
            target_token_len % cache_transfer_granularity != 0 or full_block_count > available_full_block_count
        )
        skip_save = bool(skip_save) or (
            num_tokens_to_save < chunk_boundary and partial_block_index is None and not should_save_partial_block
        )
        # Reuse snapshots can require a load and a save in the same step.
        if can_load and not save_partial_block:
            skip_save = True
        return SaveDecision(
            target_token_len=target_token_len,
            previous_saved_tokens=self.num_saved_tokens,
            save_end_token=num_tokens_to_save,
            partial_block_index=partial_block_index,
            should_save_partial_block=should_save_partial_block,
            skip_save=skip_save,
            update_snapshot=save_partial_block,
            snapshot_start_block=min(full_block_count, available_full_block_count)
            if should_save_partial_block
            else None,
            snapshot_block_index=snapshot_block_index if should_save_partial_block else None,
        )

    def apply_save_decision(self, decision: SaveDecision) -> None:
        if decision.skip_save:
            return
        self.num_saved_tokens = max(self.num_saved_tokens, decision.save_end_token)
        if decision.update_snapshot:
            self.last_layerwise_snapshot_end_token = (
                decision.target_token_len if decision.should_save_partial_block else None
            )
            self.last_layerwise_snapshot_start_block = decision.snapshot_start_block
            self.last_layerwise_snapshot_block_index = decision.snapshot_block_index
