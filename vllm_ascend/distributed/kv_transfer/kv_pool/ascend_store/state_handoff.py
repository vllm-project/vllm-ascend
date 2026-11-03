# SPDX-License-Identifier: Apache-2.0
"""Select core-offered state boundaries for native AscendStore save jobs.

This is the adapter for SchedulerOutput.kv_connector_block_state, following
MooncakeStoreScheduler._handle_boundary_state_offloads. No state is inferred
from scheduled-token counts or reconstructed from append-only block tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .metadata import BoundaryPut, block_hash_to_bytes
from .state_store.state_keys import StateKeyPolicy


def select_boundary_puts(
    offers: Sequence[tuple[int, int, int]],
    *,
    store_group_ids: Mapping[int, int],
    state_group_ids: frozenset[int],
    hash_block_size: int,
    committed_hash_count: int,
    prefill_end_tokens: int,
    save_decode_cache: bool,
) -> tuple[BoundaryPut, ...]:
    """Filter native offers against committed hashes and the configured window.

    With decode saving disabled, preserve upstream's prefill boundary. With it
    enabled, admit native state offers beyond that boundary. The
    presence of a core offer is mandatory in both cases; enabling a flag does
    not synthesize checkpoints. Group IDs are mapped explicitly because Store
    can omit native request-private groups.

    These results are not yet permission to PUT: the scheduler must retain
    their blocks and the worker must wait for native copy completion. Model
    dependency-key binding (e.g. GLM MTP token[B]) is a separate adapter step.
    """
    if hash_block_size <= 0 or min(committed_hash_count, prefill_end_tokens) < 0:
        raise ValueError("invalid committed checkpoint window")
    # Core offers include states produced by this step. num_computed_tokens
    # still describes the previous step here; the worker event fences readiness.
    end = committed_hash_count * hash_block_size
    if not save_decode_cache:
        end = min(end, prefill_end_tokens)
    selected: dict[tuple[int, int], BoundaryPut] = {}
    for group_id, block_id, boundary in offers:
        mapped_group = store_group_ids.get(group_id)
        if mapped_group not in state_group_ids or block_id == 0:
            continue
        if block_id < 0 or boundary <= 0 or boundary % hash_block_size:
            raise ValueError("invalid native boundary-state offer")
        if boundary > end:
            continue
        put = BoundaryPut(mapped_group, block_id, boundary)
        key = mapped_group, boundary
        previous = selected.get(key)
        if previous is not None and previous.block_id != block_id:
            raise ValueError("native offers disagree on the source of one state boundary")
        selected[key] = put
    return tuple(selected.values())


def uses_boundary_state_store(transfer_config) -> bool:
    """Resolve the opt-in once at runner initialization, including Multi."""
    if transfer_config is None:
        return False

    def enabled(name, options):
        if options is None:
            options = {}
        if name == "MultiConnector":
            return any(
                enabled(child["kv_connector"], child.get("kv_connector_extra_config", {}))
                for child in options.get("connectors", ())
            )
        if name != "AscendStoreConnector":
            return False
        value = options.get("enable_state_store_reuse", False)
        if type(value) is not bool:
            raise ValueError("enable_state_store_reuse must be a boolean")
        return value

    return enabled(transfer_config.kv_connector, transfer_config.kv_connector_extra_config or {})


def has_state_transfers(metadata) -> bool:
    """Inspect native connector metadata without assuming child ordering."""
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorMetadata,
    )

    from .metadata import AscendConnectorMetadata

    if isinstance(metadata, MultiKVConnectorMetadata):
        return any(has_state_transfers(child) for child in metadata.metadata)
    if isinstance(metadata, AscendConnectorMetadata):
        return any(
            (request.can_save and request.boundary_puts)
            or (
                request.load_spec is not None
                and request.load_spec.can_load
                and request.load_spec.cache_key_hashes is not None
            )
            for request in metadata.requests
        )
    return False


def limit_save_prefixes(metadata, limits: Mapping[str, int]) -> None:
    """Exclude joint checkpoints containing an uncomputed draft prefix.

    Keep admitted jobs, including empty ones: native completion still owns
    their source references. Keys describe the whole target/draft checkpoint,
    so this also handles cache groups containing both kinds of layers.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorMetadata,
    )

    from .metadata import AscendConnectorMetadata

    if isinstance(metadata, MultiKVConnectorMetadata):
        for child in metadata.metadata:
            limit_save_prefixes(child, limits)
    elif isinstance(metadata, AscendConnectorMetadata):
        for request in metadata.requests:
            limit = limits.get(request.req_id)
            if limit is None or not request.can_save:
                continue
            # The keys and the ordinary-page interval describe one joint save.
            # Keep empty jobs so their pinned source references still retire.
            request.token_len_chunk = min(request.token_len_chunk, limit)
            request.save_end_token = min(request.save_end_token, request.token_len_chunk)
            if request.cache_key_hashes is not None:
                request.cache_key_hashes = {end: key for end, key in request.cache_key_hashes.items() if end <= limit}
            if request.boundary_puts:
                request.boundary_puts = [put for put in request.boundary_puts if put.num_tokens <= limit]


def bind_cache_key_hashes(
    puts: Sequence[BoundaryPut],
    block_hashes: Sequence[bytes],
    committed_token_ids: Sequence[int],
    *,
    hash_block_size: int,
    policy: StateKeyPolicy,
) -> tuple[tuple[BoundaryPut, ...], dict[int, bytes]]:
    """Bind each unique boundary once; never fall back to an unbound key.

    The caller uses the native hash unit, not a physical cache block size.
    Missing inputs defer that checkpoint; they are not replaced by recompute.
    """
    if hash_block_size <= 0:
        raise ValueError("hash block size must be positive")
    resolved: dict[int, bytes | None] = {}
    for put in puts:
        boundary = put.num_tokens
        if boundary in resolved:
            continue
        if boundary <= 0 or boundary % hash_block_size:
            raise ValueError("boundary does not align with native prefix hashes")
        index = boundary // hash_block_size - 1
        resolved[boundary] = (
            policy.bind(block_hash_to_bytes(block_hashes[index]), boundary, committed_token_ids)
            if index < len(block_hashes)
            else None
        )
    keys = {boundary: value for boundary, value in resolved.items() if value is not None}
    return tuple(put for put in puts if put.num_tokens in keys), keys


def bind_prefix_key_hashes(
    block_hashes: Sequence[bytes],
    prompt_token_ids: Sequence[int],
    *,
    hash_block_size: int,
    token_limit: int,
    policy: StateKeyPolicy,
) -> dict[int, bytes]:
    """Bind native prefix hashes to committed source or reader input tokens.

    Missing extra inputs omit a candidate. Never substitute the native prefix
    hash for a state whose shifted-input identity cannot be established.
    """
    if hash_block_size <= 0 or token_limit < 0:
        raise ValueError("invalid lookup hash window")
    count = min(len(block_hashes), token_limit // hash_block_size)
    bound = {}
    for index in range(count):
        boundary = (index + 1) * hash_block_size
        value = policy.bind(block_hash_to_bytes(block_hashes[index]), boundary, prompt_token_ids)
        if value is not None:
            bound[boundary] = value
    return bound
