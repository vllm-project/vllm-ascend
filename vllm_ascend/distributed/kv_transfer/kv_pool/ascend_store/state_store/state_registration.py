# SPDX-License-Identifier: Apache-2.0
"""Compile model-described state groups into native Store layouts once.

The transport consumes RegisteredCache, not model classes. Native group IDs
remain local routing IDs; Store group IDs are assigned by semantic membership.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import gcd
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    KeyMetadata,
)

from .cache_schema import RegisteredCache, RegisteredPage, RegisteredState
from .state_keys import StateKeyPolicy
from .state_store_layout import AcceptedStateStoreLayout

if TYPE_CHECKING:
    from ..metadata import KeyMetadata as AscendKeyMetadata


def compile_state_store_layouts(
    registered: RegisteredCache,
    *,
    metadata: KeyMetadata,
    local_tp_size: int,
    store_tp_size: int,
    pp_size: int,
    num_blocks: int,
    hash_block_size: int,
    key_policy: StateKeyPolicy,
) -> dict[int, AcceptedStateStoreLayout]:
    """Return all layouts only after every group compiles successfully.

    Full cache membership and key dependencies qualify the namespace, including
    request-private components. Their presence here does not prove they can be
    restored; the mixed-cache load coordinator still owns that decision.
    """
    states = registered.states
    if pp_size <= 0 or not 0 <= metadata.pp_rank < pp_size:
        raise ValueError("invalid pipeline namespace")
    if metadata.pcp_rank or metadata.dcp_rank:
        raise NotImplementedError("state context-parallel ownership is not described")
    schema = dict(
        components=registered.semantic_schema(),
        key_policy=(key_policy.name, key_policy.extra_input_tokens),
    )
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    namespace = AcceptedStateStoreLayout.shared_namespace(store_tp_size, pp_size) + f"@cache_schema:{digest}"
    groups: dict[int, list[RegisteredState]] = {}
    for component in states:
        groups.setdefault(component.group_id, []).append(component)
    ordered_groups = sorted(
        groups.items(),
        key=lambda pair: sorted((c.layer_name, c.spec.name) for c in pair[1]),
    )
    layouts = {}
    for store_group_id, (native_group_id, components) in enumerate(ordered_groups):
        sizes = {c.block_tokens for c in components}
        if len(sizes) != 1:
            raise ValueError("one native state group must have one block token size")
        # The parent Store class names this count num_kv_heads. For the state
        # subclass it bounds divisible logical units across all projections;
        # physical addresses come exclusively from component byte templates.
        shard_units = gcd(*(n for c in components for n in c.spec.projection_sizes))
        layout = AcceptedStateStoreLayout(
            replace(metadata, group_id=store_group_id, store_namespace=namespace),
            sizes.pop(),
            hash_block_size,
            local_tp_size,
            store_tp_size,
            metadata.tp_rank,
            shard_units,
            components,
        )
        layout.register_kv_caches([c.tensor for c in components], num_blocks)
        layouts[native_group_id] = layout
    return layouts


def qualify_page_metadata(
    metadata: Sequence[AscendKeyMetadata],
    registered: RegisteredCache,
    layouts: Mapping[int, AcceptedStateStoreLayout],
) -> list[AscendKeyMetadata]:
    """Use semantic group IDs and explicit native page framing for Store keys.

    State bytes have their own canonical layout. Ordinary pages still use the
    native contiguous value path, so their object framing must match on peers.
    A different page framing misses rather than loading a short/aliased object.
    """
    if not registered.pages:
        return list(metadata)
    state_layout = next(iter(layouts.values()))
    namespace = state_layout.cache_namespace
    cache_prefix = state_layout.metadata.cache_prefix
    groups: dict[int, list[RegisteredPage]] = {}
    for page in registered.pages:
        groups.setdefault(page.group_id, []).append(page)
    ordered = sorted(groups.items(), key=lambda pair: sorted(p.layer_name for p in pair[1]))
    result = list(metadata)
    for store_group_id, (native_group_id, pages) in enumerate(ordered):
        schema = []
        # Native page descriptors follow registration order. Bind that order;
        # unlike state templates, this path does not repack page components.
        for page in pages:
            layout = page.layout
            rows = layout.kernel_tokens // layout.tokens_per_row
            if layout.row_stride_bytes != layout.row_bytes or layout.kernel_stride_bytes != rows * layout.row_bytes:
                raise NotImplementedError("native page transfer requires packed rows within each object")
            schema.append(
                (
                    page.layer_name,
                    page.spec.name,
                    layout.block_tokens,
                    layout.tokens_per_row,
                    layout.row_bytes,
                    str(page.tensor.dtype),
                )
            )
        digest = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
        result[native_group_id] = replace(
            result[native_group_id],
            head_or_tp_rank=0,
            kv_cache_group_id=store_group_id,
            store_namespace=f"{namespace}@page_schema:{digest}",
            cache_prefix=cache_prefix,
        )
    return result
