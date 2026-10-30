# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep sparse selection and attention in one Draft read coordinate system."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any


def validate_sfa_read_consumers(
    layers: Mapping[str, Any],
    routes: Sequence[Any],
    *,
    sfa_impl_type: type[Any] | None = None,
    indexer_impl_type: type[Any] | None = None,
) -> None:
    """Require an independently selecting indexer in the Draft metadata set.

    SFA interprets sparse indices relative to its input block table. Its
    indexer must select over the same logical Window, even when the physical
    cache is shared with a target layer. Target-produced IndexCache entries
    cannot be fixed by subtracting the Window start: that would discard
    selected keys without replacing them with the window's next-best keys.
    """
    from vllm_ascend.worker.v2.spec_decode.draft_window_backends import DraftWindowProviderStatus

    if sfa_impl_type is None or indexer_impl_type is None:
        from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

        sfa_impl_type = AscendSFAImpl
        indexer_impl_type = AscendSFAIndexerBackend
    by_layer = {name: route for route in routes for name in route.layers}
    for route in routes:
        if route.status is not DraftWindowProviderStatus.DIRECT_SFA:
            continue
        for name in route.layers:
            impl = layers[name].impl
            if type(impl) is not sfa_impl_type:
                raise NotImplementedError(f"Draft Window SFA implementation provider missing for {name}")
            if not impl.runtime_has_indexer or impl.skip_topk or impl.use_index_cache:
                raise NotImplementedError("Draft Window SFA shared IndexCache needs a window-local selection producer")
            if type(impl.indexer.impl) is not indexer_impl_type:
                raise NotImplementedError("Draft Window SFA indexer implementation provider missing")
            prefixes = [impl.indexer.k_cache.prefix]
            target = getattr(impl, "kv_sharing_target_layer_name", None)
            if target is not None:
                prefixes.insert(0, f"{target.removesuffix('.attn')}.indexer.k_cache")
            # Match the actual forward resolver, using the metadata names that
            # this Draft build will contain. A target-only prefix is absent.
            selected = next((prefix for prefix in prefixes if prefix in by_layer), None)
            if selected is None or by_layer[selected].status is not DraftWindowProviderStatus.DIRECT_SFA_INDEXER:
                raise NotImplementedError(
                    f"Draft Window SFA indexer read metadata provider missing for {name}: {prefixes}"
                )


def freeze_sfa_read_metadata(metadata: Mapping[str, Any], layer_names: Collection[str]) -> tuple[Any, ...]:
    """Detach RoPE scratch while preserving table, Full slot and query identity.

    Main SFA first-pass builders use module-global RoPE scratch. The indexer
    builder has per-slot-address scratch, which can be reused by another
    proposal with the same query layout. Both are live read operands, so own
    their logical extents until the proposal's final reader completes.
    """
    retained = []
    seen = set()
    for name in layer_names:
        item = metadata[name]
        if id(item) in seen:
            continue
        seen.add(id(item))
        for field in ("cos", "sin"):
            value = getattr(item, field)
            if value is not None:
                owned = value.clone()
                setattr(item, field, owned)
                retained.append(owned)
    return tuple(retained)
