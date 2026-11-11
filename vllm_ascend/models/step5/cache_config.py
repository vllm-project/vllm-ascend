# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream-style KV cache planning for Step5 on the Ascend LBNHC layout.

Upstream vLLM runs Step5 with a block-outermost layout where
``_get_packed_kv_cache_groups`` packs the per-request CSA tail ring
(``CircularBufferSpec``, one state bucket per request) and the 8-token
sparse summary (4 KiB pages) next to the main KV pages. The Ascend layout
is layer-outer (LBNHC), so that packed path never runs
(``not layout.is_block_outermost``), and the layer-outer fallback relies on
``unify_kv_cache_spec_page_size``: every group must share one page size so
block ids cannot overlap inside the single aliased backing. Unify therefore
grows the summary block size 128 -> 3072 unless the summary page is
pre-padded, and both auxiliary caches end up costing a full 96 KiB main page
per block per layer -- which triples the KV footprint of the 23 CSA layers.

This module keeps the auxiliary caches at their real upstream sizes and
splits the pool by page size, which is what upstream achieves with state
buckets:

* main KV groups (96 KiB pages) keep upstream grouping and still alias one
  backing exactly as before -- equal pages keep the shared block-id space
  collision-free;
* every other page size gets its own backing, sized to the widest group of
  that class, so the tail ring's 32 KiB pages and the summary's 4 KiB pages
  no longer pad to the main page.

The planner-side cost model matches the physical allocation through
``get_step5_pool_bytes_per_block``, the same hook the GLM/DSV4 multi-tensor
layouts use.
"""

from __future__ import annotations

from collections.abc import Callable

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_planning import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

_STEP5_AUX_SUFFIXES = (".csa_tail_ring", ".sparse_summary_cache")


def _layer_specs(group: KVCacheGroupSpec) -> list[tuple[str, KVCacheSpec]]:
    group_spec = group.kv_cache_spec
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        return list(group_spec.kv_cache_specs.items())
    return [(name, group_spec) for name in group.layer_names]


def _page_alignment(spec: KVCacheSpec) -> int:
    alignment = getattr(spec, "get_block_stride_alignment", None)
    if alignment is None:
        return 1
    return alignment()


def _padded_page(spec: KVCacheSpec) -> int:
    page = spec.page_size_bytes
    alignment = _page_alignment(spec)
    return page + (-page % alignment)


def is_step5_kv_cache_spec(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """Whether the model carries the Step5 CSA auxiliary cache layers."""
    return any(name.endswith(_STEP5_AUX_SUFFIXES) for name in kv_cache_spec)


def is_step5_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    return any(name.endswith(_STEP5_AUX_SUFFIXES) for group in kv_cache_groups for name in group.layer_names)


def get_step5_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    orig_get_kv_cache_groups: Callable[[VllmConfig, dict[str, KVCacheSpec]], list[KVCacheGroupSpec]],
) -> list[KVCacheGroupSpec] | None:
    """Group the main layers upstream, then isolate the auxiliary layers.

    Removing the tail-ring and summary specs before upstream grouping keeps
    ``unify_kv_cache_spec_page_size`` from ever seeing the small pages (it
    would grow their block sizes to keep one aliased backing). The remaining
    main layers all share the 96 KiB page, so upstream groups them exactly
    as a model without CSA layers would.
    """
    if not is_step5_kv_cache_spec(kv_cache_spec):
        return None

    main_specs = {name: spec for name, spec in kv_cache_spec.items() if not name.endswith(_STEP5_AUX_SUFFIXES)}
    groups = orig_get_kv_cache_groups(vllm_config, main_specs)

    for suffix in _STEP5_AUX_SUFFIXES:
        aux_specs = {name: spec for name, spec in kv_cache_spec.items() if name.endswith(suffix)}
        if not aux_specs:
            continue
        uniform = UniformTypeKVCacheSpecs.from_specs(aux_specs)
        if uniform is None:
            # Mixed draft/target auxiliary geometry: fall back to one group
            # per layer so each keeps its own page.
            groups.extend(KVCacheGroupSpec([name], spec) for name, spec in aux_specs.items())
        else:
            groups.append(KVCacheGroupSpec(list(aux_specs.keys()), uniform))
    return groups


def _step5_backing_classes(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> dict[int, int]:
    """Map page size -> backing width (widest group with that page size).

    Groups whose layers share one page size can alias one backing: equal
    pages mean block id ``b`` always covers the same byte span pattern, and
    globally unique block ids keep those spans disjoint. Different page
    sizes must not share a backing because their block spans would overlap.
    """
    classes: dict[int, int] = {}
    for group in kv_cache_groups:
        layer_specs = _layer_specs(group)
        pages = {_padded_page(spec) for _, spec in layer_specs}
        if len(pages) != 1:
            # A group mixing page sizes cannot alias with a same-page class;
            # give it a synthetic class keyed by its own width (negated so it
            # can never collide with a real page key) so it is allocated alone.
            width = sum(_padded_page(spec) for _, spec in layer_specs)
            classes[-width] = max(classes.get(-width, 0), width)
            continue
        page = pages.pop()
        width = sum(_padded_page(spec) for _, spec in layer_specs)
        classes[page] = max(classes.get(page, 0), width)
    return classes


def get_step5_pool_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Physical bytes one scheduler block occupies across all backings."""
    classes = _step5_backing_classes(kv_cache_groups)
    logger.info(
        "Step5 KV pool bytes/block: %d (classes %s)",
        sum(classes.values()),
        dict(sorted(classes.items())),
    )
    return sum(classes.values())


def get_step5_max_memory_usage(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Per-request worst-case KV memory with per-page-size backings.

    Groups that share a page size alias one backing and one block-id span,
    so a request's cost in each backing class is the widest claim among that
    class's groups (e.g. full attention claims one block per 128 tokens
    while the sliding-window groups share those ids), not the sum. Classes
    with different page sizes hold independent block ids and add up.
    """
    from vllm.utils.math_utils import cdiv

    classes = _step5_backing_classes(kv_cache_groups)
    claim_per_class: dict[int, int] = {}
    for group in kv_cache_groups:
        if getattr(group, "host_resident", False):
            continue
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            pages = spec.max_memory_usage_pages(vllm_config)
        else:
            pages = cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )
        layer_specs = _layer_specs(group)
        pages_set = {_padded_page(s) for _, s in layer_specs}
        key = pages_set.pop() if len(pages_set) == 1 else -sum(_padded_page(s) for _, s in layer_specs)
        claim_per_class[key] = max(claim_per_class.get(key, 0), pages)
    return sum(pages * classes[key] for key, pages in claim_per_class.items())


def get_step5_max_concurrency(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> float:
    """Honest Step5 capacity for the "NPU KV cache size" line.

    Every group draws block ids from one shared pool (``num_blocks`` ids
    with exclusive ownership), so a request's admission demand is the sum
    over all groups -- the same formula the scheduler's full-sequence
    admission gate uses. Sliding-window demand only stops growing once the
    window saturates (``max_admission_blocks_per_request``, i.e. prompts
    longer than the in-flight budget), so linearly scaling
    ``num_blocks / demand(max_model_len)`` overstates how long a single
    prompt can be. Return the smaller of the strict concurrency and
    ``longest_fitting_request / max_model_len`` so both printed numbers
    stay true.
    """
    from vllm.utils.math_utils import cdiv

    max_model_len = vllm_config.model_config.max_model_len
    num_blocks = kv_cache_config.num_blocks
    # (block size, ids needed at max_model_len, spec type name)
    entries: list[tuple[int, int, str]] = []
    for group in kv_cache_config.kv_cache_groups:
        if getattr(group, "host_resident", False):
            continue
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            ids_max = spec.max_memory_usage_pages(vllm_config)
            block_size = next(iter(spec.kv_cache_specs.values())).block_size
        else:
            ids_max = cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            block_size = spec.block_size
        entries.append((block_size, ids_max, type(spec).__name__))

    def ids_total(length: int) -> int:
        # Spec kinds interpolate differently below max_model_len: the ring
        # holds one bucket per request regardless of length, sliding-window
        # demand saturates at its admission cap, the rest grow linearly.
        total = 0
        for block_size, ids_max, spec_name in entries:
            if spec_name == "CircularBufferSpec":
                ids = ids_max
            elif spec_name == "SlidingWindowSpec":
                ids = min(cdiv(length, block_size) + 1, ids_max)
            else:
                ids = min(cdiv(length, block_size), ids_max)
            total += ids
        return total

    at_max = ids_total(max_model_len)
    strict = num_blocks / max(at_max, 1)

    # Longest single request whose full-sequence demand fits the pool. The
    # scheduler adds its watermark on top, so shave one block of slack.
    limit = num_blocks - 1
    lo, hi = 0, max_model_len
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ids_total(mid) <= limit:
            lo = mid
        else:
            hi = mid - 1
    logger.info(
        "Step5 KV capacity: num_blocks %d, ids/request at %d tokens %d; longest request that fits: %d tokens",
        num_blocks,
        max_model_len,
        at_max,
        lo,
    )
    return min(strict, lo / max_model_len)


def get_step5_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Like upstream ``get_kv_cache_config_from_groups`` for layer-outer
    layouts, but sizes each backing class by its own width instead of the
    global ``bytes_per_block``, and sizes ``num_blocks`` by the sum of the
    backing classes (the real per-block pool cost).
    """
    classes = _step5_backing_classes(kv_cache_groups)
    pool_bytes_per_block = sum(classes.values())
    num_blocks = available_memory // pool_bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    page_sizes = {spec.page_size_bytes for group in kv_cache_groups for _, spec in _layer_specs(group)}
    logger.info(
        "Step5 KV cache plan: %d groups, page sizes %s, backing widths %s (pool bytes/block %d), num_blocks %d",
        len(kv_cache_groups),
        sorted(page_sizes),
        {page: width for page, width in sorted(classes.items())},
        pool_bytes_per_block,
        num_blocks,
    )

    kv_cache_tensors: list[KVCacheTensor] = []
    for group in kv_cache_groups:
        layer_specs = _layer_specs(group)
        pages = {_padded_page(spec) for _, spec in layer_specs}
        backing_size = (
            classes[pages.pop()] * num_blocks
            if len(pages) == 1
            else sum(_padded_page(spec) for _, spec in layer_specs) * num_blocks
        )
        byte_offset = 0
        for layer_name, spec in layer_specs:
            layer_stride = spec.page_size_bytes * num_blocks
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[layer_name],
                    layer_stride=layer_stride,
                    block_stride=spec.page_size_bytes,
                    offset=byte_offset,
                )
            )
            byte_offset += layer_stride

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(vllm_config.cache_config.prefix_cache_retention_interval),
    )
