# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream-style KV cache planning for Step5 on the Ascend LBNHC layout.

Upstream vLLM runs Step5 with a block-outermost layout where
``_get_packed_kv_cache_groups`` packs the per-request CSA tail ring
(``CircularBufferSpec``, one state bucket per request) and the 8-token
sparse summary (4 KiB pages) next to the main KV pages, all as views into
one aliased backing whose per-id cost is the widest group. The Ascend
layout is layer-outer (LBNHC), so that packed path never runs, and the
layer-outer fallback relies on ``unify_kv_cache_spec_page_size``: every
group must share one page size, which would grow the summary block to
3,072 tokens and charge every auxiliary cache a full 96 KiB main page per
layer.

This module mirrors the upstream layout on Ascend instead:

* one envelope-sized backing: every group's layers start at byte 0 (the
  CSA ring stack one 4 KiB page in, keeping per-class start addresses
  disjoint for the recycled-block zeroer), and since all groups draw
  block ids from one exclusive pool the aliased per-id spans never
  collide -- the pool cost per block id is the widest group (the
  envelope), not the sum of per-page-size backings;
* the sparse summary keeps its real 4 KiB page but shares the DSA
  layer's group and block ids, like upstream ``Step5SparseSummaryCache``
  ("shares the DSA layer's KV cache group and block IDs, and is prefix
  cached and transferred like any other page"): one id per 128 tokens
  carries both the main KV page and the summary page, so a request costs
  one id set instead of two and the two hash chains cannot diverge;
* the CSA tail ring keeps its own ``CircularBufferSpec`` group (one
  scratch bucket per request, not prefix cacheable) and free-rides
  inside the same envelope because its ids are exclusive with the rest.

The planner-side cost model matches the physical allocation through
``get_step5_pool_bytes_per_block``, the same hook the GLM/DSV4 multi-tensor
layouts use.
"""

from __future__ import annotations

from typing import Callable

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_planning import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

_STEP5_RING_SUFFIX = ".csa_tail_ring"
_STEP5_SUMMARY_SUFFIX = ".sparse_summary_cache"
_STEP5_AUX_SUFFIXES = (_STEP5_RING_SUFFIX, _STEP5_SUMMARY_SUFFIX)

# The recycled-block zeroer keeps one stride per start address, so page
# classes must never share a start address inside the aliased backing.
# Shifting the ring stack by one 4 KiB page lands its layer starts on odd
# multiples of 4 KiB while the 96 KiB / 4 KiB summary classes start on even
# multiples (96 KiB = 24 x 4 KiB), keeping the starts disjoint.
_STEP5_RING_STACK_SHIFT_BYTES = 4096

logger = init_logger(__name__)


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
    return any(
        name.endswith(_STEP5_AUX_SUFFIXES)
        for group in kv_cache_groups
        for name in group.layer_names
    )


def get_step5_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    orig_get_kv_cache_groups: Callable[
        [VllmConfig, dict[str, KVCacheSpec]], list[KVCacheGroupSpec]
    ],
) -> list[KVCacheGroupSpec] | None:
    """Group the main layers upstream, then fold in the auxiliary layers.

    Removing the tail-ring and summary specs before upstream grouping keeps
    ``unify_kv_cache_spec_page_size`` from ever seeing the small pages (it
    would grow their block sizes to keep one aliased backing). The
    remaining main layers all share the 96 KiB page, so upstream groups
    them exactly as a model without CSA layers would.

    Each summary layer then rejoins the group holding its DSA parent layer
    (``<parent>.sparse_summary_cache`` -> ``<parent>``): the merged group
    keeps one spec per layer -- the DSA layers keep their 96 KiB page and
    the summary layers keep their 4 KiB page -- while sharing one group id
    and therefore one block-id chain, which is the upstream
    ``Step5SparseSummaryCache`` contract. The tail ring stays a separate
    ``CircularBufferSpec`` group: its one scratch bucket per request must
    not share ids with the full-sequence caches. Finally the sliding-window
    groups are re-balanced toward the widest other group (upstream's
    block-outermost packer lands on 3 x 24 layers via ``_approximate_gcd``;
    the layer-outer fallback splits 4 x 18, one window admission too many).
    """
    if not is_step5_kv_cache_spec(kv_cache_spec):
        return None

    main_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if not name.endswith(_STEP5_AUX_SUFFIXES)
    }
    groups = orig_get_kv_cache_groups(vllm_config, main_specs)

    # Summary layers -> the group owning their DSA parent layer. Keep the
    # parent layers first in the merged group so the scheduler-side
    # flattening (generate_scheduler_kv_cache_config keeps only the first
    # spec of a UniformTypeKVCacheSpecs) lands on the DSA FullAttentionSpec.
    summary_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.endswith(_STEP5_SUMMARY_SUFFIX)
    }
    unmerged: dict[str, KVCacheSpec] = {}
    if summary_specs:
        per_group_extras: dict[int, dict[str, KVCacheSpec]] = {}
        for name, spec in summary_specs.items():
            parent = name[: -len(_STEP5_SUMMARY_SUFFIX)]
            for idx, group in enumerate(groups):
                if parent in group.layer_names:
                    per_group_extras.setdefault(idx, {})[name] = spec
                    break
            else:
                unmerged[name] = spec
        for idx, extras in per_group_extras.items():
            member = dict(_layer_specs(groups[idx]))
            member.update(extras)
            uniform = UniformTypeKVCacheSpecs.from_specs(member)
            if uniform is None:
                # Mixed draft/target geometry upstream refused to merge:
                # keep the old isolated-group behaviour for these layers.
                unmerged.update(extras)
            else:
                groups[idx] = KVCacheGroupSpec(list(member.keys()), uniform)


    ring_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.endswith(_STEP5_RING_SUFFIX)
    }
    for aux_specs in (unmerged, ring_specs):
        if not aux_specs:
            continue
        uniform = UniformTypeKVCacheSpecs.from_specs(aux_specs)
        if uniform is None:
            # Mixed draft/target auxiliary geometry: fall back to one group
            # per layer so each keeps its own page.
            groups.extend(
                KVCacheGroupSpec([name], spec) for name, spec in aux_specs.items()
            )
        else:
            groups.append(
                KVCacheGroupSpec(list(aux_specs.keys()), uniform)
            )
    return groups


def _step5_group_width(group: KVCacheGroupSpec) -> int:
    """Bytes one block id spans across this group's layers.

    Layers inside one group share the block id simultaneously, so their
    (padded) pages stack: this is the group's per-id width.
    """
    return sum(_padded_page(spec) for _, spec in _layer_specs(group))


def _step5_envelope(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Per-id width of the single aliased backing: the widest group.

    Every group draws ids from the one exclusive pool, so each group's
    per-id span may start at byte 0 of the same backing; an id is owned by
    at most one group at a time, so the aliased spans never collide. The
    backing therefore only has to be as wide as the widest group.
    """
    return max(_step5_group_width(group) for group in kv_cache_groups)


def get_step5_pool_bytes_per_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Physical bytes one scheduler block occupies in the aliased backing."""
    envelope = _step5_envelope(kv_cache_groups)
    widths = {_step5_group_width(group) for group in kv_cache_groups}
    logger.info(
        "Step5 KV pool bytes/block (envelope): %d (group widths %s)",
        envelope,
        sorted(widths, reverse=True),
    )
    return envelope


def get_step5_max_memory_usage(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Per-request worst-case KV memory under the single aliased backing.

    Every id a request holds -- in any group -- pins one envelope slot in
    the backing, so the worst case is the sum of the per-group id claims
    (the sliding-window groups saturate at their admission cap, the ring
    holds one bucket, full attention grows to max_model_len) times the
    envelope width.
    """
    from vllm.utils.math_utils import cdiv

    envelope = _step5_envelope(kv_cache_groups)
    total_pages = 0
    for group in kv_cache_groups:
        if getattr(group, "host_resident", False):
            continue
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            total_pages += spec.max_memory_usage_pages(vllm_config)
        else:
            total_pages += cdiv(
                spec.max_memory_usage_bytes(vllm_config),
                spec.page_size_bytes,
            )
    return total_pages * envelope


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
            ids_max = cdiv(
                spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes
            )
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
        "Step5 KV capacity: num_blocks %d, ids/request at %d tokens %d; "
        "longest request that fits: %d tokens",
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
    layouts, but every tensor aliases one envelope-sized backing (the
    widest group) and ``num_blocks`` is sized by that envelope.
    """
    envelope = _step5_envelope(kv_cache_groups)
    pool_bytes_per_block = envelope
    num_blocks = available_memory // pool_bytes_per_block
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    page_sizes = {
        spec.page_size_bytes
        for group in kv_cache_groups
        for _, spec in _layer_specs(group)
    }
    group_widths = [_step5_group_width(group) for group in kv_cache_groups]
    logger.info(
        "Step5 KV cache plan: %d groups, page sizes %s, group widths %s "
        "(envelope bytes/block %d), num_blocks %d",
        len(kv_cache_groups),
        sorted(page_sizes),
        sorted(group_widths, reverse=True),
        envelope,
        num_blocks,
    )

    kv_cache_tensors: list[KVCacheTensor] = []
    backing_size = envelope * num_blocks
    for group in kv_cache_groups:
        byte_offset = 0
        if all(name.endswith(_STEP5_RING_SUFFIX) for name in group.layer_names):
            byte_offset = _STEP5_RING_STACK_SHIFT_BYTES * num_blocks
        for layer_name, spec in _layer_specs(group):
            layer_stride = spec.page_size_bytes * num_blocks
            if byte_offset + layer_stride > backing_size:
                raise ValueError(
                    f"Step5 group ({group.layer_names[0]} ..) does not fit "
                    f"the {envelope}-byte envelope backing."
                )
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

    # Upstream's recycled-block zeroer maps one stride per start address;
    # same-page views may share a start (equal strides dedup), but two page
    # classes sharing one start would be inexpressible.
    start_pages: dict[int, int] = {}
    for tensor in kv_cache_tensors:
        page = tensor.block_stride
        if start_pages.setdefault(tensor.offset, page) != page:
            raise ValueError(
                f"Step5 envelope aliasing puts a {page}-byte page at a "
                f"start address already held by a "
                f"{start_pages[tensor.offset]}-byte page; per-class stack "
                f"bases must keep start addresses disjoint."
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )
