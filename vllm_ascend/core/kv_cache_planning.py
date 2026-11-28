# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Physical page alignment applied in Core planning before KV cache grouping."""

from dataclasses import replace

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec

_SPARSE_SUMMARY_SUFFIX = ".sparse_summary_cache"


def _align_hybrid_attention_page_sizes(
    kv_cache_specs: dict[str, KVCacheSpec],
    minimum_page_size: int = 0,
) -> None:
    """Align dense K/V segments before padding a shared Attention/Mamba pool."""
    del minimum_page_size
    if any(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in kv_cache_specs.values()):
        # Paired SFA has two physical page classes. Row byte ratios must never
        # change the logical token ownership or checkpoint granularity.
        return
    # MLA, compressed-cache subclasses, and Step5 auxiliary summaries manage
    # their own storage layouts.
    attention_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if type(spec) in (FullAttentionSpec, SlidingWindowSpec) and not name.endswith((_SPARSE_SUMMARY_SUFFIX,))
    }
    if not attention_specs:
        return
    reference = max(attention_specs.values(), key=lambda spec: spec.real_page_size_bytes)
    page_size = reference.real_page_size_bytes
    for layer_name, spec in attention_specs.items():
        # Ascend packs each layer as [all K blocks][all V blocks]. Equal
        # padded pages alone do not isolate block IDs: a smaller SWA K segment
        # can overlap another request's full-attention V block. Grow scheduler
        # blocks before padding; the backend still splits them into kernel blocks.
        if (
            page_size % spec.real_page_size_bytes
            or spec.head_size * reference.head_size_v != reference.head_size * spec.head_size_v
        ):
            raise ValueError(
                f"Cannot align hybrid attention K/V pages for {layer_name}: "
                "the shared contiguous cache requires matching K/V size ratios "
                "and divisible unpadded page sizes."
            )
        ratio = page_size // spec.real_page_size_bytes
        if ratio > 1:
            kv_cache_specs[layer_name] = replace(
                spec,
                block_size=spec.block_size * ratio,
                page_size_padded=max(spec.page_size_padded, page_size) if spec.page_size_padded is not None else None,
            )


def align_hybrid_kv_cache_specs(
    kv_cache_specs: dict[str, KVCacheSpec],
    *,
    attention_layer_names: set[str] | None = None,
    descriptor_native: bool = False,
) -> dict[str, KVCacheSpec]:
    """Plan shared pages across aggregated worker specs without mutating owner specs.

    Running hybrid page alignment in Core planning rather than inside per-worker
    ``get_kv_cache_spec`` keeps layer specs pure and ensures heterogeneous
    pipeline-parallel stages (e.g. attention-only vs recurrent-only stages) see
    identical cross-stage ``MambaSpec`` and ``AttentionSpec`` geometry.
    """
    mamba_specs = {name: spec for name, spec in kv_cache_specs.items() if isinstance(spec, MambaSpec)}
    if not mamba_specs:
        return kv_cache_specs
    if attention_layer_names is None and any(
        isinstance(spec, AscendSFAIndexerCacheSpec) for spec in kv_cache_specs.values()
    ):
        # K3.1 recurrent padding is settled by NPUPlatform before layers build
        # their specs; keep indexer and draft pages at their native sizes.
        return kv_cache_specs

    specs = dict(kv_cache_specs)
    if attention_layer_names is None:
        attention_layer_names = {
            name
            for name, spec in specs.items()
            if isinstance(spec, AttentionSpec)
            and not isinstance(spec, AscendSFAIndexerCacheSpec)
            and not name.endswith((_SPARSE_SUMMARY_SUFFIX,))
        }
    if descriptor_native:
        common_page_size = max(spec.page_size_bytes for spec in mamba_specs.values())
    else:
        _align_hybrid_attention_page_sizes(specs)
        common_page_size = max(spec.page_size_bytes for spec in specs.values())

    for name in attention_layer_names:
        spec = specs[name]
        if spec.page_size_bytes < common_page_size:
            specs[name] = replace(spec, page_size_padded=common_page_size)
    if not descriptor_native:
        for name, spec in mamba_specs.items():
            if spec.page_size_bytes < common_page_size:
                specs[name] = replace(spec, page_size_padded=common_page_size)
    return specs
