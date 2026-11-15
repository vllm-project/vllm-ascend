# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec


def _align_hybrid_attention_page_sizes(kv_cache_specs: dict[str, KVCacheSpec]) -> None:
    """Align dense K/V segments before padding a shared Attention/Mamba pool."""
    # MLA and compressed-cache subclasses manage their own storage layouts.
    attention_specs = {
        name: spec for name, spec in kv_cache_specs.items() if type(spec) in (FullAttentionSpec, SlidingWindowSpec)
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


def align_hybrid_cache_pages(
    kv_cache_specs: dict[str, KVCacheSpec],
    *,
    descriptor_native: bool = False,
    excluded_attention_layers: set[str] | None = None,
) -> dict[str, KVCacheSpec]:
    """Apply the shared-page geometry before grouping attention and state caches."""
    mamba_specs = {name: spec for name, spec in kv_cache_specs.items() if isinstance(spec, MambaSpec)}
    if not mamba_specs:
        return kv_cache_specs

    specs = kv_cache_specs.copy()
    if descriptor_native:
        common_page_size = max(spec.page_size_bytes for spec in mamba_specs.values())
    else:
        _align_hybrid_attention_page_sizes(specs)
        common_page_size = max(spec.page_size_bytes for spec in specs.values())

    excluded_attention_layers = excluded_attention_layers or set()
    for name, spec in specs.items():
        align_attention = (
            isinstance(spec, AttentionSpec)
            and not isinstance(spec, AscendSFAIndexerCacheSpec)
            and name not in excluded_attention_layers
        )
        align_state = not descriptor_native and isinstance(spec, MambaSpec)
        if (align_attention or align_state) and spec.page_size_bytes < common_page_size:
            specs[name] = replace(spec, page_size_padded=common_page_size)
    return specs
