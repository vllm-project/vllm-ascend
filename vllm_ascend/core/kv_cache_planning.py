# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Physical page alignment applied in Core planning before KV cache grouping."""

from collections.abc import Iterable, Mapping
from dataclasses import replace

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec

_SPARSE_SUMMARY_SUFFIX = ".sparse_summary_cache"


def _is_dense_kv_attention_spec(layer_name: str, spec: KVCacheSpec) -> bool:
    """Whether a layer uses contiguous ``[all K blocks][all V blocks]`` storage."""
    return type(spec) in (FullAttentionSpec, SlidingWindowSpec) and not layer_name.endswith((_SPARSE_SUMMARY_SUFFIX,))


def validate_shared_backing_kv_ratios(kv_cache_specs: Mapping[str, KVCacheSpec]) -> None:
    """Reject incompatible K/V splits among dense layers sharing one physical backing."""
    attention_specs = {name: spec for name, spec in kv_cache_specs.items() if _is_dense_kv_attention_spec(name, spec)}
    if len(attention_specs) <= 1:
        return
    reference = max(attention_specs.values(), key=lambda spec: spec.real_page_size_bytes)
    for layer_name, spec in attention_specs.items():
        if spec.head_size * reference.head_size_v != reference.head_size * spec.head_size_v:
            raise ValueError(
                f"Cannot align hybrid attention K/V pages for {layer_name}: "
                "the shared contiguous cache requires matching K/V size ratios "
                "and divisible unpadded page sizes."
            )


def _align_hybrid_attention_page_sizes(
    kv_cache_specs: dict[str, KVCacheSpec],
    minimum_page_size: int = 0,
    *,
    check_kv_ratios: bool = True,
) -> None:
    """Align dense K/V segments before padding a shared Attention/Mamba pool."""
    del minimum_page_size
    if any(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in kv_cache_specs.values()):
        # Paired SFA has two physical page classes. Row byte ratios must never
        # change the logical token ownership or checkpoint granularity.
        return
    # MLA, compressed-cache subclasses, and Step5 auxiliary summaries manage
    # their own storage layouts.
    attention_specs = {name: spec for name, spec in kv_cache_specs.items() if _is_dense_kv_attention_spec(name, spec)}
    if not attention_specs:
        return
    reference = max(attention_specs.values(), key=lambda spec: spec.real_page_size_bytes)
    page_size = reference.real_page_size_bytes
    for layer_name, spec in attention_specs.items():
        # Ascend packs each layer as [all K blocks][all V blocks]. Equal
        # padded pages alone do not isolate block IDs: a smaller SWA K segment
        # can overlap another request's full-attention V block. Grow scheduler
        # blocks before padding; the backend still splits them into kernel blocks.
        if page_size % spec.real_page_size_bytes != 0:
            raise ValueError(
                f"Cannot align hybrid attention K/V pages for {layer_name}: "
                "the shared contiguous cache requires matching K/V size ratios "
                "and divisible unpadded page sizes."
            )
    if check_kv_ratios:
        validate_shared_backing_kv_ratios(attention_specs)
    for layer_name, spec in attention_specs.items():
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
    check_kv_ratios: bool = True,
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
        _align_hybrid_attention_page_sizes(specs, check_kv_ratios=check_kv_ratios)
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


def effective_kv_cache_specs_for_layout(
    vllm_config: VllmConfig,
    kv_cache_specs: Iterable[KVCacheSpec] | None,
) -> tuple[KVCacheSpec, ...] | None:
    """Preview planned page sizes for ``resolve_kv_cache_layout`` without mutating owners.

    ``EngineCore`` resolves the global KV cache layout before calling
    ``get_global_kv_cache_groups``. Returning planned copies here keeps layout
    selection and grouping consistent on effective page sizes (e.g. DCP-sharded
    target + replicated draft + Mamba state) while leaving owner specs intact.
    """
    if kv_cache_specs is None:
        return None
    specs_tuple = tuple(kv_cache_specs)
    if not specs_tuple:
        return specs_tuple

    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) == "kimi_k31" or any(
        isinstance(spec, AscendSFAIndexerCacheSpec) for spec in specs_tuple
    ):
        return specs_tuple

    has_circular_buffer = any(isinstance(spec, CircularBufferSpec) for spec in specs_tuple)
    has_mamba = any(isinstance(spec, MambaSpec) for spec in specs_tuple)
    if has_circular_buffer and not has_mamba:
        # Step5 groups main layers without unifying auxiliary summary/ring pages.
        return specs_tuple

    synthetic_specs: dict[str, KVCacheSpec] = {f"__layout_spec_{i}": spec for i, spec in enumerate(specs_tuple)}
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if getattr(scheduler_config, "disable_hybrid_kv_cache_manager", False):
        from vllm.v1.core.kv_cache_planning import unify_hybrid_kv_cache_specs

        unify_hybrid_kv_cache_specs(synthetic_specs)

    from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec

    if any(is_glm5_next_cache_spec(spec) for spec in synthetic_specs.values()):
        from vllm_ascend.models.glm5next.cache_config import _align_glm5_next_cache_specs

        try:
            _align_glm5_next_cache_specs(synthetic_specs)
        except ValueError:
            return specs_tuple
        return tuple(synthetic_specs.values())

    if any(isinstance(spec, CircularBufferSpec) for spec in synthetic_specs.values()) and any(
        isinstance(spec, MambaSpec) for spec in synthetic_specs.values()
    ):
        planned = align_hybrid_kv_cache_specs(synthetic_specs, descriptor_native=True, check_kv_ratios=False)
        return tuple(planned.values())

    if not any(isinstance(spec, MambaSpec) for spec in synthetic_specs.values()):
        return tuple(synthetic_specs.values())

    try:
        planned = align_hybrid_kv_cache_specs(synthetic_specs, check_kv_ratios=False)
    except ValueError:
        return specs_tuple
    return tuple(planned.values())
