# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared dual-version (vLLM v0.34.0 and #61452+) test factories for KV cache UTs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    CircularBufferManager,
    CrossAttentionManager,
    FullAttentionManager,
    MambaManager,
    RSWAManager,
    SinkFullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    declared_kwarg,
)
from vllm_ascend.utils import vllm_version_is


def is_v034() -> bool:
    return vllm_version_is("0.34.0")


def get_full_attention_spec_cls() -> Any:
    """Return FullAttentionSpec on v0.34.0, or AttentionSpec on #61452+."""
    if is_v034():
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        return FullAttentionSpec
    return AttentionSpec


def make_full_attention_spec(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    **spec_kwargs: Any,
) -> KVCacheSpec:
    cls = get_full_attention_spec_cls()
    return cls(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **spec_kwargs,
    )


def _wrap_layer_config(
    spec: KVCacheSpec,
    manager_cls: Any,
    *,
    is_draft: bool = False,
    **params: Any,
):
    from vllm.v1.kv_cache_interface import KVCacheLayerConfig

    return KVCacheLayerConfig(manager_cls.config(**params), spec, is_draft=is_draft)


def make_full_attention_layer_config(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    *,
    is_draft: bool = False,
    spec_cls: type | None = None,
    **spec_kwargs: Any,
):
    if spec_cls is None:
        spec_cls = get_full_attention_spec_cls()
    if is_v034():
        spec = spec_cls(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            **declared_kwarg(spec_cls, "is_draft", is_draft),
            **spec_kwargs,
        )
        return spec
    spec = spec_cls(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **spec_kwargs,
    )
    return _wrap_layer_config(spec, FullAttentionManager, is_draft=is_draft)


def make_mla_layer_config(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    *,
    is_draft: bool = False,
    ascend: bool = False,
    **spec_kwargs: Any,
):
    spec_cls = AscendMLAAttentionSpec if ascend else MLAAttentionSpec
    return make_full_attention_layer_config(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        is_draft=is_draft,
        spec_cls=spec_cls,
        **spec_kwargs,
    )


def make_sliding_window_layer_config(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    *,
    sliding_window: int = 64,
    extra_retained_tokens: int = 0,
    bounded_replay: bool = False,
    is_draft: bool = False,
    mla: bool = False,
    ascend: bool = False,
    **spec_kwargs: Any,
):
    if is_v034():
        if mla:
            if ascend:
                from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec as cls
            else:
                from vllm.v1.kv_cache_interface import SlidingWindowMLASpec as cls
        else:
            from vllm.v1.kv_cache_interface import SlidingWindowSpec as cls
        extra_kwargs: dict[str, Any] = {}
        if extra_retained_tokens:
            extra_kwargs.update(declared_kwarg(cls, "extra_retained_tokens", extra_retained_tokens))
        if bounded_replay:
            extra_kwargs.update(declared_kwarg(cls, "bounded_replay", bounded_replay))
        extra_kwargs.update(declared_kwarg(cls, "is_draft", is_draft))
        return cls(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            sliding_window=sliding_window,
            **extra_kwargs,
            **spec_kwargs,
        )
    if mla:
        spec_cls = AscendMLAAttentionSpec if ascend else MLAAttentionSpec
    else:
        spec_cls = AttentionSpec
    spec = spec_cls(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **spec_kwargs,
    )
    return _wrap_layer_config(
        spec,
        SlidingWindowManager,
        is_draft=is_draft,
        sliding_window=sliding_window,
        extra_retained_tokens=extra_retained_tokens,
        bounded_replay=bounded_replay,
    )


def make_rswa_layer_config(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    *,
    rswa_window: int = 64,
    is_draft: bool = False,
    **spec_kwargs: Any,
):
    if is_v034():
        from vllm.v1.kv_cache_interface import RSWASpec

        return RSWASpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            rswa_window=rswa_window,
            **declared_kwarg(RSWASpec, "is_draft", is_draft),
            **spec_kwargs,
        )
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **spec_kwargs,
    )
    return _wrap_layer_config(spec, RSWAManager, is_draft=is_draft, rswa_window=rswa_window)


def make_chunked_local_attention_layer_config(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    *,
    attention_chunk_size: int = 64,
    is_draft: bool = False,
    **spec_kwargs: Any,
):
    if is_v034():
        from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec

        return ChunkedLocalAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            attention_chunk_size=attention_chunk_size,
            **declared_kwarg(ChunkedLocalAttentionSpec, "is_draft", is_draft),
            **spec_kwargs,
        )
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **spec_kwargs,
    )
    return _wrap_layer_config(
        spec,
        ChunkedLocalAttentionManager,
        is_draft=is_draft,
        attention_chunk_size=attention_chunk_size,
    )


def make_cross_attention_layer_config(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    *,
    is_draft: bool = False,
    **spec_kwargs: Any,
):
    if is_v034():
        from vllm.v1.kv_cache_interface import CrossAttentionSpec

        return CrossAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            **spec_kwargs,
        )
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **spec_kwargs,
    )
    return _wrap_layer_config(spec, CrossAttentionManager, is_draft=is_draft)


def make_sink_full_attention_layer_config(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    *,
    sink_len: int = 4,
    is_draft: bool = False,
    **spec_kwargs: Any,
):
    if is_v034():
        from vllm.v1.kv_cache_interface import SinkFullAttentionSpec

        return SinkFullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            sink_len=sink_len,
            **spec_kwargs,
        )
    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **spec_kwargs,
    )
    return _wrap_layer_config(spec, SinkFullAttentionManager, is_draft=is_draft, sink_len=sink_len)


def make_circular_buffer_layer_config(
    block_size: int = 16,
    num_kv_heads: int = 1,
    head_size: int = 16,
    dtype: torch.dtype = torch.float16,
    *,
    ascend: bool = False,
    is_draft: bool = False,
    **spec_kwargs: Any,
):
    if is_v034():
        if ascend:
            from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec as cls
        else:
            from vllm.v1.kv_cache_interface import CircularBufferSpec as cls
        return cls(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            **spec_kwargs,
        )
    from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager

    spec = AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        **spec_kwargs,
    )
    manager_cls = AscendCircularBufferManager if ascend else CircularBufferManager
    return _wrap_layer_config(spec, manager_cls, is_draft=is_draft)


def make_mamba_layer_config(
    block_size: int = 16,
    shapes: tuple[tuple[int, ...], ...] = ((4, 8),),
    dtypes: tuple[torch.dtype, ...] = (torch.float16,),
    *,
    mamba_cache_mode: str = "none",
    num_speculative_blocks: int = 0,
    num_prefill_checkpoint_blocks: int = 0,
    prefill_checkpoint_alignment: int | None = None,
    is_draft: bool = False,
    **spec_kwargs: Any,
):
    if is_v034():
        return MambaSpec(
            block_size=block_size,
            shapes=shapes,
            dtypes=dtypes,
            mamba_cache_mode=mamba_cache_mode,
            num_speculative_blocks=num_speculative_blocks,
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
            prefill_checkpoint_alignment=prefill_checkpoint_alignment,
            **spec_kwargs,
        )
    spec = MambaSpec(
        block_size=block_size,
        shapes=shapes,
        dtypes=dtypes,
        **spec_kwargs,
    )
    return _wrap_layer_config(
        spec,
        MambaManager,
        is_draft=is_draft,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        prefill_checkpoint_alignment=prefill_checkpoint_alignment,
    )


def _default_manager_config_for_spec(spec: KVCacheSpec):
    if isinstance(spec, MambaSpec):
        return MambaManager.config()
    return FullAttentionManager.config()


def make_kv_cache_group(
    layer_names_or_configs: Sequence[str] | Mapping[str, Any],
    spec_or_layer_config: Any = None,
    *,
    manager_config: Any = None,
    is_eagle_group: bool = False,
    host_resident: bool = False,
    enable_kv_transfer: bool = True,
    role: Any = None,
):
    """Build KVCacheGroup (#61452+) or KVCacheGroupSpec (v0.34.0)."""
    extra_kwargs: dict[str, Any] = {
        "is_eagle_group": is_eagle_group,
        "host_resident": host_resident,
        "enable_kv_transfer": enable_kv_transfer,
    }
    if role is not None:
        extra_kwargs["role"] = role

    if is_v034():
        from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

        if isinstance(layer_names_or_configs, Mapping):
            layer_names = list(layer_names_or_configs.keys())
            specs = dict(layer_names_or_configs)
            first = next(iter(specs.values()))
            if all(spec == first for spec in specs.values()):
                kv_cache_spec = first
            else:
                kv_cache_spec = UniformTypeKVCacheSpecs.from_specs(specs)
        else:
            layer_names = list(layer_names_or_configs)
            kv_cache_spec = spec_or_layer_config
        return KVCacheGroupSpec(
            layer_names=layer_names,
            kv_cache_spec=kv_cache_spec,
            **extra_kwargs,
        )

    from vllm.v1.kv_cache_interface import KVCacheGroup, KVCacheLayerConfig

    if isinstance(layer_names_or_configs, Mapping):
        first_val = next(iter(layer_names_or_configs.values()))
        if isinstance(first_val, KVCacheLayerConfig):
            return KVCacheGroup.from_layers(layer_names_or_configs, **extra_kwargs)
        specs_map = dict(layer_names_or_configs)
        cfg = manager_config or _default_manager_config_for_spec(first_val)
        return KVCacheGroup(
            kv_cache_specs=specs_map,
            manager_config=cfg,
            block_size=first_val.block_size,
            dcp_sharded=first_val.dcp_sharded,
            **extra_kwargs,
        )

    layer_names = list(layer_names_or_configs)
    if isinstance(spec_or_layer_config, KVCacheLayerConfig):
        return KVCacheGroup.from_layers(
            {name: spec_or_layer_config for name in layer_names},
            **extra_kwargs,
        )
    spec = spec_or_layer_config
    cfg = manager_config or _default_manager_config_for_spec(spec)
    return KVCacheGroup(
        kv_cache_specs={name: spec for name in layer_names},
        manager_config=cfg,
        block_size=spec.block_size,
        dcp_sharded=spec.dcp_sharded,
        **extra_kwargs,
    )


def get_group_spec(group: Any, layer_name: str | None = None) -> KVCacheSpec:
    """Return a layer's KVCacheSpec from a KVCacheGroup (#61452+) or KVCacheGroupSpec (v0.34.0)."""
    if is_v034():
        spec = group.kv_cache_spec
        if layer_name is not None and hasattr(spec, "kv_cache_specs"):
            return spec.kv_cache_specs[layer_name]
        return spec
    if layer_name is not None:
        return group.kv_cache_specs[layer_name]
    return next(iter(group.kv_cache_specs.values()))


def with_is_draft(layer_config_or_spec: Any, is_draft: bool = True) -> Any:
    """Return a copy marked as draft on both v0.34.0 and #61452+."""
    if is_v034():
        if hasattr(layer_config_or_spec, "is_draft"):
            return replace(layer_config_or_spec, is_draft=is_draft)
        return layer_config_or_spec
    return replace(layer_config_or_spec, is_draft=is_draft)
