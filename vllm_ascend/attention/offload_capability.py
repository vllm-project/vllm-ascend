# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small, model-neutral capability helpers used by layerwise offload."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_ascend.attention.cache_layout import LayerCacheLayoutDescriptor
    from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass(frozen=True)
class DsaOffloadCacheMemory:
    """Backend-owned memory accounting for one DSA cache layer.

    The planner only needs these aggregate values.  It must not interpret the
    physical cache tuple, since that interpretation is an attention backend
    concern and can differ between models or device layouts.
    """

    main_bytes_per_token: int
    device_bytes_per_block: int

    def __post_init__(self) -> None:
        if self.main_bytes_per_token <= 0:
            raise ValueError("main_bytes_per_token must be positive")
        if self.device_bytes_per_block <= 0:
            raise ValueError("device_bytes_per_block must be positive")


class DsaOffloadAttentionBackendCapability:
    """Explicit capability contract for attention backends serving DSA offload."""

    @classmethod
    def supports_dsa_offload(cls) -> bool:
        return False

    @staticmethod
    def is_dsa_offload_attention_impl(attn_impl: Any) -> bool:
        """Return whether ``attn_impl`` belongs to this offload backend."""

        del attn_impl
        return False

    @staticmethod
    def get_dsa_offload_layer_id(layer_name: str) -> int | None:
        """Return the base-model layer id represented by ``layer_name``."""

        del layer_name
        return None

    @staticmethod
    def get_dsa_offload_layer_count() -> int:
        raise NotImplementedError(
            "DSA offload backend does not provide a layer count"
        )

    @staticmethod
    def get_dsa_offload_group_specs(num_hidden_layers: int) -> Sequence[Any]:
        del num_hidden_layers
        raise NotImplementedError(
            "DSA offload backend does not provide cache group specifications"
        )

    @staticmethod
    def get_dsa_offload_default_pool_size() -> int:
        raise NotImplementedError(
            "DSA offload backend does not provide a default pool size"
        )

    @staticmethod
    def get_dsa_offload_topk() -> int | None:
        """Return the backend's DSA selection width, when configured."""

        return None

    @staticmethod
    def get_dsa_offload_cache_dimensions(attn_impl: Any) -> tuple[int, int]:
        del attn_impl
        raise NotImplementedError(
            "DSA offload backend does not provide cache dimensions"
        )

    @staticmethod
    def bind_dsa_offload_runtime(
        attn_impl: Any,
        runtime: Any,
        layer_id: int,
    ) -> None:
        del attn_impl, runtime, layer_id
        raise NotImplementedError(
            "DSA offload backend does not support runtime binding"
        )

    @staticmethod
    def bind_dsa_cache_layout_descriptor(
        attn_impl: Any,
        descriptor: LayerCacheLayoutDescriptor,
    ) -> None:
        del attn_impl, descriptor
        raise NotImplementedError(
            "DSA offload backend does not support cache descriptor binding"
        )

    @classmethod
    def build_layer_cache_descriptor(
        cls,
        *,
        layer_name: str,
        layer_id: int,
        group_id: int,
        cache_family_id: str,
        kv_cache_spec: KVCacheSpec,
        is_mtp_layer: bool = False,
    ) -> LayerCacheLayoutDescriptor:
        del cls, layer_name, layer_id, group_id, cache_family_id, kv_cache_spec
        del is_mtp_layer
        raise NotImplementedError(
            "DSA offload backend does not provide a cache layout descriptor"
        )

    @classmethod
    def get_dsa_offload_cache_memory(
        cls,
        *,
        layer_name: str,
        kv_cache_spec: KVCacheSpec,
    ) -> DsaOffloadCacheMemory:
        del cls, layer_name, kv_cache_spec
        raise NotImplementedError(
            "DSA offload backend does not provide cache memory accounting"
        )

    @classmethod
    def is_dsa_offload_speculative_layer(
        cls,
        *,
        layer_name: str,
        kv_cache_spec: Any,
    ) -> bool:
        del cls, layer_name, kv_cache_spec
        return False


def is_dsa_offload_backend(backend: Any) -> bool:
    """Return whether ``backend`` explicitly implements DSA offload support."""

    return (
        isinstance(backend, type)
        and issubclass(backend, DsaOffloadAttentionBackendCapability)
        and backend.supports_dsa_offload()
    )


def get_dsa_offload_backends(
    vllm_config: Any,
    layer_names: Sequence[str],
) -> dict[str, type[DsaOffloadAttentionBackendCapability]]:
    """Resolve the selected attention backend for each cache layer.

    When layers are available, resolving the backend from them keeps planning
    aligned with the backend that will consume the cache at runtime.  The
    centralized engine-core planner has no materialized layers, so it uses the
    same vLLM selector as a fallback.
    """

    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase, layer_names)
    backends: dict[str, type[DsaOffloadAttentionBackendCapability]] = {}
    if len(layers) != len(layer_names):
        # The centralized KV planner runs in engine-core, where model layers
        # are not materialized. Resolve the same backend through vLLM's
        # selector in that process. Derive the selector inputs from the model
        # config so another capable backend is not forced through SFA's MLA
        # and sparse-attention selection path.
        from vllm.config import set_current_vllm_config
        from vllm.v1.attention.selector import get_attn_backend
        from vllm_ascend.ops.dsa_offload import get_dsa_config_value

        model_config = vllm_config.model_config
        model_dtype = model_config.dtype
        use_mla = bool(get_dsa_config_value(model_config, "use_mla", False))
        use_sparse = get_dsa_config_value(model_config, "index_topk") is not None
        with set_current_vllm_config(vllm_config):
            backend = get_attn_backend(
                0,
                model_dtype,
                None,
                use_mla=use_mla,
                use_sparse=use_sparse,
                use_mm_prefix=model_config.is_mm_prefix_lm,
            )
        if not is_dsa_offload_backend(backend):
            raise NotImplementedError(
                "Layerwise DSA Host offload requires the selected attention "
                f"backend to support_dsa_offload=True; backend={backend!r}"
            )
        return {layer_name: backend for layer_name in layer_names}
    for layer_name in layer_names:
        layer = layers.get(layer_name)
        if layer is None:
            raise RuntimeError(
                f"DSA Host offload cannot find attention layer {layer_name}"
            )
        backend = layer.get_attn_backend()
        if not is_dsa_offload_backend(backend):
            raise NotImplementedError(
                "Layerwise DSA Host offload requires an attention backend with "
                f"supports_dsa_offload=True; layer={layer_name}, "
                f"backend={backend!r}"
            )
        backends[layer_name] = backend
    return backends


class CacheLayerRole(str, Enum):
    BASE = "base"
    SPECULATIVE = "speculative"


def resolve_cache_layer_role(
    layer_name: str,
    *,
    num_hidden_layers: int | None = None,
    is_eagle_group: bool = False,
    group_layer_count: int = 0,
) -> CacheLayerRole:
    """Resolve a cache layer role from its name and model depth."""
    lowered = layer_name.lower()
    if "mtp" in lowered or "draft" in lowered:
        return CacheLayerRole.SPECULATIVE
    if num_hidden_layers is not None:
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        from vllm.model_executor.models.utils import extract_layer_index

        if extract_layer_index(layer_name) >= num_hidden_layers:
            return CacheLayerRole.SPECULATIVE
    if is_eagle_group and group_layer_count == 1:
        return CacheLayerRole.SPECULATIVE
    return CacheLayerRole.BASE


def is_speculative_cache_layer(
    layer_name: str,
    *,
    num_hidden_layers: int | None = None,
    is_eagle_group: bool = False,
    group_layer_count: int = 0,
) -> bool:
    return (
        resolve_cache_layer_role(
            layer_name,
            num_hidden_layers=num_hidden_layers,
            is_eagle_group=is_eagle_group,
            group_layer_count=group_layer_count,
        )
        is CacheLayerRole.SPECULATIVE
    )
