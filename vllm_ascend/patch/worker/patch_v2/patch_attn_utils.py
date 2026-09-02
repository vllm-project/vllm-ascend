import torch
import vllm
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.logger import init_logger

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)

logger = init_logger(__name__)
_reshape_debug_emitted = False


def _get_ascend_sfa_indexer_backend(_self):
    return AscendSFAIndexerBackend


_upstream_reshape_and_cache = AscendAttentionBackendImpl.reshape_and_cache


def _reshape_and_cache_with_diagnostics(self, *args, **kwargs):
    global _reshape_debug_emitted
    if not _reshape_debug_emitted:
        kv_cache = kwargs.get("kv_cache")
        if kv_cache is None and len(args) >= 4:
            kv_cache = args[3]
        logger.warning(
            "Ascend MRV2 attention cache binding: kv_cache=%s, key_cache=%s, value_cache=%s",
            (
                type(kv_cache).__name__,
                [tuple(item.shape) for item in kv_cache]
                if isinstance(kv_cache, tuple)
                else getattr(kv_cache, "shape", None),
            ),
            getattr(self.key_cache, "shape", None),
            getattr(self.value_cache, "shape", None),
        )
        _reshape_debug_emitted = True
    return _upstream_reshape_and_cache(self, *args, **kwargs)


AscendAttentionBackendImpl.reshape_and_cache = _reshape_and_cache_with_diagnostics


_upstream_allocate_kv_cache = vllm.v1.worker.gpu.attn_utils.allocate_kv_cache


def _allocate_kv_cache_for_ascend(*args, **kwargs):
    """Adapt vLLM's packed KV views to Ascend's split K/V interface.

    The current vLLM allocator exposes regular attention as one logical
    ``[B, H, N, K+V]`` view.  Ascend's paged-cache operator keeps K and V as
    separate four-dimensional tensors, so split the last dimension without
    copying the backing allocation.
    """
    kv_caches = _upstream_allocate_kv_cache(*args, **kwargs)

    kv_cache_config = args[0] if args else kwargs.get("kv_cache_config")
    layer_specs = {}
    if kv_cache_config is not None:
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            per_layer_specs = getattr(spec, "kv_cache_specs", None)
            for layer_name in group.layer_names:
                layer_specs[layer_name] = (
                    per_layer_specs.get(layer_name, spec)
                    if per_layer_specs is not None
                    else spec
                )

    logger.warning(
        "Ascend MRV2 KV cache allocation: %s",
        {
            layer_name: (
                tuple(kv_cache.shape),
                type(layer_specs.get(layer_name)).__name__,
                getattr(layer_specs.get(layer_name), "head_size", None),
                getattr(layer_specs.get(layer_name), "head_size_v", None),
            )
            for layer_name, kv_cache in kv_caches.items()
            if isinstance(kv_cache, torch.Tensor)
        },
    )

    for layer_name, kv_cache in kv_caches.items():
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 4:
            continue
        spec = layer_specs.get(layer_name)
        k_dim = getattr(spec, "head_size", None)
        v_dim = getattr(spec, "head_size_v", None) or k_dim
        if k_dim is None or v_dim is None:
            continue
        if kv_cache.shape[-1] != k_dim + v_dim:
            continue
        kv_caches[layer_name] = (
            kv_cache[..., :k_dim],
            kv_cache[..., k_dim : k_dim + v_dim],
        )
    logger.warning(
        "Ascend MRV2 KV cache allocation result: %s",
        {
            layer_name: tuple(item.shape for item in kv_cache)
            if isinstance(kv_cache, tuple)
            else tuple(kv_cache.shape)
            for layer_name, kv_cache in list(kv_caches.items())[:1]
        },
    )
    return kv_caches


DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils.allocate_kv_cache = _allocate_kv_cache_for_ascend
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
