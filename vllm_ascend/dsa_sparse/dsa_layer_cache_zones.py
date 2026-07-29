"""DSA 每层 HBM cache zone 的发现、注册与稳定性校验。

首次执行某个 attention layer 时，本模块从 ``ForwardContext`` 解析该层 MLA
noPE/ROPE resident cache、可选 indexer cache 和层内 block size，并按 layer id
注册为 worker-lifetime 资源；后续 forward 直接查表，同时拒绝同一 layer 的
cache tensor 地址或布局发生漂移。

这里不承载请求跨 step 状态，也不构造 model-forward batch。请求行语义属于
``dsa_input_batch_state.py``，数据契约属于 ``dsa_forward_batch.py``，动态
eager/固定地址 row-mode 物化分别属于 builder/runtime 模块。
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.forward_context import ForwardContext


@dataclass(frozen=True)
class LayerCacheZones:
    nopek_cache_zone: object
    ropek_cache_zone: object
    indexer_cache_zone: object | None
    layerwise_global_block_size: int | None


class DSALayerCacheRegistry:
    """Persistent cache-zone registry for one DSA worker instance.

    KV cache tensors are allocated during worker/model-runner initialization
    and are expected to stay stable for the worker lifetime. DSA keeps these
    layer zones here so begin/finish hooks can use a small layer id lookup
    instead of re-discovering and overwriting cache bindings every layer call.

    If a later forward sees different cache tensors for the same layer, that is
    treated as a lifecycle violation and reported explicitly. The intended
    recovery path for a true KV cache rebuild is to recreate/reinitialize the
    DSA worker-side manager, not to silently reuse stale residency metadata.
    """

    def __init__(self, num_layers: int | None = None) -> None:
        # Layer ids are dense model layer indices, so a list gives the hot
        # attention_begin path one direct index operation instead of a dict
        # lookup. The list can still grow in tests or unusual initialization
        # paths where the final layer count is not known yet.
        initial_layers = 0 if num_layers is None else max(0, int(num_layers))
        self._cache_zones_by_layer: list[LayerCacheZones | None] = [
            None for _ in range(initial_layers)
        ]

    def _ensure_layer_capacity(self, layer_id: int) -> None:
        if layer_id < 0:
            raise RuntimeError(f"DSA got invalid negative layer id {layer_id}")
        missing = layer_id + 1 - len(self._cache_zones_by_layer)
        if missing > 0:
            self._cache_zones_by_layer.extend([None] * missing)

    @staticmethod
    def _same_cache_object(left: object, right: object) -> bool:
        if left is right:
            return True
        left_data_ptr = getattr(left, "data_ptr", None)
        right_data_ptr = getattr(right, "data_ptr", None)
        if callable(left_data_ptr) and callable(right_data_ptr):
            try:
                return (
                    left_data_ptr() == right_data_ptr()
                    and getattr(left, "shape", None)
                    == getattr(right, "shape", None)
                    and getattr(left, "dtype", None)
                    == getattr(right, "dtype", None)
                    and getattr(left, "device", None)
                    == getattr(right, "device", None)
                )
            except Exception:
                return False
        return False

    @classmethod
    def _same_cache_zones(cls, left: LayerCacheZones,
                          right: LayerCacheZones) -> bool:
        return (
            cls._same_cache_object(left.nopek_cache_zone,
                                   right.nopek_cache_zone)
            and cls._same_cache_object(left.ropek_cache_zone,
                                       right.ropek_cache_zone)
            and cls._same_cache_object(left.indexer_cache_zone,
                                       right.indexer_cache_zone)
            and left.layerwise_global_block_size
            == right.layerwise_global_block_size
        )

    def bind_or_validate(self, layer_id: int,
                         cache_zones: LayerCacheZones) -> LayerCacheZones:
        """Bind a layer once, then verify later observations are identical."""
        layer_id = int(layer_id)
        self._ensure_layer_capacity(layer_id)
        existing = self._cache_zones_by_layer[layer_id]
        if existing is None:
            self._cache_zones_by_layer[layer_id] = cache_zones
            return cache_zones
        if not self._same_cache_zones(existing, cache_zones):
            raise RuntimeError(
                f"DSA layer cache zones changed for layer {layer_id}; "
                "KV cache tensors must stay stable for one worker lifetime")
        return existing

    def get(self, layer_id: int) -> LayerCacheZones | None:
        layer_id = int(layer_id)
        if layer_id < 0 or layer_id >= len(self._cache_zones_by_layer):
            return None
        return self._cache_zones_by_layer[layer_id]

    def require(self, layer_id: int) -> LayerCacheZones:
        cache_zones = self.get(layer_id)
        if cache_zones is None:
            raise RuntimeError(
                f"DSA layer cache registry has no zones for layer {layer_id}")
        return cache_zones


def _looks_like_cache_tensor(value: object) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "device")
    )


def _select_virtual_engine_cache(kv_cache: object,
                                 virtual_engine: int) -> object:
    if isinstance(kv_cache, (tuple, list)):
        if (
            len(kv_cache) >= 2
            and _looks_like_cache_tensor(kv_cache[0])
            and _looks_like_cache_tensor(kv_cache[1])
        ):
            return kv_cache
        if not kv_cache:
            return kv_cache
        engine = max(0, min(int(virtual_engine), len(kv_cache) - 1))
        return kv_cache[engine]
    return kv_cache


def resolve_layer_cache_zones(
        layer_name: str,
        forward_context: ForwardContext,
) -> LayerCacheZones:
    attn = forward_context.no_compile_layers[layer_name]
    virtual_engine = int(getattr(forward_context, "virtual_engine", 0) or 0)
    sfa_cache = _select_virtual_engine_cache(attn.mla_attn.kv_cache,
                                             virtual_engine)
    if not isinstance(sfa_cache, (tuple, list)) or len(sfa_cache) < 2:
        raise RuntimeError(
            f"DSA requires MLA cache zones for {layer_name}, got "
            f"{type(sfa_cache).__name__}")
    nopek_cache_zone = sfa_cache[0]
    ropek_cache_zone = sfa_cache[1]

    impl = getattr(attn.mla_attn, "impl", None)
    indexer_layer_name = getattr(impl, "indexer_k_cache_layer_name", None)
    indexer_cache_zone = None
    if indexer_layer_name is not None:
        indexer_layer = forward_context.no_compile_layers.get(
            indexer_layer_name)
        if indexer_layer is not None:
            indexer_cache_zone = _select_virtual_engine_cache(
                indexer_layer.kv_cache, virtual_engine)
            if isinstance(indexer_cache_zone, (tuple, list)):
                indexer_cache_zone = (indexer_cache_zone[0]
                                      if indexer_cache_zone else None)
    if indexer_cache_zone is None and len(sfa_cache) > 2:
        indexer_cache_zone = sfa_cache[2]

    shape = getattr(nopek_cache_zone, "shape", None)
    layerwise_global_block_size = (
        int(shape[0]) if shape is not None and len(shape) > 0 else None)
    return LayerCacheZones(
        nopek_cache_zone=nopek_cache_zone,
        ropek_cache_zone=ropek_cache_zone,
        indexer_cache_zone=indexer_cache_zone,
        layerwise_global_block_size=layerwise_global_block_size,
    )
