from types import SimpleNamespace

from vllm_ascend.dsa_sparse.dsa_layer_cache_zones import (
    resolve_layer_cache_zones,
)


class _FakeCacheTensor:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = "bfloat16"
        self.device = "npu"


def test_resolve_layer_cache_zones_accepts_direct_mla_attention():
    layer_name = "model.layers.0.self_attn.attn"
    nopek_cache = _FakeCacheTensor((64, 16, 512))
    ropek_cache = _FakeCacheTensor((64, 16, 64))
    mla_attention = SimpleNamespace(
        kv_cache=(nopek_cache, ropek_cache),
        impl=SimpleNamespace(indexer_k_cache_layer_name=None),
    )
    forward_context = SimpleNamespace(
        no_compile_layers={layer_name: mla_attention},
        virtual_engine=0,
    )

    cache_zones = resolve_layer_cache_zones(layer_name, forward_context)

    assert cache_zones.nopek_cache_zone is nopek_cache
    assert cache_zones.ropek_cache_zone is ropek_cache
    assert cache_zones.indexer_cache_zone is None
    assert cache_zones.layerwise_global_block_size == 64
