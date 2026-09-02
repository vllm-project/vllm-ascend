from types import SimpleNamespace

import vllm_ascend.worker.kvpp_v1 as kvpp_v1_module
from vllm_ascend.worker.kvpp_v1 import KVPPV1Runtime
from vllm_ascend.worker.v2.kvpp import KVPPCacheLayout, KVPPRuntime


def test_kvpp_runtime_v1_disabled_prepare_is_noop():
    KVPPV1Runtime().prepare_forward(SimpleNamespace(), 1, [1])


def test_kvpp_runtime_v1_converts_cache_layout(monkeypatch):
    expected_runtime = KVPPRuntime()
    captured: dict[str, object] = {}

    def capture_cache_layout(_runtime_cls, **kwargs):
        captured.update(kwargs)
        return expected_runtime

    monkeypatch.setattr(
        KVPPRuntime,
        "create_from_cache_layout",
        classmethod(capture_cache_layout),
    )
    monkeypatch.setattr(
        kvpp_v1_module.KVPPConfig,
        "from_vllm_config",
        lambda _config: SimpleNamespace(size=2),
    )
    vllm_config = object()
    kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=("layer",))])
    static_forward_context = {"layer": object()}
    kv_caches = {"layer": object()}
    block_tables = [SimpleNamespace(blocks_per_phys_block=2, logical_block_size=128)]

    runtime = KVPPV1Runtime.create_from_kv_cache(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        static_forward_context=static_forward_context,
        kv_caches=kv_caches,
        block_tables=block_tables,
    )

    assert runtime._kvpp_runtime is expected_runtime
    assert captured["vllm_config"] is vllm_config
    assert captured["kv_cache_config"] is kv_cache_config
    assert captured["static_forward_context"] is static_forward_context
    assert captured["cache_layout"] == KVPPCacheLayout(
        layer_caches=kv_caches,
        physical_blocks_per_kv_block=(2,),
        tokens_per_block=(128,),
    )
