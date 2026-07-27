from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.kv_cache_interface import KVCacheTensor

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)


def _make_vllm_config(num_layers: int, num_shared_buffers: int):
    model_config = MagicMock()
    model_config.get_num_layers.return_value = num_layers
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": num_shared_buffers,
            },
        ),
        model_config=model_config,
        parallel_config=MagicMock(),
    )


def test_no_reuse_skips_topology_validation():
    original_tensors = [
        KVCacheTensor(size=16, shared_by=["model.layers.0.self_attn"]),
        KVCacheTensor(size=16, shared_by=["model.layers.1.self_attn"]),
        KVCacheTensor(size=16, shared_by=["model.mtp.0.self_attn"]),
    ]
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=original_tensors.copy(),
        kv_cache_groups=[object(), object()],
    )

    apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(2, 2))

    assert kv_cache_config.kv_cache_tensors == original_tensors


def test_base_layers_are_merged_into_shared_slots():
    original_tensors = [KVCacheTensor(size=16, shared_by=[f"model.layers.{layer}.self_attn"]) for layer in range(6)]
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=original_tensors,
        kv_cache_groups=[object()],
    )

    apply_layerwise_kv_cache_plan(kv_cache_config, _make_vllm_config(6, 2))

    assert [tensor.shared_by for tensor in kv_cache_config.kv_cache_tensors] == [
        ["model.layers.0.self_attn"],
        ["model.layers.5.self_attn"],
        ["model.layers.1.self_attn", "model.layers.3.self_attn"],
        ["model.layers.2.self_attn", "model.layers.4.self_attn"],
    ]
