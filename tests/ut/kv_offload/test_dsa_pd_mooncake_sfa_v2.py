# SPDX-License-Identifier: Apache-2.0
"""Exercise real SFA cache specs through the V2 Host allocator entry point."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.kv_offload import dsa_pd_mooncake as registry
from vllm_ascend.worker.v2 import attn_utils


@pytest.fixture(autouse=True)
def clear_host_registry():
    registry.reset_dsa_pd_mooncake_kv_registries()
    attn_utils._dsa_pd_mooncake_pending_offloaded_ranges.clear()
    yield
    registry.reset_dsa_pd_mooncake_kv_registries()
    attn_utils._dsa_pd_mooncake_pending_offloaded_ranges.clear()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("scenario", ["host", "producer", "disabled", "baseline", "default_mode", "other_connector"])
def test_sfa_main_host_allocation_and_final_view_registration(monkeypatch, packed, scenario):
    main_name = "model.layers.0.self_attn.attn"
    indexer_name = "model.layers.0.self_attn.indexer.k_cache"
    main = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=6,
        dtype=torch.int8 if packed else torch.float16,
        cache_sparse_sfa_c8=packed,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        scale_dim=1,
        scale_dtype=torch.float16,
        sfa_dcp_replicated_indexer_size=1,
    )
    count = 2
    specs = {main_name: main, indexer_name: indexer}
    kv_config = KVCacheConfig(
        num_blocks=count,
        kv_cache_tensors=[
            KVCacheTensor(
                size=count * spec.page_size_bytes,
                layers=[name],
                layer_stride=count * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
            for name, spec in specs.items()
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec) for name, spec in specs.items()],
    )
    dsa = {
        "mode": "baseline" if scenario == "baseline" else "fused_overlap",
        "enable_cpu_kv_store": scenario != "disabled",
    }
    if scenario == "default_mode":
        dsa.pop("mode")
    config = SimpleNamespace(
        additional_config={"dsa_sparse_attention_config": dsa},
        kv_transfer_config=SimpleNamespace(
            kv_connector="OtherConnector" if scenario == "other_connector" else "MooncakeConnectorV2",
            is_kv_consumer=scenario != "producer",
        ),
        model_config=SimpleNamespace(
            dtype=torch.float16, hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace(index_topk=2048)
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(cache_dtype="int8" if packed else "auto"),
        quant_config=None,
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "_uses_single_raw_mla_cache", lambda *_args: False)
    monkeypatch.setattr(attn_utils, "enable_fa_quant", lambda *_args: False)
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_args: (4, 2))
    swapped_storage = []

    def swapped(shape, *, dtype):
        tensor = torch.empty(shape, dtype=dtype)
        swapped_storage.append(tensor)
        return tensor

    with patch.object(attn_utils, "empty_mooncake_cpu_kv_memory", side_effect=swapped) as allocator:
        raw = attn_utils._allocate_kv_cache(kv_config, shared_layers={}, device=torch.device("cpu"))
    expected = scenario == "host"
    assert allocator.call_count == ((1 if packed else 2) if expected else 0)
    backend = SimpleNamespace(get_kv_cache_shape=lambda n, b, h, d, _dtype="auto", **_kw: (n, b, h, d))
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=[
            SimpleNamespace(kv_cache_group_id=i, kv_cache_spec=spec, layer_names=[name], backend=backend)
            for i, (name, spec) in enumerate(specs.items())
        ],
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[128, 128],
        shared_kv_cache_layers={},
        kv_cache_config=kv_config,
    )
    attn_utils._register_dsa_pd_mooncake_final_kv_views(caches)
    main_parts = [caches[main_name]] if isinstance(caches[main_name], torch.Tensor) else list(caches[main_name])
    indexer_parts = list(caches[indexer_name])
    for tensor in main_parts:
        assert registry.is_dsa_pd_mooncake_offloaded_kv_buffer(tensor.data_ptr(), tensor.nbytes) == expected
        assert tensor.data_ptr() % (2 * 1024 * 1024) == 0
    for tensor in indexer_parts:
        assert not registry.is_dsa_pd_mooncake_offloaded_kv_buffer(tensor.data_ptr(), tensor.nbytes)


@pytest.mark.parametrize("requested", [False, True])
def test_host_request_cannot_silently_fall_back_to_device_layout(monkeypatch, requested):
    from vllm_ascend import ascend_config

    config = SimpleNamespace(
        additional_config={
            "dsa_sparse_attention_config": {
                "mode": "fused_overlap",
                "enable_cpu_kv_store": requested,
            }
        },
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True, kv_connector="MooncakeConnectorV2"),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    caches = {"main": torch.empty(64, dtype=torch.int8)}
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: {})
    monkeypatch.setattr(
        ascend_config,
        "get_ascend_config",
        lambda: SimpleNamespace(dsa_sparse_attention_config=SimpleNamespace(mode="fused_overlap")),
    )
    monkeypatch.setattr(attn_utils, "_allocate_kv_cache", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(attn_utils, "_reshape_kv_cache_v2", lambda **_kwargs: caches)
    kv_config = SimpleNamespace(kv_cache_groups=[])
    if requested:
        with pytest.raises(RuntimeError, match="no Host main KV was allocated"):
            attn_utils.allocate_kv_cache_main(kv_config, torch.device("cpu"), None, [])
    else:
        assert attn_utils.allocate_kv_cache_main(kv_config, torch.device("cpu"), None, []) is caches
