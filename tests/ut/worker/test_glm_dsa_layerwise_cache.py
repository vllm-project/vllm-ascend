# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import msgspec
import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker, push_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import MooncakeMetadataThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeTransferMetadataGroups
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import plan_layerwise_cache_reuse
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
    get_layerwise_reuse_backing_bytes,
)
from vllm_ascend.worker.v2 import attn_utils


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16], ids=["fp8_c8", "bf16"])
def test_glm_dsa_reuses_mla_and_indexer_and_preserves_views(monkeypatch, dtype):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    packed = dtype == torch.float8_e4m3fn
    cache_dtype = "fp8" if packed else "bfloat16"
    text_config = SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=78, index_topk=2048)
    transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "backend": "memcache",
            "use_layerwise": True,
            "layerwise_num_shared_buffers": 4,
            "layerwise_independent_layers": [],
        },
    )
    runtime = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
            use_mla=True,
            supports_kv_cache_grouping=False,
            get_total_num_hidden_layers=lambda: text_config.num_hidden_layers,
        ),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=0,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, cp_kv_cache_interleave_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=transfer_config,
    )
    main = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=656 if packed else 576,
        dtype=dtype,
        cache_sparse_sfa_c8=packed,
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=dtype,
        scale_dim=1 if packed else 0,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=packed,
    )
    main_entries = {layer: f"model.layers.{layer}.self_attn.attn" for layer in range(79)}
    indexer_entries = {layer: f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in (0, 1, *range(2, 79, 4))}
    specs = dict.fromkeys(main_entries.values(), main) | dict.fromkeys(indexer_entries.values(), indexer)
    draft_entries = {main_entries[78], indexer_entries[78]}
    for name in draft_entries:
        specs[name] = replace(specs[name], is_draft=True)

    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(runtime, specs)
    assert len(groups) == 1 and len(groups[0].layer_names) == 101
    bytes_per_block = builder._get_pool_bytes_per_block(runtime, groups)
    assert bytes_per_block == 5 * (main.page_size_bytes + indexer.page_size_bytes)
    config = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=3)
    layout = plan_layerwise_cache_reuse(config, runtime)
    assert layout is not None
    assert layout.prefetch_layer_map[10] == (0, 6)
    assert layout.draft_entry_names == draft_entries
    assert layout.reused_cache_names == specs.keys() - draft_entries
    assert len(layout.kv_cache_tensors) == 10
    expected_bytes = config.num_blocks * bytes_per_block
    assert get_layerwise_reuse_backing_bytes(config) == expected_bytes
    worker_config = apply_layerwise_kv_cache_plan(config, transfer_config, layout)

    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    attn_layers = {
        name: SimpleNamespace(
            prefix=name,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            get_attn_backend=lambda spec=spec: (
                AscendSFAIndexerBackend if isinstance(spec, AscendSFAIndexerCacheSpec) else AscendSFABackend
            ),
        )
        for name, spec in specs.items()
    }
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: attn_layers)
    monkeypatch.setattr(
        attn_utils,
        "_get_attention_kv_cache_dims",
        lambda name, _: (attn_layers[name].kv_lora_rank, attn_layers[name].qk_rope_head_dim),
    )
    allocations = []

    def allocate(size, alignment, device):
        backing = torch.zeros(size, dtype=torch.int8, device=device)
        allocations.append(backing)
        return backing

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", allocate)
    caches = attn_utils.allocate_kv_cache_main(worker_config, torch.device("cpu"), None, [128])
    (backing,) = allocations
    assert backing.numel() == expected_bytes
    assert all(view.untyped_storage().data_ptr() == backing.data_ptr() for views in caches.values() for view in views)
    for entries, source_layer in ((main_entries, 6), (indexer_entries, 0)):
        assert [view.data_ptr() for view in caches[entries[10]]] == [
            view.data_ptr() for view in caches[entries[source_layer]]
        ]
    for name in draft_entries:
        assert all(name not in tensor.layers or tensor.layers == [name] for tensor in layout.kv_cache_tensors)

    for name in main_entries.values():
        components = caches[name]
        expected_widths = (656,) if packed else (512, 64)
        assert tuple(component.shape[-1] for component in components) == expected_widths
        assert all(component.is_contiguous() and component.dtype == dtype for component in components)
        if not packed:
            nope, rope = components
            assert rope.data_ptr() == nope.data_ptr() + nope.numel() * nope.element_size()
    for name in indexer_entries.values():
        key, *scales = caches[name]
        assert key.is_contiguous() and key.dtype == dtype
        if packed:
            (scale,) = scales
            assert scale.is_contiguous()
            assert scale.data_ptr() == key.data_ptr() + key.numel() * key.element_size()
        else:
            assert not scales

    # Allocate D with the same cache specs and the normal, non-layerwise layout.
    decode_transfer = KVTransferConfig(kv_connector="AscendStoreConnector", kv_role="kv_consumer")
    decode_runtime = SimpleNamespace(**(vars(runtime) | {"kv_transfer_config": decode_transfer}))
    decode_config = builder.get_kv_cache_config_from_groups(decode_runtime, groups, num_blocks=5)
    with monkeypatch.context() as decode_patch:
        decode_patch.setattr(attn_utils, "get_current_vllm_config", lambda: decode_runtime)
        decode_caches = attn_utils.allocate_kv_cache_main(decode_config, torch.device("cpu"), None, [128])

    monkeypatch.setattr(base_worker.global_te, "register_buffer", lambda *_: None)
    producer = _register_push_worker(runtime, worker_config, caches, "p")
    consumer = _register_push_worker(decode_runtime, decode_config, decode_caches, "d")
    handshake = MooncakeMetadataThread(
        "127.0.0.1", 9000, "d", {0: consumer.transfer_metadata}, 1, 1, 1, 1, False, Event()
    )
    remote = msgspec.msgpack.decode(handshake.encoded_metadata, type=MooncakeTransferMetadataGroups)
    request = push_worker.LayerTransferRequest("request", (1, 2), (2, 4), 0, 2 * main.block_size)
    ready = SimpleNamespace(synchronize=lambda: None)

    def copy(_session, sources, destinations, sizes):
        writes = list(zip(sources, destinations, sizes, strict=True))
        assert writes == expected_writes
        for source, destination, size in writes:
            ctypes.memmove(destination, source, size)
        return 0

    producer.engine = SimpleNamespace(batch_transfer_sync_write=copy)
    for name in (main_entries[10], indexer_entries[10]):
        pairs = [
            (source[p_block], destination[d_block])
            for source, destination in zip(caches[name], decode_caches[name], strict=True)
            for p_block, d_block in zip(request.local_block_ids, request.remote_block_ids, strict=True)
        ]
        for value, (source, _) in enumerate(pairs, start=1):
            source.view(torch.uint8).fill_(value)
        expected_writes = [(source.data_ptr(), destination.data_ptr(), source.nbytes) for source, destination in pairs]
        batch = push_worker.LayerTransferBatch(name, (0, 0, 0), remote, (request,), ready)
        assert producer._execute_layer_transfer(batch) is None
        for source, destination in pairs:
            torch.testing.assert_close(source.view(torch.uint8), destination.view(torch.uint8))


def _register_push_worker(runtime, config, caches, engine_id):
    worker = push_worker.MooncakePushConnectorWorker.__new__(push_worker.MooncakePushConnectorWorker)
    worker.vllm_config = runtime
    worker.kv_cache_config = config
    worker.ascend_config = SimpleNamespace(kvpp_config=attn_utils.KVPPConfig.from_vllm_config(runtime))
    worker.block_size = config.kv_cache_groups[0].kv_cache_spec.block_size
    worker.tp_size = worker.pcp_size = worker.dcp_size = 1
    worker.tp_rank = worker.pcp_rank = 0
    worker.engine_id = engine_id
    worker.te_rpc_port = worker.handshake_port = 1
    worker.side_channel_host = "127.0.0.1"
    # Use the real metadata builder without starting NPU and network workers.
    base_worker.MooncakeBaseConnectorWorker.register_kv_caches(worker, caches)
    worker._layer_indices = {name: index for index, name in enumerate(worker.transfer_metadata.layer_names)}
    return worker


def test_indexer_slot_checks_capacity_after_scale_alignment():
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    spec = AscendSFAIndexerCacheSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float8_e4m3fn,
        scale_dim=1,
        scale_dtype=torch.float32,
        cache_sparse_li_c8=True,
    )
    # One K byte and four scale bytes fit tightly, but FP32 alignment moves the scale to offset four.
    raw_cache = torch.empty(spec.page_size_bytes, dtype=torch.int8)
    with pytest.raises(ValueError, match=r"requires 8 bytes for aligned K/scale views, got 5 bytes"):
        AscendSFAIndexerBackend.prepare_layerwise_cache_view(raw_cache, spec, SimpleNamespace(prefix=layer_name))
