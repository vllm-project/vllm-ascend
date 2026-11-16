# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport registration, byte restoration and old pool-format isolation."""

import copy
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.models.test_dsv4_layerwise import make_layout
from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import MemcacheBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker


def make_indexer_views():
    # A5 has a 4096-byte data payload followed by 128 bytes of float scales.
    raw = torch.arange(4 * 4224, dtype=torch.int64).to(torch.uint8)
    data = raw.as_strided((4, 32, 1, 128), (4224, 128, 128, 1))
    scale = raw.view(torch.float32).as_strided((4, 32, 1, 1), (1056, 1, 1, 1), 1024)
    full = raw.view(4, 32, 1, 132)
    return data, scale, full


def register_transport(caches, *, dsv4=True):
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.kv_cache_config = SimpleNamespace(dsv4_cache_format="format") if dsv4 else SimpleNamespace()
    worker.use_recoverssm = False
    worker._set_num_blocks = Mock()
    worker._register_dspark_prefix_cache = Mock()
    worker._register_kv_caches = Mock()
    worker.register_kv_caches(caches)
    return worker._register_kv_caches.call_args.args[0]


def test_layerwise_registration_preserves_kernel_views_and_other_groups():
    indexer = make_indexer_views()
    swa = torch.zeros(4, 32, 1, 512, dtype=torch.bfloat16)
    state = torch.zeros(4, 2, 1, 2048, dtype=torch.float32)
    caches = {
        "model.layers.2.self_attn.indexer.k_cache": indexer,
        "model.layers.2.self_attn.swa_cache": swa,
        "model.layers.2.self_attn.compressor.state_cache": state,
    }
    transferred = register_transport(caches)
    assert transferred is not caches
    assert caches["model.layers.2.self_attn.indexer.k_cache"] is indexer
    assert len(indexer) == 3
    assert len(transferred["model.layers.2.self_attn.indexer.k_cache"]) == 2
    assert transferred["model.layers.2.self_attn.indexer.k_cache"][0] is indexer[0]
    assert transferred["model.layers.2.self_attn.indexer.k_cache"][1] is indexer[1]
    assert transferred["model.layers.2.self_attn.swa_cache"] is swa
    assert transferred["model.layers.2.self_attn.compressor.state_cache"] is state


def test_saved_indexer_restores_full_kernel_view_after_source_reuse():
    name = "model.layers.2.self_attn.indexer.k_cache"
    source = make_indexer_views()
    expected = source[2].clone()
    views = register_transport({name: source})[name]
    # Immutable CPU copies stand in for the external byte store. Mutating the
    # source models the layer's HBM being reused before a second request loads.
    pages = [torch.cat([view[page].view(torch.uint8).reshape(-1) for view in views]) for page in range(4)]
    assert all(page.numel() == 4224 for page in pages)
    source[2].zero_()
    destination = make_indexer_views()
    destination[2].zero_()
    for page, payload in enumerate(pages):
        offset = 0
        for view in register_transport({name: destination})[name]:
            target = view[page].view(torch.uint8).reshape(-1)
            target.copy_(payload[offset : offset + target.numel()])
            offset += target.numel()
    assert torch.equal(destination[2], expected)
    assert torch.count_nonzero(source[2]) == 0


def test_non_dsv4_registration_keeps_existing_three_component_behavior():
    caches = {"other_model.cache": make_indexer_views()}
    assert register_transport(caches, dsv4=False) is caches


def test_group_metadata_and_pool_size_count_indexer_payload_once():
    caches = {
        "model.layers.2.self_attn.attn": torch.zeros(4, 32, 1, 512, dtype=torch.bfloat16),
        "model.layers.2.self_attn.indexer.k_cache": make_indexer_views(),
    }
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.kv_caches = register_transport(caches)
    worker.kv_cache_config = None
    worker.hf_config = SimpleNamespace(num_hidden_layers=43)
    worker.num_blocks, worker.num_layers, worker.total_layers = 4, 43, 43
    worker.use_layerwise, worker.use_kvpp = True, False
    worker.put_step, worker.dcp_size, worker.pp_rank = 8, 1, 0
    worker._global_to_local_layer = {}
    worker.group_kv_caches_base_addr, worker.group_block_len, worker.group_block_stride = {}, {}, {}
    worker.group_layer_cache_entry_offsets, worker.group_cache_entry_segments, worker.group_num_layers = {}, {}, {}
    worker._infer_cache_group_metadata(0, list(caches))
    assert worker.group_block_len[0] == [32768, 4096, 128]
    assert worker.group_block_stride[0] == [32768, 4224, 4224]
    assert worker.group_layer_cache_entry_offsets[0] == [0, 3]
    assert worker._global_group_alloc_size(0) == 43 * (32768 + 4096 + 128)


def test_invalid_indexer_alias_is_rejected_before_transport_registration():
    data, scale, full = make_indexer_views()
    with pytest.raises(ValueError, match="not disjoint"):
        register_transport({"model.layers.2.self_attn.indexer.k_cache": (data, scale, full.clone())})


@pytest.mark.parametrize("model", ["flash", "pro"])
@pytest.mark.parametrize("cache_format", ["a3", "a5_bf16", "a5_packed"])
def test_v1_objects_are_isolated_but_resident_and_rotating_v2_keys_match(model, cache_format):
    keys = []
    for buffers, blocks, independent in ((None, 16, []), (1, 8, []), (3, 32, [0, 1]), (4, 64, [0, 1, 4])):
        runtime, groups, _ = make_layout(model, cache_format=cache_format, buffers=buffers, independent=independent)
        if buffers is None:
            runtime.kv_transfer_config.kv_connector_extra_config.pop("layerwise_num_shared_buffers")
        config = AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(runtime, groups, num_blocks=blocks)
        # Reconstruct the published v1 format identity, independently of v2.
        old_groups = [
            [
                (name, type(spec).__name__, asdict(spec))
                for name in group.layer_names
                for spec in (group.kv_cache_spec.kv_cache_specs[name],)
            ]
            for group in config.kv_cache_groups
        ]
        legacy = copy.copy(config)
        payload = json.dumps(("dsv4-components-v1", old_groups), default=str, sort_keys=True, separators=(",", ":"))
        legacy.dsv4_cache_format = hashlib.sha256(payload.encode()).hexdigest()[:24]
        assert legacy.dsv4_cache_format != config.dsv4_cache_format
        pair = []
        for item in (legacy, config):
            bound = MemcacheBackend.bind_layerwise_keys(
                vllm_config=runtime,
                kv_cache_config=item,
                model_name=model,
                use_hybrid=True,
                grouped_block_size=[128] * len(groups),
            )
            pair.append([bound.make_hit_check_keys(group, "same-prefix-hash", 1) for group in range(len(groups))])
        assert all(old != new for old, new in zip(*pair))
        keys.append(pair[1])
    assert all(key_set == keys[0] for key_set in keys[1:])
