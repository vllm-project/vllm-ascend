# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    apply_layerwise_kv_cache_plan,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("indexer_c8", [False, True])
def test_standardized_sfa_allocation_preserves_layerwise_aliases(reuse, indexer_c8):
    main_names = [f"model.layers.{i}.self_attn.attn" for i in range(4)] + ["model.mtp.0.self_attn.attn"]
    indexer_names = [f"model.layers.{i}.self_attn.indexer.k_cache" for i in (1, 2, 3)]
    main_spec = AscendMLAAttentionSpec(block_size=2, num_kv_heads=1, head_size=12, dtype=torch.bfloat16)
    indexer_spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8 if indexer_c8 else torch.bfloat16,
        scale_dim=1 if indexer_c8 else 0,
        scale_dtype=torch.float16,
        cache_sparse_li_c8=indexer_c8,
    )
    num_blocks = 3
    backing_size = (
        max(len(main_names) * main_spec.page_size_bytes, len(indexer_names) * indexer_spec.page_size_bytes) * num_blocks
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=backing_size,
                layers=names,
                layer_stride=num_blocks * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
            for names, spec in ((main_names, main_spec), (indexer_names, indexer_spec))
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)
            for names, spec in ((main_names, main_spec), (indexer_names, indexer_spec))
        ],
    )
    connector = (
        SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={"backend": "memcache", "use_layerwise": True, "layerwise_num_shared_buffers": 2},
        )
        if reuse
        else None
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=connector,
        model_config=SimpleNamespace(get_num_layers=lambda _: 4),
        parallel_config=None,
    )
    apply_layerwise_kv_cache_plan(config, vllm_config)
    runner = object.__new__(NPUModelRunner)
    runner.vllm_config = vllm_config
    runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    runner.use_sparse = True
    runner.use_compress = False
    runner.sparse_kv_offload_enabled = False
    runner.runner_only_attn_layers = set()
    runner.device = torch.device("cpu")
    specs = dict.fromkeys(main_names, main_spec) | dict.fromkeys(indexer_names, indexer_spec)
    runner._get_layer_kv_cache_specs = lambda _: specs
    runner._get_attention_kv_cache_dims = lambda *_: (8, 4)
    allocations = []

    def allocate(size, alignment):
        tensor = torch.zeros(size, dtype=torch.int8)
        allocations.append(tensor)
        return tensor

    runner._allocate_int8_cache_tensor = allocate
    raw = runner._allocate_kv_cache_tensors(config)

    assert set(raw) == set(specs)
    if reuse:
        for tensor in config.kv_cache_tensors:
            for name in tensor.layers:
                assert raw[name] is raw[tensor.layers[0]]
    else:
        for names in (main_names, indexer_names):
            assert len({raw[name][0].data_ptr() for name in names}) == len(names)
    assert raw[main_names[0]][0].data_ptr() != raw[main_names[1]][0].data_ptr()
    expected_main_slots = 3 if reuse else len(main_names)
    expected_indexer_slots = 2 if reuse else len(indexer_names)
    expected_bytes = num_blocks * (
        expected_main_slots * main_spec.page_size_bytes + expected_indexer_slots * indexer_spec.page_size_bytes
    )
    assert sum(tensor.numel() for tensor in allocations) == expected_bytes
    raw[main_names[1]][0].fill_(7)
    assert torch.all(raw[main_names[3]][0] == (7 if reuse else 0))
    assert torch.all(raw[main_names[0]][0] == 0)
