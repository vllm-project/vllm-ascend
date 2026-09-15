# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for the real Mooncake cache-address helpers."""

import ast
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_functions(relative_path, names, scope, class_name=None):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        body = next(node for node in body if getattr(node, "name", None) == class_name).body
    tree.body = [node for node in body if getattr(node, "name", None) in names]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(compile(tree, relative_path, "exec"), scope)


@pytest.fixture
def helpers():
    scope = dict(
        torch=torch,
        math=math,
        dataclass=dataclass,
        OrderedDict=OrderedDict,
        Any=Any,
        logger=MagicMock(),
        KV_CACHE_BUFFER_ALIGNMENT=2 * 1024 * 1024,
        get_kv_cache_tensor_layers=lambda config: config.layers,
    )
    load_functions(
        "vllm_ascend/distributed/kv_transfer/utils/utils.py",
        {
            "RegisterRegions",
            "tensor_storage_key",
            "split_kv_cache_head_slots",
        },
        scope,
    )
    load_functions(
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/utils.py",
        {
            "as_kv_cache_tensors",
            "_get_storage_nbytes",
            "_get_tensor_span_nbytes",
            "collect_configured_register_regions",
        },
        scope,
    )
    load_functions(
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py",
        {"_get_shared_page_metadata"},
        scope,
        "MooncakeBaseConnectorWorker",
    )
    return SimpleNamespace(**scope)


def test_head_slot_aliases_and_transfer_do_not_overwrite_neighbor_layers(helpers):
    # Each physical page contains target MLA, draft GQA and Mamba payloads.
    pages, heads, tokens, dim = 5, 2, 16, 64
    payload = 2 * heads * tokens * dim
    page_stride = payload + 3072
    source = torch.arange(pages * page_stride, dtype=torch.float32)
    dest = torch.full_like(source, -1)
    source_cache = source.as_strided((pages, 2 * heads, tokens, dim), (page_stride, tokens * dim, dim, 1), 512)
    dest_cache = dest.as_strided(source_cache.shape, source_cache.stride(), 512)
    source_planes = helpers.split_kv_cache_head_slots(source_cache, heads)
    dest_planes = helpers.split_kv_cache_head_slots(dest_cache, heads)
    for source_plane, dest_plane in zip(source_planes, dest_planes):
        assert source_plane.stride(0) == page_stride
        assert source_plane.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        # Emulate Mooncake's payload-sized, stride-addressed copy for reordered blocks.
        for local_block, remote_block in [(1, 3), (2, 4)]:
            dest_plane[local_block].copy_(source_plane[remote_block])
    assert torch.equal(dest_cache[1], source_cache[3])
    assert torch.equal(dest_cache[2], source_cache[4])
    changed = dest != -1
    expected = torch.zeros_like(changed)
    for page in (1, 2):
        expected[page * page_stride + 512 : page * page_stride + 512 + payload] = True
    assert torch.equal(changed, expected)
    config = SimpleNamespace(kv_cache_tensors=[SimpleNamespace(layers=["draft"], size=source.nbytes)])
    regions = helpers.collect_configured_register_regions(config, {"draft": source_planes})
    last_byte = source_planes[1][-1, -1, -1, -1].data_ptr() + source.element_size()
    assert len(regions.ptrs) == 1
    assert regions.ptrs[0] + regions.lengths[0] >= last_byte


@pytest.mark.parametrize("bad", [torch.empty(3, 3, 16, 64), torch.empty(3, 4, 64, 16).transpose(2, 3)])
def test_head_slot_rejects_incompatible_inner_layout(helpers, bad):
    with pytest.raises(ValueError, match="head-slot"):
        helpers.split_kv_cache_head_slots(bad, 2)


def test_single_mla_view_is_not_a_whole_shared_page(helpers):
    backing = torch.empty(4, 3, 16, 576)
    cache = backing[:, 0]
    worker = SimpleNamespace(num_blocks=4)
    assert helpers._get_shared_page_metadata(worker, (cache,)) is None
    config = SimpleNamespace(kv_cache_tensors=[SimpleNamespace(layers=["mla"], size=backing.nbytes)])
    region = helpers.collect_configured_register_regions(config, {"mla": cache})
    assert region.ptrs[0] <= cache.data_ptr()
    assert region.ptrs[0] + region.lengths[0] >= cache.data_ptr() + helpers._get_tensor_span_nbytes(cache)


def test_packed_multi_component_mla_keeps_existing_whole_page_contract(helpers):
    backing = torch.empty(4 * 64, dtype=torch.float16)
    key = backing.as_strided((4, 32), (64, 1), 0)
    scale = backing.as_strided((4, 8), (64, 1), 32)
    result = helpers._get_shared_page_metadata(SimpleNamespace(num_blocks=4), (key, scale))
    assert result == (backing.data_ptr(), 128, (32,), 1)


@pytest.mark.parametrize("layout", ["head_slots", "planar", "single_mla", "packed_mla"])
def test_v2_registration_preserves_runner_views_and_payload_boundaries(helpers, layout):
    class FullSpec:
        num_kv_heads = 2
        block_size = 16

    class MLASpec(FullSpec):
        pass

    class IndexerSpec:
        pass

    scope = dict(vars(helpers))
    engine = MagicMock()
    scope.update(
        FullAttentionSpec=FullSpec,
        MLAAttentionSpec=MLASpec,
        SlidingWindowMLASpec=MLASpec,
        AscendSFAIndexerCacheSpec=IndexerSpec,
        global_te=engine,
        validate_register_region_count=lambda regions: None,
        MooncakeTransferMetadata=SimpleNamespace,
    )
    load_functions(
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py",
        {"register_kv_caches"},
        scope,
        "MooncakeBaseConnectorWorker",
    )
    backing = torch.empty(4, 6, 16, 64)
    spec = FullSpec()
    if layout == "head_slots":
        cache = backing[:, :4]
        expected_planes = (backing[:, :2], backing[:, 2:4])
    elif layout == "planar":
        cache = (backing[:, :2], backing[:, 2:4])
        expected_planes = cache
    elif layout == "single_mla":
        spec = MLASpec()
        cache = backing[:, 0]
        expected_planes = (cache,)
    else:
        spec = MLASpec()
        cache = (backing[:, :2], backing[:, 2:4])
        expected_planes = cache
    canonical = {"layer": cache}
    worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=2,
            kv_cache_tensors=[SimpleNamespace(layers=["layer"], size=backing.nbytes)],
        ),
        _build_kv_cache_spec_mappings=lambda: None,
        kv_cache_specs=[spec],
        layer_name_to_group_index={"layer": 0},
        layer_name_to_spec_index={"layer": 0},
        engine_id="d",
        te_rpc_port=9000,
        block_size=16,
        side_channel_host="127.0.0.1",
        handshake_port=5000,
    )
    worker._get_shared_page_metadata = lambda caches: helpers._get_shared_page_metadata(worker, caches)
    scope["register_kv_caches"](worker, canonical)
    assert canonical["layer"] is cache
    assert worker.kv_caches is not canonical
    metadata = worker.transfer_metadata
    if layout == "packed_mla":
        assert metadata.block_lens == [[backing.stride(0) * backing.element_size()]]
    else:
        assert metadata.kv_caches_base_addr == [[plane.data_ptr() for plane in expected_planes]]
        assert metadata.block_lens == [[plane[0].numel() * plane.element_size() for plane in expected_planes]]
        assert metadata.block_strides == [[plane.stride(0) * plane.element_size() for plane in expected_planes]]
        assert metadata.block_size_scales == [[2] * len(expected_planes)]
    engine.register_buffer.assert_called_once()
    ptrs, lengths = engine.register_buffer.call_args.args
    assert len(ptrs) == 1
    for plane in expected_planes:
        assert ptrs[0] <= plane.data_ptr()
        assert ptrs[0] + lengths[0] >= plane.data_ptr() + helpers._get_tensor_span_nbytes(plane)
