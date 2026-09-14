# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for the real Mooncake cache-address helpers."""

import ast
import hashlib
import logging
import math
import random
import struct
from collections import OrderedDict
from collections.abc import Iterator
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
        Iterator=Iterator,
        logger=MagicMock(),
        REGISTER_MERGE_GAP_BYTES=4096,
    )
    load_functions(
        "vllm_ascend/distributed/kv_transfer/utils/utils.py",
        {
            "RegisterRange",
            "RegisterRegions",
            "iter_kv_cache_tensors",
            "tensor_storage_key",
            "split_kv_cache_head_slots",
            "collect_storage_merged_register_regions",
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
    regions = helpers.collect_storage_merged_register_regions({"draft": source_planes})
    last_byte = source_planes[1][-1, -1, -1, -1].data_ptr() + source.element_size()
    assert len(regions.ptrs) == 1
    assert regions.ptrs[0] + regions.lengths[0] == last_byte


@pytest.mark.parametrize("bad", [torch.empty(3, 3, 16, 64), torch.empty(3, 4, 64, 16).transpose(2, 3)])
def test_head_slot_rejects_incompatible_inner_layout(helpers, bad):
    with pytest.raises(ValueError, match="head-slot"):
        helpers.split_kv_cache_head_slots(bad, 2)


def test_single_mla_view_is_not_a_whole_shared_page(helpers):
    backing = torch.empty(4, 3, 16, 576)
    cache = backing[:, 0]
    worker = SimpleNamespace(num_blocks=4)
    assert helpers._get_shared_page_metadata(worker, (cache,)) is None
    region = helpers.collect_storage_merged_register_regions({"mla": cache})
    assert region.lengths == [((4 - 1) * cache.stride(0) + cache[0].numel()) * cache.element_size()]
    assert region.logical_total_bytes == cache.nbytes


def test_packed_multi_component_mla_keeps_existing_whole_page_contract(helpers):
    backing = torch.empty(4 * 64, dtype=torch.float16)
    key = backing.as_strided((4, 32), (64, 1), 0)
    scale = backing.as_strided((4, 8), (64, 1), 32)
    result = helpers._get_shared_page_metadata(SimpleNamespace(num_blocks=4), (key, scale))
    assert result == (backing.data_ptr(), 128, (32,), 1)


def test_mooncake_v1_registration_splits_gqa_and_deduplicates_shared_backing(helpers):
    class FullSpec:
        num_kv_heads = 2

    class MLASpec(FullSpec):
        pass

    class StopAfterRegistration(Exception):
        pass

    scope = dict(vars(helpers))
    engine = MagicMock()
    engine.register_buffer.side_effect = StopAfterRegistration
    scope.update(
        ascend_envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        FullAttentionSpec=FullSpec,
        MLAAttentionSpec=MLASpec,
        enable_sfa_dcp_replicated_indexer=lambda config: False,
        global_te=engine,
        validate_register_region_count=lambda regions: None,
    )
    load_functions(
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py",
        {"register_kv_caches", "_group_skip_kv_reformat"},
        scope,
        "MooncakeConnectorWorker",
    )
    backing = torch.empty(4, 6, 16, 64)
    canonical = {"draft": backing[:, :4]}
    group = {"layer_names": ["draft"], "kv_cache_spec_type": "FullAttentionSpec"}
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=True, hf_text_config=SimpleNamespace())
        ),
        kv_cache_config=SimpleNamespace(num_blocks=4),
        _is_hma_required=True,
        _build_kv_group2layeridx=lambda: {0: (group, [0])},
        _requires_group_aware_attention_transfer=lambda: False,
        _has_mamba_group=lambda: True,
        _get_layer_spec=lambda name: FullSpec(),
        _as_kv_cache_tuple=lambda value: value if isinstance(value, tuple) else [value],
        _get_registered_kv_tensor_buffers=lambda caches: ([1000, 1000], [8192, 8192]),
    )
    with pytest.raises(StopAfterRegistration):
        scope["register_kv_caches"](worker, canonical)
    engine.register_buffer.assert_called_once_with([1000], [8192])
    assert isinstance(canonical["draft"], torch.Tensor)
    assert len(worker.kv_caches["draft"]) == 2
    assert worker.block_len_per_addr == [[2 * 16 * 64 * 4] * 2]
    assert worker.block_stride_per_addr == [[backing.stride(0) * 4] * 2]
    assert scope["_group_skip_kv_reformat"](group)


@pytest.mark.parametrize(
    "overrides, allowed",
    [
        ({}, True),
        ({"remote_ptp_size": 4}, False),
        ({"remote_dcp_size": 1}, True),
        ({"remote_dcp_size": 4}, False),
        ({"remote_pcp_size": 2}, False),
        ({"remote_block_size": 32}, False),
    ],
)
def test_flash_pd_checks_actual_peer_topology_before_loading(overrides, allowed):
    class ReachedTransfer(Exception):
        pass

    scope = {
        "MooncakeConnectorMetadata": SimpleNamespace,
        "logging": logging,
        "ascend_envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        "logger": logging.getLogger(__name__),
    }
    load_functions(
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py",
        {"start_load_kv"},
        scope,
        "MooncakeConnectorWorker",
    )
    values = dict(
        remote_ptp_size=8,
        remote_dcp_size=2,
        remote_pcp_size=1,
        remote_block_size=16,
        do_virtual=False,
        remote_request_id="p-request",
    )
    values.update(overrides)
    worker = SimpleNamespace(
        tp_size=8,
        dcp_size=2,
        block_size=16,
        _prefill_tp_size=8,
        _get_sfa_replicate_k_block_ids=MagicMock(side_effect=ReachedTransfer),
    )
    metadata = SimpleNamespace(reqs_in_batch=[], requests={"d-request": SimpleNamespace(**values)})
    with pytest.raises(ReachedTransfer if allowed else ValueError):
        scope["start_load_kv"](worker, metadata)
    assert worker._get_sfa_replicate_k_block_ids.call_count == int(allowed)


@pytest.fixture
def decode_dcp_worker():
    """Load the actual CPU metadata/routing path without the NPU runtime."""
    path = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if getattr(node, "name", None) == "MooncakeConnectorWorker")
    names = {
        "_expand_block_ids",
        "_get_kv_cache_group_id",
        "_get_kernel_block_scale",
        "_get_kernel_block_ids",
        "_get_decode_only_dcp_metadata",
        "_get_kv_split_metadata",
        "_get_group_pulls_metadata",
        "_get_hybrid_remote_rank_group_pulls",
        "_get_attention_group_num_need_pulls",
        "_get_attention_group_num_need_pulls_for_decode_tp",
        "_get_attention_group_num_key_value_heads",
        "_get_attention_group_remote_rank",
        "_group_use_mla_rank_routing",
        "_is_m3_index_cache_group",
        "_get_remote_ranks_for_req",
    }
    worker.body = [node for node in worker.body if getattr(node, "name", None) in names]
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *[
            node
            for node in tree.body
            if getattr(node, "name", None)
            in {"GroupPull", "string_to_int64_hash", "transfer_groups_need_independent_block_ids"}
        ],
        worker,
    ]
    scope = dict(
        __name__=__name__,
        dataclass=dataclass,
        hashlib=hashlib,
        random=random,
        struct=struct,
        OrderedDict=OrderedDict,
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    result = scope["MooncakeConnectorWorker"]()
    result.vllm_config = SimpleNamespace(model_config=SimpleNamespace(is_deepseek_mla=True))
    result.tp_size = result._decode_tp_size = result._prefill_tp_size = result.dcp_size = 8
    result._prefill_pp_size = 1
    result._is_hma_required = True
    result._get_selected_pcp_rank = lambda *args: 0
    result.block_size = 768
    result.block_size_scale = [[6], [48], [1]]
    # Target MLA and GQA draft share manager group 0 but have different
    # kernel-page counts. Mamba is manager group 1 / transfer group 2.
    result.kv_group2layeridx = {
        0: (
            {"kv_cache_spec_type": "MLAAttentionSpec", "kv_cache_group_id": 0, "kv_cache_spec": {"num_kv_heads": 1}},
            [0],
        ),
        1: (
            {
                "kv_cache_spec_type": "AscendDCPReplicatedDraftAttentionSpec",
                "kv_cache_group_id": 0,
                "kv_cache_spec": {"total_num_kv_heads": 16},
            },
            [1],
        ),
        2: ({"kv_cache_spec_type": "MambaSpec", "kv_cache_group_id": 1}, [2]),
    }
    return result


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("prefix_blocks", [0, 9, 17])
def test_decode_only_dcp_transfers_full_gqa_lanes_and_sharded_mla(decode_dcp_worker, rank, prefix_blocks):
    worker = decode_dcp_worker
    worker.tp_rank = worker.dcp_rank = rank
    local_manager_ids = [3, 1, 4]
    remote_manager_ids = list(reversed(range(1, 19)))
    meta = SimpleNamespace(
        remote_ptp_size=8,
        remote_dcp_size=1,
        remote_pcp_size=1,
        remote_port=30000,
        remote_block_size=768,
        num_prompt_blocks=18,
        num_computed_tokens=prefix_blocks * 768,
        local_block_ids=(local_manager_ids, [5]),
        local_full_block_ids=(local_manager_ids, [99]),
        remote_block_ids=(remote_manager_ids, [7]),
    )
    ports, local_ids, remote_ids = worker._get_kv_split_metadata("k3", meta)
    assert ports == [[30000 + rank]]
    pulls = worker._get_group_pulls_metadata("k3", ports, 8, 30000, 1, 1)
    assert {pull.group_id for pull in pulls[0][0]} == {0, 1, 2}
    assert all(pull.num_group_pulls == 1 and pull.is_group_transfer_end for pull in pulls[0][0])
    # KDA belongs to the same TP owner regardless of the target DCP rank,
    # including ranks with no remaining attention page to receive.
    assert local_ids[0][2] == [5]
    assert remote_ids[0][2] == [7]

    kernel_pages = 6
    scope = {"torch": torch}
    load_functions(
        "vllm_ascend/attention/context_parallel/common_cp.py",
        {"expand_dcp_replicated_block_table"},
        scope,
    )
    target_table = torch.tensor([[block * kernel_pages + page for block in local_manager_ids for page in range(6)]])
    draft_table = scope["expand_dcp_replicated_block_table"](
        target_table, 768, 128, 8, torch.arange(target_table.shape[1] * 8)
    )[0]

    # Emulate only payload-sized copies into cache views with independent
    # first-axis strides, then read through the runner's real block table.
    src = torch.full((19 * 6, 128), -1, dtype=torch.int64)
    for global_block, physical_block in enumerate(remote_manager_ids):
        src[physical_block * 6 : (physical_block + 1) * 6] = torch.arange(
            global_block * 768, (global_block + 1) * 768
        ).view(6, 128)
    for group_id, table in ((0, target_table[0]), (1, draft_table)):
        backing = torch.full((5 * 48, 2, 128), -1, dtype=torch.int64)
        cache = backing[:, 0]
        for dst_page, src_page in zip(local_ids[0][group_id], remote_ids[0][group_id]):
            cache[dst_page].copy_(src[src_page])
        assert torch.all(backing[:, 1] == -1)
        for global_block in range(prefix_blocks, 18):
            if group_id == 0 and global_block % 8 != rank:
                continue
            sequence_block = global_block // 8 if group_id == 0 else global_block
            pages = table[sequence_block * 6 : (sequence_block + 1) * 6]
            assert torch.equal(cache[pages].flatten(), torch.arange(global_block * 768, (global_block + 1) * 768))
