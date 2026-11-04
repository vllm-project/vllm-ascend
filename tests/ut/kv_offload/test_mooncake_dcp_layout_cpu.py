# SPDX-License-Identifier: Apache-2.0
"""CPU regression contracts for per-group DCP layout and PD source ownership."""

import ast
import copy
import logging
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"


@pytest.fixture(
    params=[
        "FullAttentionSpec",
        "MLAAttentionSpec",
        "AscendMLAAttentionSpec",
        "SlidingWindowSpec",
        "AscendSlidingWindowMLASpec",
        "AscendSFAIndexerCacheSpec",
        "UniformTypeKVCacheSpecs",
    ]
)
def worker(request):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MooncakeConnectorWorker")
    names = {
        "_get_kernel_block_ids",
        "_group_compress_ratio",
        "_get_dcp_shard_pulls",
        "_get_kv_transfer_spec_key",
        "_get_spec_num_key_value_heads",
        "_add_group_source_ports",
        "_get_kv_cache_group_id",
        "_get_kernel_block_scale",
        "_get_decode_only_dcp_metadata",
        "_expand_block_ids",
        "_get_kv_split_metadata",
        "_get_local_remote_cp_params",
        "_get_group_kernel_params",
        "_local_kernel_ids_for_shard",
        "_get_group_pulls_metadata",
        "_get_hybrid_remote_rank_group_pulls",
        "_get_remote_host_info_by_port",
    }
    cls.body = [node for node in cls.body if getattr(node, "name", None) in names]
    nodes = [
        node
        for node in tree.body
        if getattr(node, "name", None)
        in {"GroupPull", "transfer_groups_need_independent_block_ids", "is_dcp_replicated_attention_group"}
    ]
    scope: dict[str, Any] = {
        "dataclass": dataclass,
        "defaultdict": defaultdict,
        "copy": copy,
        "math": math,
        "logger": logging.getLogger(__name__),
        "OrderedDict": OrderedDict,
    }
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(ast.Module(body=[*nodes, cls], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        scope,
    )
    w = scope["MooncakeConnectorWorker"]()
    w.block_size = 768
    w.block_size_scale = [[6], [6]]
    w.tp_size = w._prefill_tp_size = 2
    w._prefill_pp_size = 1
    w.side_channel_port = 4000
    w.spec = {
        "kv_cache_spec_type": request.param,
        "dcp_sharded": False,
        "block_size": 768,
        "kv_cache_group_id": 1,
    }
    w.kv_group2layeridx = {0: ({"kv_cache_spec_type": "MLAAttentionSpec"}, [0]), 1: (w.spec, [1])}
    w._get_tp_num_need_pulls = lambda *_: 1
    w._group_use_mla_rank_routing = lambda *_: False
    w._get_attention_group_num_need_pulls = lambda *_: 1
    w._get_attention_group_num_key_value_heads = lambda *_: 16
    w._get_selected_pcp_rank = lambda *_: 0
    w._get_remote_ranks_for_req = lambda *_args, **_kwargs: [[0], [1]]
    w._get_attention_group_remote_rank = lambda *_: [w.tp_rank]
    w.GroupPull = scope["GroupPull"]
    w.independent = scope["transfer_groups_need_independent_block_ids"]
    return w


def metadata(prompt=3073, computed=0, remote_dcp=2):
    return SimpleNamespace(
        local_block_ids=([4, 5, 6], [10, 13, 19, 29, 37][computed // 768 :]),
        local_full_block_ids=([4, 5, 6], [10, 13, 19, 29, 37]),
        remote_block_ids=([1, 2, 3], [20, 23, 27, 31, 35]),
        remote_block_size=768,
        remote_dcp_size=remote_dcp,
        remote_pcp_size=1,
        remote_ptp_size=2,
        remote_port=3000,
        num_computed_tokens=computed,
        num_external_tokens=max(0, prompt - 1 - computed),
    )


@pytest.mark.parametrize("remote_dcp", [1, 2, 4])
@pytest.mark.parametrize("prompt,computed", [(1, 0), (128, 0), (769, 0), (1537, 768), (3073, 1536), (3073, 1664)])
def test_global_token_page_mapping(worker, remote_dcp, prompt, computed):
    meta = metadata(prompt, computed, remote_dcp)
    local, remote = worker._get_kernel_block_ids([1], meta, 1, worker.spec)
    tokens = range(computed // 128 * 128, prompt - 1, 128)
    assert local == [meta.local_full_block_ids[1][t // 768] * 6 + t % 768 // 128 for t in tokens]
    page = 768
    assert remote == [meta.remote_block_ids[1][t // page] * (page // 128) + t % page // 128 for t in tokens]


def test_missing_remote_page_fails_before_transferring(worker):
    meta = metadata()
    meta.remote_block_ids = ([1], [20])
    with pytest.raises(ValueError, match="Insufficient remote"):
        worker._get_kernel_block_ids([1], meta, 1, worker.spec)


def test_prefix_hit_without_full_local_table(worker):
    meta = metadata(computed=1536)
    meta.local_block_ids = (meta.local_block_ids[0], [19, 29])
    meta.local_full_block_ids = (meta.local_full_block_ids[0], [10, 13, 19, 29])
    meta.remote_block_ids = (meta.remote_block_ids[0], [20, 23, 27, 31])
    expected = worker._get_kernel_block_ids([1], meta, 1, worker.spec)
    meta.local_full_block_ids = ()
    assert worker._get_kernel_block_ids([1], meta, 1, worker.spec) == expected


@pytest.mark.parametrize("rank", [0, 1])
def test_matching_head_source_owns_complete_sequence_once(worker, rank):
    worker.tp_rank = rank
    meta = metadata()
    ports = [[3001, 3000], [3000, 3001]]
    pulls = worker._get_dcp_shard_pulls(ports, 2, meta.remote_port, 1, "r")
    sources = [
        ports[s][p]
        for s, shard in enumerate(pulls)
        for p, groups in enumerate(shard)
        for g in groups
        if g.group_id == 1
    ]
    assert sources == [3000 + rank]
    assert sum(g.group_id == 0 for shard in pulls for groups in shard for g in groups) == 2


def test_ports_are_included_before_completion_counting(worker):
    meta = metadata()
    mappings = {4000: [[3000, 3000]], 4001: [[3000, 3000]]}
    worker._add_group_source_ports("r", meta, mappings)
    assert mappings == {4000: [[3000, 3000]], 4001: [[3000, 3000], [3001, 3001]]}
    counts = Counter(p for heads in mappings.values() for shards in heads for p in shards)
    assert counts == {3000: 4, 3001: 2}
    worker._add_group_source_ports("r", meta, mappings)
    assert len(mappings[4001]) == 2


def test_replication_uses_independent_transfer_table_even_for_equal_scales(worker):
    worker.kv_group2layeridx[1][0]["kv_cache_group_id"] = 0
    assert worker.independent(worker.kv_group2layeridx, [[6], [6]])


@pytest.mark.parametrize("rank", [0, 1])
def test_decode_only_dcp_routes_group_global_pages(worker, rank):
    worker.tp_rank = worker.dcp_rank = rank
    worker.dcp_size = 2
    worker._is_hma_required = True
    worker._get_hybrid_remote_rank_group_pulls = lambda *_: ([rank], {})
    meta = metadata(remote_dcp=1)
    meta.num_prompt_blocks = 4
    meta.remote_block_ids = ([1, 2, 3, 4], meta.remote_block_ids[1])
    ports, local, remote = worker._get_decode_only_dcp_metadata("r", meta, 2)
    assert ports == [[3000 + rank]]
    assert local[0][0] == [b * 6 + i for b in [4, 5] for i in range(6)]
    assert remote[0][0] == [meta.remote_block_ids[0][j] * 6 + i for j in range(rank, 4, 2) for i in range(6)]
    assert local[0][1] == [b * 6 + i for b in [10, 13, 19, 29] for i in range(6)]
    assert remote[0][1] == [b * 6 + i for b in [20, 23, 27, 31] for i in range(6)]


@pytest.mark.parametrize("local_dcp,remote_dcp", [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)])
def test_mixed_layout_transfer_plan(worker, local_dcp, remote_dcp):
    worker.tp_size = worker._decode_tp_size = worker._prefill_tp_size = 4
    worker.pcp_size = 1
    worker.dcp_size = local_dcp
    worker.use_mla = True
    worker.use_sparse = False
    worker._is_hma_required = True
    worker.num_key_value_heads = 1
    worker._get_remote_ranks_for_req = lambda *_args, **_kwargs: [[r] for r in range(4)]
    worker.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_port=3000))
    for rank in range(4):
        worker.tp_rank = rank
        worker.dcp_rank = rank % local_dcp
        worker.handshake_port = 4000 + rank
        worker.local_remote_block_port_mapping = {}
        worker.remote_port_send_num = {}
        meta = metadata(remote_dcp=remote_dcp)
        meta.remote_ptp_size = 4
        meta.num_prompt_blocks = 4
        meta.remote_engine_id = "p"
        meta.remote_host = "localhost"
        meta.remote_multi_nodes_meta_mapping = {}
        meta.local_block_ids = (list(range(4, 4 + math.ceil(4 / local_dcp))), [10, 13, 19, 29])
        meta.local_full_block_ids = meta.local_block_ids
        meta.remote_block_ids = (list(range(1, 1 + math.ceil(4 / remote_dcp))), [20, 23, 27, 31])
        ports, local, remote = worker._get_kv_split_metadata("r", meta)
        pulls = worker._get_group_pulls_metadata("r", ports, 4, 3000, 1, remote_dcp)
        sources = [
            (s, p)
            for s, shard in enumerate(pulls)
            for p, groups in enumerate(shard)
            if any(g.group_id == 1 for g in groups)
        ]
        assert len(sources) == 1
        shard, port = sources[0]
        assert ports[shard][port] == 3000 + rank
        assert local[shard][1] == [b * 6 + i for b in [10, 13, 19, 29] for i in range(6)]
        assert remote[shard][1] == [b * 6 + i for b in [20, 23, 27, 31] for i in range(6)]
        assert sorted(b for groups in local for b in groups[0]) == [
            b * 6 + i for b in meta.local_block_ids[0] for i in range(6)
        ]


def test_transfer_key_separates_sharding_and_page_geometry(worker):
    spec = SimpleNamespace(num_kv_heads=8, block_size=128, dcp_sharded=True)
    sharded = worker._get_kv_transfer_spec_key(spec, 16)
    spec.dcp_sharded = False
    replicated = worker._get_kv_transfer_spec_key(spec, 16)
    assert replicated != sharded
    spec.block_size = 256
    assert worker._get_kv_transfer_spec_key(spec, 16) != replicated


def test_complete_sequence_sources_preserve_pp_and_pcp(worker):
    worker.tp_rank = 0
    worker._prefill_pp_size = 2
    worker._get_attention_group_remote_rank = lambda *_: [0, 2]
    ports = [[3002, 3006], [3000, 3004]]
    pulls = worker._get_dcp_shard_pulls(ports, 2, 3000, 2, "r")
    group_pulls = [
        (ports[s][p], g.prefill_pp_rank)
        for s, shard in enumerate(pulls)
        for p, groups in enumerate(shard)
        for g in groups
        if g.group_id == 1
    ]
    assert group_pulls == [(3002, 0), (3006, 1)]


def test_group_source_ports_preserve_pp_and_pcp(worker):
    worker._prefill_pp_size = 2
    worker._get_selected_pcp_rank = lambda *_: 1
    worker._get_remote_ranks_for_req = lambda *_args, **_kwargs: [[0, 2], [1, 3]]
    meta = metadata()
    meta.remote_pcp_size = 2
    mappings = {4000: [[3000, 3001]], 4001: [[3000, 3001]]}
    worker._add_group_source_ports("r", meta, mappings)
    assert mappings[4000] == [[3000, 3001], [3002, 3002], [3006, 3006]]
    assert mappings[4001] == [[3000, 3001], [3003, 3003], [3007, 3007]]


def test_missing_group_source_fails_before_transfer(worker):
    worker.tp_rank = 1
    with pytest.raises(ValueError, match="TP sources are absent"):
        worker._get_dcp_shard_pulls([[3000]], 2, 3000, 1, "r")


def test_unsharded_only_model_with_dcp_topology(worker):
    worker.tp_rank = worker.dcp_rank = 1
    worker.dcp_size = worker.tp_size = worker._decode_tp_size = 2
    worker.pcp_size = 1
    worker.use_mla = True
    worker.use_sparse = False
    worker._is_hma_required = False
    worker.num_key_value_heads = 1
    worker.handshake_port = 4001
    worker.local_remote_block_port_mapping = {}
    worker.remote_port_send_num = {}
    worker.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_port=3000))
    worker.spec["kv_cache_group_id"] = 0
    worker.kv_group2layeridx = {0: (worker.spec, [1])}
    meta = metadata()
    meta.local_block_ids = meta.local_full_block_ids = ([10, 13, 19, 29],)
    meta.remote_block_ids = ([20, 23, 27, 31],)
    meta.num_prompt_blocks = 4
    meta.remote_engine_id = "p"
    meta.remote_host = "localhost"
    meta.remote_multi_nodes_meta_mapping = {}
    ports, local, remote = worker._get_kv_split_metadata("r", meta)
    pulls = worker._get_group_pulls_metadata("r", ports, 2, 3000, 1, 2)
    sources = [(s, p) for s, shard in enumerate(pulls) for p, groups in enumerate(shard) if groups]
    assert len(sources) == 1
    shard, port = sources[0]
    assert ports[shard][port] == 3001
    assert len(local[shard][0]) == len(remote[shard][0]) == 24


@pytest.mark.parametrize("dcp_size", [1, 2, 8])
def test_scheduler_prompt_truncation_uses_each_groups_dcp_size(dcp_size):
    tree = ast.parse(SOURCE.read_text())
    info = next(n for n in tree.body if getattr(n, "name", None) == "GroupTransferInfo")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MooncakeConnectorScheduler")
    cls.body = [n for n in cls.body if getattr(n, "name", None) == "_get_transfer_block_ids"]
    scope: dict[str, Any] = {"dataclass": dataclass, "cdiv": lambda a, b: (a + b - 1) // b}
    exec(
        compile(
            "from __future__ import annotations\n" + ast.unparse(ast.Module(body=[info, cls], type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        scope,
    )
    scheduler = scope["MooncakeConnectorScheduler"]()
    scheduler.dcp_size = dcp_size
    scheduler.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    info_cls = scope["GroupTransferInfo"]
    scheduler.group_transfer_info = [
        info_cls(128, 0, False, True),
        info_cls(128, 0, False, False),
        info_cls(128, 0, True, False),
    ]
    blocks = tuple(list(range(1, 12)) for _ in range(3))
    sharded, replicated, state = scheduler._get_transfer_block_ids(blocks, 513)
    assert sharded == blocks[0][: math.ceil(513 / (128 * dcp_size))]
    assert replicated == [1, 2, 3, 4, 5]
    assert state == [5]
