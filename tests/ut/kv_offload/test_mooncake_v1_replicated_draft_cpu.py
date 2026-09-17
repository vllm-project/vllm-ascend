# SPDX-License-Identifier: Apache-2.0
"""Regression contracts for V1 replicated draft pages and source ownership."""

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"


@pytest.fixture
def worker():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MooncakeConnectorWorker")
    names = {
        "_get_replicated_draft_block_ids",
        "_set_replicated_draft_transfer_metadata",
        "_add_replicated_draft_ports",
        "_get_kv_cache_group_id",
        "_get_kernel_block_scale",
        "_get_decode_only_dcp_metadata",
        "_expand_block_ids",
    }
    cls.body = [node for node in cls.body if getattr(node, "name", None) in names]
    nodes = [
        node
        for node in tree.body
        if getattr(node, "name", None)
        in {"GroupPull", "transfer_groups_need_independent_block_ids", "is_replicated_draft_group"}
    ]
    scope: dict[str, Any] = {"dataclass": dataclass}
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
        "kv_cache_spec_type": "FullAttentionSpec",
        "dcp_sharded": False,
        "block_size": 768,
        "kv_cache_group_id": 1,
    }
    w.kv_group2layeridx = {0: ({"kv_cache_spec_type": "MLAAttentionSpec"}, [0]), 1: (w.spec, [1])}
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
    local, remote = worker._get_replicated_draft_block_ids(meta, 1, worker.spec, [1])
    tokens = range(computed // 128 * 128, prompt - 1, 128)
    assert local == [meta.local_full_block_ids[1][t // 768] * 6 + t % 768 // 128 for t in tokens]
    page = 768
    assert remote == [meta.remote_block_ids[1][t // page] * (page // 128) + t % page // 128 for t in tokens]


def test_missing_remote_page_fails_before_transferring(worker):
    meta = metadata()
    meta.remote_block_ids = ([1], [20])
    with pytest.raises(ValueError, match="Insufficient remote"):
        worker._get_replicated_draft_block_ids(meta, 1, worker.spec, [1])


def test_prefix_hit_requires_full_local_table(worker):
    meta = metadata(computed=1536)
    meta.local_full_block_ids = ()
    with pytest.raises(ValueError, match="full local block table"):
        worker._get_replicated_draft_block_ids(meta, 1, worker.spec, [1])


@pytest.mark.parametrize("rank", [0, 1])
def test_matching_head_source_owns_all_draft_pages_once(worker, rank):
    worker.tp_rank = rank
    meta = metadata()
    ports = [[3001, 3000], [3000, 3001]]
    local: list[tuple[list[int], list[int]]] = [([4], []), ([5], [])]
    remote: list[tuple[list[int], list[int]]] = [([1], []), ([2], [])]
    pulls = [[[worker.GroupPull(0, 0, 1), worker.GroupPull(1, 0, 1)] for _ in shard] for shard in ports]
    worker._set_replicated_draft_transfer_metadata("r", meta, ports, local, remote, pulls)
    sources = [
        ports[s][p]
        for s, shard in enumerate(pulls)
        for p, groups in enumerate(shard)
        for g in groups
        if g.group_id == 1
    ]
    assert sources == [3000 + rank]
    assert [blocks[0] for blocks in local] == [[4], [5]]
    assert [blocks[0] for blocks in remote] == [[1], [2]]
    assert sum(g.group_id == 0 for shard in pulls for groups in shard for g in groups) == 4
    assert local[0][1] == [block * 6 + i for block in [10, 13, 19, 29] for i in range(6)]
    assert local[1][1] == []


def test_ports_are_included_before_completion_counting(worker):
    meta = metadata()
    mappings = {4000: [[3000, 3000]], 4001: [[3000, 3000]]}
    worker._add_replicated_draft_ports("r", meta, mappings)
    assert mappings == {4000: [[3000, 3000]], 4001: [[3000, 3000], [3001, 3001]]}
    counts = Counter(p for heads in mappings.values() for shards in heads for p in shards)
    assert counts == {3000: 4, 3001: 2}
    worker._add_replicated_draft_ports("r", meta, mappings)
    assert len(mappings[4001]) == 2


def test_replication_uses_independent_transfer_table_even_for_equal_scales(worker):
    worker.kv_group2layeridx[1][0]["kv_cache_group_id"] = 0
    assert worker.independent(worker.kv_group2layeridx, [[6], [6]])


@pytest.mark.parametrize("rank", [0, 1])
def test_decode_only_dcp_routes_draft_global_pages(worker, rank):
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
