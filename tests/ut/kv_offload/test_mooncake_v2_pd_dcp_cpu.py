# SPDX-License-Identifier: Apache-2.0
"""CPU address/rank regressions for DP4/TP8 P-DCP1 to D-DCP8."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


class FullSpec:
    pass


class ReplicatedSpec(FullSpec):
    pass


class MLASpec(FullSpec):
    pass


class MambaSpec:
    pass


def receiver():
    source = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if getattr(node, "name", None) == "MooncakePullRecvingThread"
    )
    cls.bases = []
    names = {
        "_get_layer_remote_tp_rank_groups",
        "_get_mamba_remote_tp_rank_groups",
        "_infer_total_num_kv_heads",
        "_get_head_interval",
        "_append_spec_transfer_addresses",
        "_append_block_transfer_addresses",
        "_get_attention_remote_tp_rank_groups",
        "_compute_group_block_ids",
        "_expand_block_ids",
        "_select_remote_tp_rank",
    }
    cls.body = [node for node in cls.body if getattr(node, "name", None) in names]
    scope = dict(
        np=np,
        FullAttentionSpec=FullSpec,
        AscendDCPReplicatedDraftAttentionSpec=ReplicatedSpec,
        MLAAttentionSpec=MLASpec,
        MambaSpec=MambaSpec,
        SlidingWindowSpec=type("Sliding", (FullSpec,), {}),
        SlidingWindowMLASpec=type("SlidingMLA", (MLASpec,), {}),
        AscendSFAIndexerCacheSpec=type("Indexer", (), {}),
    )
    helpers = ast.parse((ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/utils.py").read_text())
    function = next(n for n in helpers.body if getattr(n, "name", None) == "group_concurrent_contiguous")
    exec(compile("from __future__ import annotations\n" + ast.unparse(function), str(source), "exec"), scope)
    exec(compile("from __future__ import annotations\n" + ast.unparse(cls), str(source), "exec"), scope)
    thread = scope[cls.name]()
    thread.tp_size, thread.dcp_size = 8, 8
    thread.num_speculative_tokens = 3
    thread.block_shapes = [[(2, 128, 64)]]
    return thread


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize(
    "prompt,computed",
    [(p, c) for p in [1, 127, 128, 767, 768, 769, 6144, 6145, 12289, 200000] for c in [0, 768, 6144] if c < p],
)
def test_replicated_draft_routes_matching_tp_shard_and_token_pages(rank, prompt, computed):
    worker = receiver()
    worker.tp_rank = worker.dcp_rank = rank
    remote = SimpleNamespace(block_shapes=[[(2, 128, 64)]])
    groups = worker._get_layer_remote_tp_rank_groups(0, 0, ReplicatedSpec(), remote, 8, 1)
    assert groups == [[rank]]
    full_local = [100 + 7 * i for i in range((prompt + 6143) // 6144)]
    local = full_local[computed // 6144 :]
    remote_ids = [20 + 3 * i for i in range((prompt + 767) // 768)]
    actual = worker._compute_group_block_ids(
        "request",
        groups,
        1,
        0,
        6144,
        768,
        local,
        full_local,
        remote_ids,
        prompt,
        prompt,
        computed,
        48,
        6,
        ReplicatedSpec(),
        0,
    )
    expected_local, expected_remote = [], []
    for token in range(computed // 128 * 128, prompt - 1, 128):
        expected_local.append(full_local[token // 6144] * 48 + token % 6144 // 128)
        expected_remote.append(remote_ids[token // 768] * 6 + token % 768 // 128)
    assert actual == [(rank, expected_local, expected_remote)]


@pytest.mark.parametrize("rank", range(8))
def test_target_mla_and_mamba_keep_their_own_mapping(rank):
    worker = receiver()
    worker.tp_rank = worker.dcp_rank = rank
    remote = SimpleNamespace(block_shapes=[[(1, 128, 576)]])
    groups = worker._get_layer_remote_tp_rank_groups(0, 0, MLASpec(), remote, 8, 1)
    remote_ids = list(range(20, 36))
    actual = worker._compute_group_block_ids(
        "request",
        groups,
        1,
        0,
        768,
        768,
        [100, 107],
        [100, 107],
        remote_ids,
        12288,
        12288,
        0,
        6,
        6,
        MLASpec(),
        0,
    )
    assert actual == [
        (
            groups[0][0],
            [b * 6 + k for b in [100, 107] for k in range(6)],
            [remote_ids[i] * 6 + k for i in [rank, rank + 8] for k in range(6)],
        )
    ]
    assert worker._get_layer_remote_tp_rank_groups(0, 0, MambaSpec(), remote, 8, 1) == [[rank]]
    assert worker._compute_group_block_ids(
        "request",
        [[rank]],
        1,
        1,
        768,
        768,
        [90, 91, 92, 93],
        [90, 91, 92, 93],
        [7],
        12288,
        12288,
        0,
        1,
        1,
        MambaSpec(),
        0,
    ) == [(rank, [90], [7])]


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("padded", [False, True])
def test_replicated_draft_transfer_byte_addresses(rank, padded):
    worker = receiver()
    worker.tp_rank = worker.dcp_rank = rank
    worker.kv_cache_specs = [ReplicatedSpec()]
    worker.layer_names = ["draft"]
    worker.kv_caches_base_addr = [[100000, 200000]]
    page_bytes = 2 * 128 * 64 * 2
    stride = page_bytes + 128 * padded
    worker.block_strides = [[stride, stride]]
    worker.block_lens = [[page_bytes, page_bytes]]
    worker.block_shapes = [[(2, 128, 64), (2, 128, 64)]]
    remote = SimpleNamespace(
        block_shapes=worker.block_shapes,
        block_strides=worker.block_strides,
        block_lens=worker.block_lens,
        metadata_by_tp_rank={rank: SimpleNamespace(kv_caches_base_addr=[[300000, 400000]])},
    )
    src, dst, lengths = [], [], []
    worker._append_spec_transfer_addresses(
        0, rank, 8, 1, {(0, 0): [("request", [5, 9], [17, 24])]}, remote, src, dst, lengths
    )
    assert src == [base + page * stride for base in [100000, 200000] for page in [5, 9]]
    assert dst == [base + page * stride for base in [300000, 400000] for page in [17, 24]]
    assert lengths == [page_bytes] * 4
