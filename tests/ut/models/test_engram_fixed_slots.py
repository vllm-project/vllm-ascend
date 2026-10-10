# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import utils
from vllm_ascend.models.deepseek_v41.engram import parallel


@pytest.fixture
def runtime(monkeypatch):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=16),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    ascend = SimpleNamespace(scheduler_config=SimpleNamespace(recompute_scheduler_enable=True))
    context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([7, 7])))
    group = SimpleNamespace(rank_in_group=0, world_size=2)
    monkeypatch.setattr(utils, "get_ascend_config", lambda: ascend)
    monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(parallel, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(parallel, "get_forward_context", lambda: context)
    monkeypatch.setattr(parallel, "get_potential_max_tokens", lambda: 96)
    monkeypatch.setattr(parallel, "get_dp_group", lambda: group)
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: group)
    return config, ascend, context


def test_fixed_slot_ignores_rank_local_metadata(runtime):
    _, _, context = runtime
    # These are the unsynchronized metadata vectors seen by two DP ranks.
    for local_count in (0, 1, 42, 48, 96):
        context.dp_metadata.num_tokens_across_dp_cpu.fill_(local_count)
        assert parallel.engram_gathered_num_tokens() == 96


@pytest.mark.parametrize("mode", ["prefill", "recompute_off", "profile", "uniform_warmup"])
def test_synchronized_paths_keep_metadata_slot(runtime, mode):
    config, ascend, context = runtime
    if mode == "prefill":
        config.kv_transfer_config.is_kv_consumer = False
        config.kv_transfer_config.is_kv_producer = True
    elif mode == "recompute_off":
        ascend.scheduler_config.recompute_scheduler_enable = False
    elif mode == "profile":
        context.in_profile_run = True
    else:
        context.engram_uniform_dp_warmup = True
    context.dp_metadata.num_tokens_across_dp_cpu = torch.tensor([900, 1024])
    assert parallel.engram_gathered_num_tokens() == 1024


@pytest.mark.parametrize(
    "max_batched,max_seqs,spec,potential,expected",
    [
        (2048, 128, 5, 512, 768),
        (640, 128, 5, 512, 640),
        (1024, 16, None, 32, 32),
    ],
)
def test_fixed_slot_covers_eager_decode_and_graph_padding(
    runtime, monkeypatch, max_batched, max_seqs, spec, potential, expected
):
    config, _, _ = runtime
    config.scheduler_config.max_num_batched_tokens = max_batched
    config.scheduler_config.max_num_seqs = max_seqs
    config.speculative_config = None if spec is None else SimpleNamespace(num_speculative_tokens=spec)
    monkeypatch.setattr(parallel, "get_potential_max_tokens", lambda: potential)
    assert parallel.engram_gathered_num_tokens() == expected


def test_overflow_fails_before_collective_and_bypasses_do_not_need_metadata(runtime, monkeypatch):
    hashes = torch.empty((97, 2, 24), dtype=torch.int32)
    with pytest.raises(ValueError, match="exceeds the DP token slot"):
        parallel.gather_engram_hashes(hashes)
    assert parallel.gather_engram_hashes(hashes, dp_shared_memory=True) is hashes
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: None)
    assert parallel.gather_engram_hashes(hashes) is hashes


@pytest.mark.parametrize("counts", [(3, 0, 1, 0), (6, 6, 6, 6)])
def test_dp_pcp_split_table_round_trip_preserves_query_and_head_order(runtime, monkeypatch, counts):
    config, _, _ = runtime
    config.scheduler_config.max_num_batched_tokens = 8
    monkeypatch.setattr(parallel, "get_potential_max_tokens", lambda: 8)
    hashes = [torch.arange(count, dtype=torch.int32).view(-1, 1, 1) + rank * 10 for rank, count in enumerate(counts)]
    padded = torch.cat([torch.cat((ids, ids.new_full((8 - len(ids), 1, 1), parallel.DEAD_ID))) for ids in hashes])

    def lookup(owner):
        ids = padded[:, 0, 0].view(-1, 1, 1)
        heads = torch.arange(owner * 2, owner * 2 + 2).view(1, 2, 1)
        return torch.where(ids == parallel.DEAD_ID, 0, ids * 100 + heads)

    for rank, ids in enumerate(hashes):

        def all_gather(local, dim, rank=rank):
            assert dim == 0
            torch.testing.assert_close(local, padded[rank * 8 : (rank + 1) * 8])
            return padded

        group = SimpleNamespace(world_size=4, rank_in_group=rank, device_group=object(), all_gather=all_gather)
        monkeypatch.setattr(parallel, "get_engram_dp_group", lambda group=group: group)
        gathered = parallel.gather_engram_hashes(ids)
        torch.testing.assert_close(gathered, padded)

        def all_to_all(recv, staged, *, group, rank=rank):
            torch.testing.assert_close(staged, lookup(rank))
            recv.copy_(torch.cat([lookup(owner)[rank * 8 : (rank + 1) * 8] for owner in range(4)]))

        monkeypatch.setattr(parallel.dist, "all_to_all_single", all_to_all)
        output = parallel.exchange_engram_rows(lookup(rank), len(ids))
        expected = ids[:, 0, 0].view(-1, 1, 1) * 100 + torch.arange(8).view(1, 8, 1)
        torch.testing.assert_close(output, expected)
        assert parallel.gather_engram_hashes(ids, dp_shared_memory=True) is ids
