# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.attention.dsa_v41_indexer_shard import (
    IndexerRowPartition,
    gather_indexer_rows,
    plan_indexer_rows,
    rebase_indexer_lengths,
)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_eager_partition_caps_padding_and_accepts_mixed_request_order(tp_size):
    offsets = [0, 4096, 4102, 8192]
    kwargs = dict(topk=512, ratio=2, full_graph=False, query_offsets=offsets, seq_lens=[4096, 8192, 7001])
    partitions = [plan_indexer_rows(8192, tp_size, rank, **kwargs) for rank in range(tp_size)]
    assert all(partition is not None for partition in partitions)
    assert [row for part in partitions for row in range(part.start, part.stop)] == list(range(8192))
    assert max(partitions[0].sizes) <= 8192 // tp_size * 5 // 4


def test_eager_replay_tail_uses_visible_history_to_choose_path():
    for seq_len, expected in [(128, False), (8192, True)]:
        part = plan_indexer_rows(
            128, 8, 0, topk=512, ratio=1, full_graph=False, query_offsets=[0, 128], seq_lens=[seq_len]
        )
        assert (part is not None) == expected


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("layout", ["mixed", "compact_replay", "full_decode"])
def test_rebased_causal_bound_matches_original_for_every_rank(ratio, layout):
    offsets, lengths = {
        "mixed": ([0, 9, 15, 20], [31, 101, 9]),
        "compact_replay": ([0, 128, 133, 261], [8191, 25, 16684]),
        # Last two request segments are graph padding with no keys.
        "full_decode": ([0, 6, 12, 18, 24], [73, 200, 0, 0]),
    }[layout]
    for rank in range(8):
        part = plan_indexer_rows(offsets[-1], 8, rank, topk=512, ratio=ratio, full_graph=True)
        local_offsets, local_keys, residual = rebase_indexer_lengths(
            torch.tensor(offsets, dtype=torch.int32),
            torch.tensor(lengths, dtype=torch.int32),
            part,
            offsets[-1],
            ratio,
        )
        assert local_offsets[-1] == part.stop - part.start
        assert (residual is None) == (ratio == 1)
        for req, (start, stop, length) in enumerate(zip(offsets, offsets[1:], lengths)):
            kept_rows = range(max(part.start, start), min(part.stop, stop))
            rebased_length = int(local_keys[req]) * ratio
            if residual is not None:
                rebased_length += int(residual[req])
            if not kept_rows or length == 0:
                assert rebased_length == 0
                continue
            for local_row, original_row in enumerate(kept_rows, 1):
                assert (rebased_length - len(kept_rows) + local_row) // ratio == (
                    length - (stop - start) + (original_row - start) + 1
                ) // ratio


def test_full_decode_partition_depends_on_bucket_and_not_runtime_lengths():
    kwargs = dict(topk=512, ratio=2, full_graph=True, query_offsets=[0, 6, 12, 18, 24])
    capture = plan_indexer_rows(24, 8, 7, seq_lens=[6, 6, 6, 6], **kwargs)
    replay = plan_indexer_rows(24, 8, 7, seq_lens=[1001, 800, 0, 0], **kwargs)
    assert capture == replay == IndexerRowPartition((3,) * 8, 7)
    assert plan_indexer_rows(6, 8, 0, topk=512, ratio=2, full_graph=True) is None


def test_gather_restores_row_order_and_topk_lengths_in_one_collective():
    payload = torch.arange(24, dtype=torch.int32).view(8, 3)
    chunks = payload.split([3, 2, 3])

    class Group:
        calls = 0

        def all_gather(self, local, dim):
            self.calls += 1
            assert dim == 0
            assert torch.equal(local, F.pad(chunks[1], (0, 0, 0, 1)))
            return torch.cat([chunks[0], local, chunks[2]])

    group = Group()
    assert torch.equal(gather_indexer_rows(chunks[1], IndexerRowPartition((3, 2, 3), 1), group), payload)
    assert group.calls == 1
