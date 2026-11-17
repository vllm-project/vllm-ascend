# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.dsa_v41 import (
    AscendDSAV41MetadataBuilder,
    build_replay_swa_visible_lens,
)
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.deepseek_v41 import mixed_quant_attention


@pytest.mark.parametrize("lengths", [(6, 1, 0, 0), (1, 6, 0, 0), (1, 1, 0, 0), (0, 0, 0, 0), (2, 0, 3, 0), (9,), ()])
def test_v41_replay_window_uses_logical_token_count(lengths):
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    history = [121, 255, 127, 0]
    seq_lens = torch.tensor([h + length for h, length in zip(history, lengths)], dtype=torch.int32)
    replay_start = torch.tensor([120, 253, 126, 0], dtype=torch.int32)
    actual = build_replay_swa_visible_lens(4, starts, seq_lens, replay_start, num_tokens=sum(lengths))
    expected = []
    for req, length in enumerate(lengths):
        for position in range(history[req], history[req] + length):
            expected.append(position - max(position - 3, replay_start[req].item(), 0) + 1)
    torch.testing.assert_close(actual[:, 0], torch.tensor(expected, dtype=torch.int32))


def _window_reference(positions, window_size, *, visible_lens=None, indices_output=None, lengths_output=None):
    lengths = (positions + 1).clamp(0, window_size).to(torch.int32).unsqueeze(1)
    if visible_lens is not None:
        lengths = torch.minimum(lengths, visible_lens.clamp_min(0))
    columns = torch.arange(window_size).view(1, 1, -1)
    indices = positions.view(-1, 1, 1) + 1 - lengths.unsqueeze(-1) + columns
    indices = torch.where(columns < lengths.unsqueeze(-1), indices, -1).int()
    if indices_output is not None:
        indices_output.copy_(indices)
        lengths_output.copy_(lengths)
        return indices_output, lengths_output
    return indices, lengths


@pytest.mark.parametrize("specialized", [False, True])
def test_v41_a5_replay_window_keeps_current_token_and_masks_padding(monkeypatch, specialized):
    """A shorter sparse list must keep the newest KV, including this query's KV."""
    monkeypatch.setattr(dsa_v41.DeviceOperator, "get_dsv41_packed_cache_ops", lambda: None)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(sliding_window=4, head_dim=8, num_attention_heads=4, index_topk=8)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    spec = AscendSlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=4, model_version="deepseek_v41"
    )
    builder_cls = dsa_v41.AscendDSAV41SWAMetadataBuilder if specialized else AscendDSAV41MetadataBuilder
    builder = builder_cls(spec, [], config, torch.device("cpu"))
    peer = builder_cls(spec, [], config, torch.device("cpu"))
    # Execute the real metadata builder, replacing only A5's index producer.
    builder._a5_causal_swa_indices = torch.empty(16, 1, 4, dtype=torch.int32)
    builder._a5_causal_swa_lengths = torch.empty(16, 1, dtype=torch.int32)
    builder._device_backend = SimpleNamespace(build_window_indices=_window_reference)
    peer._a5_causal_swa_indices = torch.empty(16, 1, 4, dtype=torch.int32)
    peer._a5_causal_swa_lengths = torch.empty(16, 1, dtype=torch.int32)
    peer._device_backend = builder._device_backend
    common = SimpleNamespace(
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 3, 7, 16, 16]),
        seq_lens=torch.zeros(4, dtype=torch.int32),
        seq_lens_cpu=torch.tensor([127, 261, 0, 0]),
        positions=torch.full((16,), 299, dtype=torch.int64),
        slot_mapping=torch.arange(16),
        block_table_tensor=torch.zeros(4, 8, dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=7,
        num_input_tokens=16,
        max_query_len=6,
        max_seq_len=261,
        is_prefilling=torch.tensor([False, True, False, False]),
        replay_start=torch.tensor([121, 255, 0, 0], dtype=torch.int32),
    )
    monkeypatch.setattr(dsa_v41, "get_cos_and_sin_dsa", lambda *args, **kwargs: ({}, {}))
    for lengths in ((6, 1), (1, 6), (1, 1), (2, 3), (6, 1)):
        end = sum(lengths)
        common.num_actual_tokens = end
        common.query_start_loc.copy_(torch.tensor([0, lengths[0], end, end, end]))
        common.seq_lens[:2] = torch.tensor([121 + lengths[0], 255 + lengths[1]])
        common.positions[:end] = torch.cat([torch.arange(121, 121 + lengths[0]), torch.arange(255, 255 + lengths[1])])
        shared: dict[str, Any] = {}
        metadata = builder.build(0, common, num_actual_reqs=2, common_v41_batch_metadata=shared)
        # A peer shares request coordinates but has a different replay floor.
        # Its build must not reuse or overwrite this group's sparse window.
        peer_common = SimpleNamespace(**vars(common))
        peer_common.replay_start = torch.zeros_like(common.replay_start)
        peer_metadata = peer.build(0, peer_common, num_actual_reqs=2, common_v41_batch_metadata=shared)
        expected = torch.full((16, 1, 4), -1, dtype=torch.int32)
        for row, position in enumerate(common.positions[:end].tolist()):
            floor = 121 if row < lengths[0] else 255
            keys = torch.arange(max(position - 3, floor), position + 1, dtype=torch.int32)
            expected[row, 0, : keys.numel()] = keys
        torch.testing.assert_close(metadata.ori_sparse_indices, expected)
        torch.testing.assert_close(metadata.ori_topk_length, (expected >= 0).sum(-1, dtype=torch.int32))
        assert metadata.ori_sparse_indices.data_ptr() == builder._a5_causal_swa_indices.data_ptr()
        assert peer_metadata.ori_sparse_indices.data_ptr() == peer._a5_causal_swa_indices.data_ptr()
        assert peer_metadata.ori_topk_length[:end].tolist() == [[4]] * end
        assert peer_metadata.ori_topk_length[end:].tolist() == [[0]] * (16 - end)
        assert peer_metadata.ori_sparse_indices[0, 0].tolist() == [118, 119, 120, 121]


@pytest.mark.parametrize("prebuilt", [False, True])
def test_v41_a5_window_resolution_preserves_replay_floor(monkeypatch, prebuilt):
    monkeypatch.setattr(mixed_quant_attention, "build_window_indices_triton", _window_reference)
    positions = torch.tensor([255, 256])
    lengths = torch.tensor([[1], [2]], dtype=torch.int32)
    indices = torch.tensor([[[255, -1, -1, -1]], [[255, 256, -1, -1]]], dtype=torch.int32)
    metadata = SimpleNamespace(
        positions=positions,
        swa=SimpleNamespace(ori_sparse_indices=indices if prebuilt else None, ori_topk_length=lengths),
    )
    actual_indices, actual_lengths = mixed_quant_attention._resolve_window_indices(torch.empty(2, 1, 8), metadata, 4)
    torch.testing.assert_close(actual_indices, indices)
    torch.testing.assert_close(actual_lengths, lengths)
