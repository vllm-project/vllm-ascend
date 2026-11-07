# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-local reuse of SWA bounded-replay visible lengths."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import dsa_v41


def _builder(batch_cache, window_size=4):
    owner = dsa_v41.AscendDSAV41MetadataBuilder.__new__(dsa_v41.AscendDSAV41MetadataBuilder)
    owner._window_size = window_size
    owner._a5_causal_swa_indices = None
    owner.common_v41_batch_metadata = batch_cache
    return owner


def _prepare(owner, seq_lens, query_start_loc, replay_start, num_tokens=5, kind="swa"):
    common = SimpleNamespace(query_start_loc=query_start_loc, replay_start=replay_start)
    req = SimpleNamespace(seq_lens=seq_lens, num_reqs=len(seq_lens), num_actual_tokens=num_tokens, cache_kind=kind)
    owner._prepare_query_inputs(common, req)
    return req.ori_topk_length


class TestReplayVisibleLengthsReuse:
    def test_same_views_shared_across_swa_groups(self, monkeypatch):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        seq_lens = torch.tensor([6, 5])
        query_start_loc = torch.tensor([0, 3, 5])
        replay_start = torch.tensor([3, 3])
        batch_cache = {}
        results = [_prepare(_builder(batch_cache), seq_lens[:], query_start_loc[:], replay_start[:]) for _ in range(20)]
        assert build.call_count == 1
        assert all(result is results[0] for result in results)
        # Query positions are [3, 4, 5] and [3, 4]; replay starts at 3.
        torch.testing.assert_close(results[0], torch.tensor([[1], [2], [3], [1], [2]], dtype=torch.int32))

    @pytest.mark.parametrize("changed", ["floor", "seq_lens", "query_start_loc", "window"])
    def test_different_inputs_not_shared(self, monkeypatch, changed):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        seq_lens = torch.tensor([6, 5])
        query_start_loc = torch.tensor([0, 3, 5])
        replay_start = torch.zeros(2, dtype=torch.int64)
        batch_cache = {}
        first = _prepare(_builder(batch_cache), seq_lens, query_start_loc, replay_start)
        window_size = 4
        expected = [4, 4, 4, 4, 4]
        if changed == "floor":
            replay_start = torch.tensor([3, 3])
            expected = [1, 2, 3, 1, 2]
        elif changed == "seq_lens":
            seq_lens = torch.tensor([3, 2])
            expected = [1, 2, 3, 1, 2]
        elif changed == "query_start_loc":
            query_start_loc = torch.tensor([0, 2, 5])
            expected = [4, 4, 3, 4, 4]
        else:
            window_size = 2
            expected = [2, 2, 2, 2, 2]
        second = _prepare(_builder(batch_cache, window_size), seq_lens, query_start_loc, replay_start)
        assert build.call_count == 2
        assert first is not second
        torch.testing.assert_close(first, torch.full((5, 1), 4, dtype=torch.int32))
        torch.testing.assert_close(second, torch.tensor(expected, dtype=torch.int32).unsqueeze(1))

    def test_new_batch_recomputes_updated_buffers(self, monkeypatch):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        seq_lens = torch.tensor([6, 5])
        query_start_loc = torch.tensor([0, 3, 5])
        replay_start = torch.zeros(2, dtype=torch.int64)
        first = _prepare(_builder({}), seq_lens, query_start_loc, replay_start)
        replay_start.fill_(3)
        second = _prepare(_builder({}), seq_lens, query_start_loc, replay_start)
        assert build.call_count == 2
        torch.testing.assert_close(first, torch.full((5, 1), 4, dtype=torch.int32))
        torch.testing.assert_close(second, torch.tensor([[1], [2], [3], [1], [2]], dtype=torch.int32))

    @pytest.mark.parametrize("kind,replay_start", [("long_kv", torch.tensor([0])), ("swa", None)])
    def test_other_paths_do_not_build_visible_lengths(self, monkeypatch, kind, replay_start):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        result = _prepare(_builder({}), torch.tensor([1]), torch.tensor([0, 1]), replay_start, 1, kind)
        assert result is None
        build.assert_not_called()

    @pytest.mark.parametrize("num_tokens", [0, 1])
    def test_empty_and_single_token(self, num_tokens):
        result = _prepare(
            _builder({}), torch.tensor([num_tokens]), torch.tensor([0, num_tokens]), torch.tensor([0]), num_tokens
        )
        torch.testing.assert_close(result, torch.ones((num_tokens, 1), dtype=torch.int32))
