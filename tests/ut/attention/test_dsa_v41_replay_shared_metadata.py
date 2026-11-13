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


def _prepare(owner, seq_lens, query_start_loc, replay_start, num_tokens=5, kind="swa", num_reqs=None):
    common = SimpleNamespace(query_start_loc=query_start_loc, replay_start=replay_start)
    req = SimpleNamespace(
        seq_lens=seq_lens,
        num_reqs=len(seq_lens) if num_reqs is None else num_reqs,
        num_actual_tokens=num_tokens,
        cache_kind=kind,
    )
    owner._prepare_query_inputs(common, req)
    return req.ori_topk_length


class TestReplayVisibleLengthsReuse:
    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("window_size", [1, 4, 128])
    @pytest.mark.parametrize(
        "seq_lens,query_start_loc,replay_start",
        [
            ([1], [0, 1], [0]),
            ([6, 5], [0, 3, 5], [3, 2]),
            ([8, 4, 7], [0, 0, 2, 5], [8, 1, 4]),
            ([0, 0], [0, 0, 0], [0, 0]),
        ],
        ids=["single-token", "different-floors", "empty-request", "empty-batch"],
    )
    def test_matches_scalar_reference(self, monkeypatch, dtype, window_size, seq_lens, query_start_loc, replay_start):
        expected = []
        for i, (seq_len, floor) in enumerate(zip(seq_lens, replay_start)):
            query_len = query_start_loc[i + 1] - query_start_loc[i]
            for offset in range(query_len):
                position = seq_len - query_len + offset
                expected.append(position - max(0, position - window_size + 1, floor) + 1)
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        batch_cache: dict[str, object] = {}
        results = [
            _prepare(
                _builder(batch_cache, window_size),
                torch.tensor(seq_lens, dtype=dtype),
                torch.tensor(query_start_loc, dtype=dtype),
                torch.tensor(replay_start, dtype=dtype),
                query_start_loc[-1],
            )
            for _ in range(20)
        ]
        assert build.call_count == 1
        assert all(result is results[0] for result in results)
        torch.testing.assert_close(results[0], torch.tensor(expected, dtype=torch.int32).reshape(-1, 1), rtol=0, atol=0)

    def test_same_views_shared_across_swa_groups(self, monkeypatch):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        seq_lens = torch.tensor([6, 5])
        query_start_loc = torch.tensor([0, 3, 5])
        replay_start = torch.tensor([3, 3])
        batch_cache: dict[str, object] = {}
        results = [_prepare(_builder(batch_cache), seq_lens[:], query_start_loc[:], replay_start[:]) for _ in range(20)]
        assert build.call_count == 1
        assert all(result is results[0] for result in results)
        # Query positions are [3, 4, 5] and [3, 4]; replay starts at 3.
        torch.testing.assert_close(results[0], torch.tensor([[1], [2], [3], [1], [2]], dtype=torch.int32))

    @pytest.mark.parametrize("changed", ["floor", "seq_lens", "query_start_loc"])
    def test_different_request_contexts_use_separate_caches(self, monkeypatch, changed):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        seq_lens = torch.tensor([6, 5])
        query_start_loc = torch.tensor([0, 3, 5])
        replay_start = torch.zeros(2, dtype=torch.int64)
        batch_cache: dict[str, object] = {}
        first = _prepare(_builder(batch_cache), seq_lens, query_start_loc, replay_start)
        if changed == "floor":
            # Decoder replay gives the first SWA layer its own batch dictionary
            # because it uses the encoder replay floor instead of the decoder's.
            replay_start = torch.tensor([3, 3])
            expected = [1, 2, 3, 1, 2]
        elif changed == "seq_lens":
            seq_lens = torch.tensor([3, 2])
            expected = [1, 2, 3, 1, 2]
        elif changed == "query_start_loc":
            query_start_loc = torch.tensor([0, 2, 5])
            expected = [4, 4, 3, 4, 4]
        separate_cache: dict[str, object] = {}
        results = [_prepare(_builder(separate_cache), seq_lens, query_start_loc, replay_start) for _ in range(20)]
        second = results[0]
        assert build.call_count == 2
        assert first is not second
        assert all(result is second for result in results)
        torch.testing.assert_close(first, torch.full((5, 1), 4, dtype=torch.int32))
        torch.testing.assert_close(second, torch.tensor(expected, dtype=torch.int32).unsqueeze(1))

    def test_different_windows_not_shared(self, monkeypatch):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        seq_lens = torch.tensor([6, 5])
        query_start_loc = torch.tensor([0, 3, 5])
        replay_start = torch.zeros(2, dtype=torch.int64)
        batch_cache: dict[str, object] = {}
        first = _prepare(_builder(batch_cache, 4), seq_lens, query_start_loc, replay_start)
        second = _prepare(_builder(batch_cache, 2), seq_lens, query_start_loc, replay_start)
        assert _prepare(_builder(batch_cache, 4), seq_lens, query_start_loc, replay_start) is first
        assert _prepare(_builder(batch_cache, 2), seq_lens, query_start_loc, replay_start) is second
        assert build.call_count == 2
        torch.testing.assert_close(first, torch.full((5, 1), 4, dtype=torch.int32))
        torch.testing.assert_close(second, torch.full((5, 1), 2, dtype=torch.int32))

    @pytest.mark.parametrize("changed", ["floor", "seq_lens", "query_start_loc", "num_tokens", "num_reqs"])
    def test_new_batch_recomputes_updated_buffers(self, monkeypatch, changed):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        seq_lens = torch.tensor([6, 5, 9])
        query_start_loc = torch.tensor([0, 3, 5, 7])
        replay_start = torch.zeros(3, dtype=torch.int64)
        first = _prepare(_builder({}), seq_lens, query_start_loc, replay_start, num_reqs=2)
        num_tokens, num_reqs = 5, 2
        if changed == "floor":
            replay_start.fill_(3)
            expected = [1, 2, 3, 1, 2]
        elif changed == "seq_lens":
            seq_lens[:2].copy_(torch.tensor([3, 2]))
            expected = [1, 2, 3, 1, 2]
        elif changed == "query_start_loc":
            query_start_loc[1] = 2
            expected = [4, 4, 3, 4, 4]
        elif changed == "num_tokens":
            query_start_loc[:3].copy_(torch.tensor([0, 2, 4]))
            num_tokens = 4
            expected = [4] * num_tokens
        else:
            num_reqs, num_tokens = 3, 7
            expected = [4] * num_tokens
        second = _prepare(_builder({}), seq_lens, query_start_loc, replay_start, num_tokens, num_reqs=num_reqs)
        assert build.call_count == 2
        assert first is not second
        torch.testing.assert_close(first, torch.full((5, 1), 4, dtype=torch.int32))
        torch.testing.assert_close(second, torch.tensor(expected, dtype=torch.int32).unsqueeze(1))

    def test_padded_strided_views_only_use_active_requests(self, monkeypatch):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        seq_lens = torch.tensor([-1, 6, -1, 5, -1, 999])[1::2]
        query_start_loc = torch.tensor([-1, 0, -1, 3, -1, 5, -1, 999])[1::2]
        replay_start = torch.tensor([-1, 3, -1, 3, -1, 999])[1::2]
        batch_cache: dict[str, object] = {}
        results = [
            _prepare(_builder(batch_cache), seq_lens[:], query_start_loc[:], replay_start[:], num_reqs=2)
            for _ in range(20)
        ]
        assert build.call_count == 1
        assert all(result is results[0] for result in results)
        torch.testing.assert_close(results[0], torch.tensor([[1], [2], [3], [1], [2]], dtype=torch.int32))

    def test_causal_swa_cache_does_not_supply_replay_lengths(self, monkeypatch):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        owner = _builder({})
        owner._a5_causal_swa_indices = torch.empty((5, 4), dtype=torch.int32)
        owner._a5_causal_swa_lengths = torch.empty((5, 1), dtype=torch.int32)
        indices = torch.zeros((5, 4), dtype=torch.int32)
        causal_lengths = torch.full((5, 1), 4, dtype=torch.int32)
        backend = Mock()
        backend.build_window_indices.return_value = (indices, causal_lengths)
        owner._device_backend = backend
        common = SimpleNamespace(query_start_loc=torch.tensor([0, 3, 5]), replay_start=torch.tensor([3, 3]))
        results = []
        for _ in range(2):
            req = SimpleNamespace(
                seq_lens=torch.tensor([6, 5]),
                num_reqs=2,
                num_actual_tokens=5,
                num_input_tokens=5,
                positions=torch.tensor([3, 4, 5, 3, 4]),
                cache_kind="swa",
            )
            owner._prepare_query_inputs(common, req)
            assert req.ori_sparse_indices is indices
            results.append(req.ori_topk_length)
        backend.build_window_indices.assert_called_once()
        assert build.call_count == 1
        assert results[0] is results[1]
        assert results[0] is not causal_lengths
        torch.testing.assert_close(causal_lengths, torch.full((5, 1), 4, dtype=torch.int32))
        torch.testing.assert_close(results[0], torch.tensor([[1], [2], [3], [1], [2]], dtype=torch.int32))

    @pytest.mark.parametrize("kind,replay_start", [("long_kv", torch.tensor([0])), ("swa", None)])
    def test_other_paths_do_not_build_visible_lengths(self, monkeypatch, kind, replay_start):
        build = Mock(wraps=dsa_v41.build_replay_swa_visible_lens)
        monkeypatch.setattr(dsa_v41, "build_replay_swa_visible_lens", build)
        result = _prepare(_builder({}), torch.tensor([1]), torch.tensor([0, 1]), replay_start, 1, kind)
        assert result is None
        build.assert_not_called()
