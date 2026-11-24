# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for active layouts, configuration and cache ownership."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import TypeAdapter

from vllm_ascend.ascend_config import AscendConfig, DraftSamplingConfig, SparseKVOffloadConfig
from vllm_ascend.worker.v2.spec_decode.dspark import hccl_sampling
from vllm_ascend.worker.v2.spec_decode.dspark.cache_adapter import compact_rejection_inputs
from vllm_ascend.worker.v2.spec_decode.dspark.hccl_sampling import DSparkHCCLSamplingMixin


def _spec(monkeypatch, tp_size=4, max_num_reqs=4096, steps=3):
    monkeypatch.setattr(hccl_sampling, "get_tp_group", lambda: SimpleNamespace(world_size=tp_size))
    spec = object.__new__(DSparkHCCLSamplingMixin)
    spec.device = torch.device("cpu")
    spec.max_num_reqs = max_num_reqs
    spec.num_speculative_steps = steps
    spec._step_cols = torch.arange(steps, dtype=torch.int32)
    spec.draft_tokens = torch.empty(max_num_reqs, steps, dtype=torch.int64)
    return spec


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("num_reqs", [1, 3, 8, 10, 17])
def test_layout_ownership_padding_and_restore(monkeypatch, tp_size, num_reqs):
    spec = _spec(monkeypatch, tp_size=tp_size)
    capacity, requests, rows, restore, tokens, gathered = spec._build_hccl_layout(num_reqs)

    assert capacity == (num_reqs + tp_size - 1) // tp_size
    for rank in range(tp_size):
        expected = list(range(rank, num_reqs, tp_size))
        assert requests[rank].tolist() == expected + [-1] * (capacity - len(expected))
    assert sorted(requests[requests >= 0].tolist()) == list(range(num_reqs))
    assert requests.flatten()[restore].tolist() == list(range(num_reqs))
    assert tokens.shape == (capacity, 3)
    assert gathered.shape == (tp_size * capacity, 3)
    # Label each source row independently: verify that valid rows select the
    # original request's entire block and padding safely reads request zero.
    labels = torch.arange(num_reqs * 3).view(num_reqs, 3)
    selected = labels.flatten()[rows]
    for rank in range(tp_size):
        for local_idx, req in enumerate(requests[rank].tolist()):
            torch.testing.assert_close(selected[rank, local_idx], labels[max(req, 0)])


def test_layouts_are_not_retained_on_speculator(monkeypatch):
    spec = _spec(monkeypatch, tp_size=8, steps=8)
    for num_reqs in (1, 17, 4096, 10):
        layout = spec._build_hccl_layout(num_reqs)
        rows = layout[2]
        assert rows.numel() <= (num_reqs + 7) * 8
        refs = [weakref.ref(tensor) for tensor in layout[1:]]
        del rows, layout
        gc.collect()
        assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("num_reqs", [0, -1, 4097])
def test_layout_rejects_invalid_batch_size(monkeypatch, num_reqs):
    with pytest.raises(ValueError, match="num_reqs"):
        _spec(monkeypatch)._build_hccl_layout(num_reqs)


def test_zero_requests_does_not_build_a_layout(monkeypatch):
    spec = _spec(monkeypatch)
    spec._hccl_draft_enabled = True
    spec._build_hccl_layout = MagicMock(side_effect=AssertionError("unexpected layout"))
    spec._sample_sequential(0, torch.empty(0, 8))
    spec._build_hccl_layout.assert_not_called()


def test_small_batch_uses_parent_without_building_layout(monkeypatch):
    class Parent:
        def _sample_sequential(self, num_reqs, hidden):
            return "fallback"

    class Spec(DSparkHCCLSamplingMixin, Parent):
        pass

    spec = object.__new__(Spec)
    spec._hccl_draft_enabled = True
    spec._hccl_min_reqs = 8
    monkeypatch.setattr(hccl_sampling, "get_tp_group", lambda: SimpleNamespace(world_size=8, rank_in_group=0))
    spec.num_speculative_steps = 3
    spec._build_hccl_layout = MagicMock(side_effect=AssertionError("unexpected layout"))
    assert spec._sample_sequential(3, torch.empty(9, 8)) == "fallback"


def test_draft_sampling_config_defaults_and_nested_parsing():
    defaults = DraftSamplingConfig()
    assert defaults.batch_sharded is False
    assert defaults.resolve_shard_min_reqs(8, 4096) == 8
    config = TypeAdapter(DraftSamplingConfig).validate_python({"batch_sharded": True, "shard_min_reqs": 10})
    assert config.resolve_shard_min_reqs(8, 4096) == 10
    parent = AscendConfig(
        sparse_kv_offload_config=SparseKVOffloadConfig(),
        draft_sampling_config={"batch_sharded": True, "shard_min_reqs": 10},
    )
    assert isinstance(parent.draft_sampling_config, DraftSamplingConfig)
    assert parent.draft_sampling_config.batch_sharded is True


@pytest.mark.parametrize("values", [{"shard_min_reqs": 0}, {"shard_min_reqs": -1}, {"unexpected": True}])
def test_draft_sampling_config_rejects_bad_values(values):
    with pytest.raises(ValueError):
        TypeAdapter(DraftSamplingConfig).validate_python(values)


def test_draft_sampling_threshold_rejects_excess_capacity():
    with pytest.raises(ValueError, match="max_num_seqs"):
        DraftSamplingConfig(shard_min_reqs=17).resolve_shard_min_reqs(8, 16)


def test_compact_rejection_reorders_metadata_and_rejects_stale_ownership():
    # Slot 5 maps to row 1, but row 1 now belongs to slot 7.
    cache = SimpleNamespace(
        values=torch.arange(12, dtype=torch.float32).view(2, 2, 3),
        row_mapping=torch.tensor([-1, -1, 0, -1, -1, 1, -1, 1], dtype=torch.int32),
        inverse_row_mapping=torch.tensor([2, 7], dtype=torch.int32),
    )
    temperature = torch.arange(8, dtype=torch.float32) + 0.5
    seeds = torch.arange(8, dtype=torch.int64) * 104729 + 17
    q, sampled, indices, expanded, local_temp, local_seeds = compact_rejection_inputs(
        cache,
        torch.tensor([9, 10, 11, 12]),
        torch.tensor([5, 2], dtype=torch.int32),
        torch.tensor([5, 5, 2, 2], dtype=torch.int32),
        torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        temperature,
        seeds,
    )
    torch.testing.assert_close(q[0], torch.zeros_like(q[0]))
    torch.testing.assert_close(q[1], cache.values[0])
    assert sampled.tolist() == [9, -1, 11, 12]
    assert indices.tolist() == [0, 1]
    assert expanded.tolist() == [0, 0, 1, 1]
    torch.testing.assert_close(local_temp, temperature[[5, 2]])
    torch.testing.assert_close(local_seeds, seeds[[5, 2]])
