# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.worker.v2.spec_decode.dspark import speculator as module


@pytest.fixture
def replay_spec(monkeypatch):
    spec = module.AscendDSparkSpeculator.__new__(module.AscendDSparkSpeculator)
    spec.attn_architecture = "GQA"
    spec.use_dcp = False
    spec.pcp_manager = None
    spec.num_query_per_req = 7
    spec.max_num_tokens = 64
    spec._group_causal = True
    spec._draft_metadata_num_reqs = 2
    spec.arange_np = np.arange(16, dtype=np.int32)
    spec.input_batch = SimpleNamespace(num_reqs=2, is_prefilling_np=np.array([False, False]))
    spec.input_buffers = SimpleNamespace(seq_lens=torch.tensor([128, 256]), positions=torch.arange(64))
    spec.vllm_config = SimpleNamespace(parallel_config=object())
    spec._update_draft_attn_metadata = Mock()
    calls = []

    def parent(self, **kwargs):
        metadata = {"draft": object(), "lengths": self.input_buffers.seq_lens.clone()}
        calls.append((kwargs, metadata))
        return metadata

    monkeypatch.setattr(DSparkSpeculator, "_build_attn_metadata", parent)
    monkeypatch.setattr(module, "build_attn_metadata_wrapper", lambda: nullcontext())
    monkeypatch.setattr(module, "build_attn_metadata_factory", lambda *a, **k: nullcontext())
    monkeypatch.setattr(module, "vllm_version_is", lambda version: False)
    monkeypatch.setattr(module.AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    return spec, calls


def _build(spec, upper, **changes):
    kwargs = dict(
        num_reqs=2,
        batch_desc=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=28, num_reqs=4),
        query_start_loc_np=np.array([0, 7, 14], dtype=np.int32),
        seq_lens_cpu_upper_bound=upper,
        step=7,
        causal=True,
    )
    kwargs.update(changes)
    return spec._build_attn_metadata(**kwargs)


def test_same_proposal_reuses_metadata_once_for_replay(replay_spec):
    spec, calls = replay_spec
    upper = torch.tensor([128, 256])
    with spec._full_replay_metadata_scope(True):
        metadata = _build(spec, upper)
        assert spec.build_draft_attn_metadatas(4, upper)[0] is metadata
        assert len(calls) == 1
        spec._update_draft_attn_metadata.assert_called_once_with(metadata, 4)
        # The callback is a single consumer; a later call rebuilds normally.
        assert spec.build_draft_attn_metadatas(4, upper)[0] is not metadata
        assert len(calls) == 2
        assert spec._full_replay_metadata_cache is None
    assert spec._full_replay_metadata_cache is None
    assert not spec._reuse_full_replay_metadata


@pytest.mark.parametrize("enabled", [False, True])
def test_metadata_never_survives_a_proposal_or_rejection_update(replay_spec, enabled):
    spec, calls = replay_spec
    upper = torch.tensor([128, 256])
    previous = None
    for iteration in range(3):
        spec.input_buffers.seq_lens.add_(1)
        with spec._full_replay_metadata_scope(enabled):
            metadata = _build(spec, upper)
            assert metadata is not previous
            torch.testing.assert_close(metadata["lengths"], torch.tensor([129 + iteration, 257 + iteration]))
            consumed = spec.build_draft_attn_metadatas(4, upper)[0]
            assert (consumed is metadata) == enabled
        previous = metadata
    assert len(calls) == (3 if enabled else 6)


@pytest.mark.parametrize(
    "changed", ["batch", "padding", "requests", "upper", "length_buffer", "positions", "causal", "width"]
)
def test_changed_replay_inputs_rebuild_metadata(replay_spec, changed):
    spec, calls = replay_spec
    upper = torch.tensor([128, 256])
    padded = 4
    with spec._full_replay_metadata_scope(True):
        metadata = _build(spec, upper)
        if changed == "batch":
            spec.input_batch = SimpleNamespace(num_reqs=2)
        elif changed == "padding":
            padded = 3
        elif changed == "requests":
            spec._draft_metadata_num_reqs = 1
        elif changed == "upper":
            upper = upper.clone()
        elif changed == "length_buffer":
            spec.input_buffers.seq_lens = spec.input_buffers.seq_lens.clone()
        elif changed == "positions":
            spec.input_buffers.positions = spec.input_buffers.positions.clone()
        elif changed == "causal":
            spec._group_causal = False
        elif changed == "width":
            spec.num_query_per_req = 5
        assert spec.build_draft_attn_metadatas(padded, upper)[0] is not metadata
        assert len(calls) == 2
        assert spec._full_replay_metadata_cache is None


@pytest.mark.parametrize("invalid", ["eager", "tokens", "step", "causal", "query", "requests", "dcp"])
def test_nonuniform_or_incompatible_build_is_not_retained(replay_spec, invalid):
    spec, calls = replay_spec
    upper = torch.tensor([128, 256])
    changes = {}
    if invalid == "eager":
        changes["batch_desc"] = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=14, num_reqs=2)
    elif invalid == "tokens":
        changes["batch_desc"] = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=27, num_reqs=4)
    elif invalid == "step":
        changes["step"] = 1
    elif invalid == "causal":
        changes["causal"] = False
    elif invalid == "query":
        changes["query_start_loc_np"] = np.array([0, 6, 14], dtype=np.int32)
    elif invalid == "requests":
        changes["num_reqs"] = 1
    elif invalid == "dcp":
        changes["dcp_local_seq_lens"] = torch.tensor([128, 256])
    with spec._full_replay_metadata_scope(True):
        metadata = _build(spec, upper, **changes)
        assert spec._full_replay_metadata_cache is None
        assert spec.build_draft_attn_metadatas(4, upper)[0] is not metadata
        assert len(calls) == 2


def test_exception_discards_unconsumed_metadata(replay_spec):
    spec, _ = replay_spec
    with pytest.raises(RuntimeError, match="replay failed"), spec._full_replay_metadata_scope(True):
        _build(spec, torch.tensor([128, 256]))
        assert spec._full_replay_metadata_cache is not None
        raise RuntimeError("replay failed")
    assert spec._full_replay_metadata_cache is None
    assert not spec._reuse_full_replay_metadata


@pytest.mark.parametrize("fail", [False, True])
def test_propose_build_and_replay_share_one_metadata_then_retire(replay_spec, monkeypatch, fail):
    spec, calls = replay_spec
    upper = torch.tensor([128, 256])

    def parent(self, *args, **kwargs):
        metadata = _build(self, upper)
        assert self.build_draft_attn_metadatas(4, upper)[0] is metadata
        if fail:
            raise RuntimeError("parent failed")
        return torch.zeros(2, 7, dtype=torch.int64)

    monkeypatch.setattr(DSparkSpeculator, "propose", parent)
    args = (spec.input_batch, {}, {}, torch.empty(0), None, *[torch.empty(0) for _ in range(6)])
    if fail:
        with pytest.raises(RuntimeError, match="parent failed"):
            spec.propose(*args)
    else:
        assert spec.propose(*args).shape == (2, 7)
    assert len(calls) == 1
    assert spec._full_replay_metadata_cache is None
    assert not spec._reuse_full_replay_metadata
    # Capture/callbacks outside propose must always perform a fresh build.
    spec.build_draft_attn_metadatas(4, upper)
    assert len(calls) == 2


@pytest.mark.parametrize("fallback", ["none", "dummy", "profile", "pcp", "dcp", "mla", "other"])
def test_propose_scopes_reuse_to_real_gqa_without_context_parallelism(replay_spec, monkeypatch, fallback):
    spec, calls = replay_spec
    spec.use_dcp = fallback == "dcp"
    spec.pcp_manager = object() if fallback == "pcp" else None
    spec.attn_architecture = "MLA" if fallback == "mla" else None if fallback == "other" else "GQA"
    # Metadata delegation outside GQA is covered by existing architecture tests;
    # here the parent witnesses the production propose scope and its lifetime.
    seen = []

    def parent(self, *args, **kwargs):
        seen.append(self._reuse_full_replay_metadata)
        return torch.zeros(2, 7, dtype=torch.int64)

    monkeypatch.setattr(DSparkSpeculator, "propose", parent)
    batch = spec.input_batch
    result = spec.propose(
        batch,
        {},
        {},
        torch.empty(0),
        None,
        *[torch.empty(0) for _ in range(6)],
        dummy_run=fallback == "dummy",
        is_profile=fallback == "profile",
    )
    assert result.shape == (2, 7)
    assert seen == [fallback == "none"]
    assert not spec._reuse_full_replay_metadata
    assert spec._full_replay_metadata_cache is None
    assert spec._proposal_host_lengths is None
    assert not calls
