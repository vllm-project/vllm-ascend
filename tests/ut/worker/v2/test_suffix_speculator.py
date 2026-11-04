# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU configuration and lifecycle tests; kernel behavior is tested on NPU."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.suffix import speculator as suffix


def _config(corpus_tokens=128, max_cached_requests=10000, depth=8):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=4,
            suffix_decoding_max_tree_depth=depth,
            suffix_decoding_max_cached_requests=max_cached_requests,
            suffix_decoding_corpus_tokens=corpus_tokens,
            enable_adaptive_verification=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=3),
        model_config=SimpleNamespace(max_model_len=1025),
    )


@pytest.fixture
def make_speculator(monkeypatch):
    # These tests allocate real CPU tensors but never launch Triton kernels.
    monkeypatch.setattr(suffix, "HAS_TRITON", True)
    return lambda config: suffix.SuffixSpeculator(config, torch.device("cpu"), SimpleNamespace())


def test_inexact_address_keys_are_rejected_before_allocation(monkeypatch, make_speculator):
    def unexpected_allocation(*args, **kwargs):
        pytest.fail("invalid corpus allocated device buffers before validation")

    monkeypatch.setattr(suffix.torch, "zeros", unexpected_allocation)
    monkeypatch.setattr(suffix.torch, "full", unexpected_allocation)
    with pytest.raises(ValueError, match=r"2\*\*24 cells"):
        make_speculator(_config(corpus_tokens=(1 << 22) + 4))


def test_exact_address_key_limit_reaches_allocation(monkeypatch, make_speculator):
    class AllocationReached(Exception):
        pass

    def stop_before_large_allocation(*args, **kwargs):
        raise AllocationReached

    monkeypatch.setattr(suffix.torch, "zeros", stop_before_large_allocation)
    with pytest.raises(AllocationReached):
        make_speculator(_config(corpus_tokens=1 << 22))


@pytest.mark.parametrize(
    "config",
    [
        _config(corpus_tokens=0),
        _config(corpus_tokens=1 << 30, max_cached_requests=0),
        _config(corpus_tokens=1 << 30, depth=1),
    ],
)
def test_disabled_corpus_does_not_allocate_emit_scratch(config, make_speculator):
    spec = make_speculator(config)
    assert not spec.has_corpus
    assert spec.corpus.numel() == 1
    assert spec.table.numel() == 16
    assert not hasattr(spec, "emit_keys")
    # No request-state buffers are needed when ingestion is disabled.
    spec.on_requests_finished(["finished"])
    assert spec.num_docs == 0


def test_emit_scratch_is_configuration_sized(make_speculator):
    spec = make_speculator(_config())
    assert spec.emit_p == 2048
    assert spec.emit_c_max == 2 * 3 * 2048
    for buffer in (spec.emit_keys, spec.emit_values, spec.emit_prefixes):
        assert buffer.shape == (spec.emit_c_max,)
    assert spec.emit_keys.dtype == torch.float32
    assert spec.emit_values.dtype == spec.emit_prefixes.dtype == torch.int64
    assert spec.emit_summary_values.shape == (12,)
    assert spec.emit_summary_heads.shape == spec.emit_carries.shape == (12,)


def test_unknown_or_empty_finished_requests_do_not_ingest(make_speculator):
    spec = make_speculator(_config())
    spec.req_states.req_id_to_index = {}
    spec.on_requests_finished([])
    spec.on_requests_finished(["already-removed"])
    assert spec.num_docs == 0
    assert spec.doc_seq.tolist() == [0]
    assert spec.corpus_head.tolist() == [0]
    assert (spec.table == -1).all()


def test_reset_clears_learned_state_but_reuses_scratch(make_speculator):
    spec = make_speculator(_config())
    emit_storage = spec.emit_keys.data_ptr()
    spec.corpus.fill_(7)
    spec.table.fill_(123)
    spec.corpus_head.fill_(8)
    spec.doc_seq.fill_(9)
    spec.num_docs = 9
    spec.accept_stats.fill_(4)
    spec.last_bucket.fill_(2)
    spec.last_valid.fill_(3)
    spec.num_steps = 12
    spec.new_slots.extend([0, 2])
    spec.reset_state()
    assert (spec.corpus == -1).all()
    assert (spec.table == -1).all()
    assert spec.corpus_head.tolist() == spec.doc_seq.tolist() == [0]
    assert spec.num_docs == spec.num_steps == 0
    assert not spec.accept_stats.any()
    assert (spec.last_bucket == -1).all()
    assert not spec.last_valid.any()
    assert spec.new_slots == []
    assert spec.emit_keys.data_ptr() == emit_storage


def test_seed_mask_uses_request_indices_and_excludes_greedy(make_speculator):
    spec = make_speculator(_config())
    indices = np.array([2, 0, 1])
    assert spec.seeded_requests(indices) is None
    seeds_set = np.array([True, False, True])
    temperatures = np.array([0.0, 1.0, 0.8])
    spec.set_sampling_states(
        SimpleNamespace(
            seeds_set=seeds_set,
            temperature=SimpleNamespace(np=temperatures),
            any_explicit_seed=lambda idx: bool(seeds_set[idx].any()),
        )
    )
    assert spec.seeded_requests(indices).tolist() == [True, False, False]
    assert spec.seeded_requests(np.array([0, 1])) is None


def test_slot_bookkeeping_is_only_enabled_for_adaptive(make_speculator):
    config = _config()
    spec = make_speculator(config)
    spec.add_request(2)
    assert spec.new_slots == []
    config.speculative_config.enable_adaptive_verification = True
    adaptive = make_speculator(config)
    adaptive.add_request(2)
    adaptive.add_request(0)
    assert adaptive.new_slots == [2, 0]
