# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which speculative drafts a structured-output request may accept.

`StructuredOutputManager.grammar_bitmask` fills row i before it inspects
`req_tokens[i]`, so with the first -1 placeholder at index j rows 0..j carry a
real mask and rows j+1.. carry the all-permissive `_full_mask`. Drafts j..K-1
must therefore be rejected, or the request samples with no grammar constraint
at a position the model really does sample.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs

from vllm_ascend.core.grammar_output import GrammarOutput
from vllm_ascend.patch.worker.patch_v2.patch_mrv2_grammar import shard_grammar_output
from vllm_ascend.worker.v2.structured_outputs import grammar_invalid_drafts


def _input_batch(req_ids, num_drafts, num_admitted=None):
    """`num_admitted` is what adaptive verification kept on device."""
    num_admitted = num_drafts if num_admitted is None else num_admitted
    rows = [n + 1 for n in num_admitted]
    return SimpleNamespace(
        req_ids=list(req_ids),
        num_reqs=len(req_ids),
        num_draft_tokens=sum(num_drafts),
        num_draft_tokens_per_req=np.array(num_drafts, dtype=np.int32),
        cu_num_logits=torch.tensor([0, *np.cumsum(rows)], dtype=torch.int32),
        expanded_local_pos=torch.cat([torch.arange(n, dtype=torch.int32) for n in rows]),
    )


def _invalid_rows(batch, grammar_req_ids, num_acceptable):
    mask = grammar_invalid_drafts(batch, grammar_req_ids, num_acceptable)
    return None if mask is None else mask.nonzero().flatten().tolist()


def test_reject_whole_window_when_nothing_was_backfilled():
    # Request "g" is at rows 0..3 (1 + 3 drafts), so its drafts are rows 1..3.
    batch = _input_batch(["g", "p"], [3, 3])
    assert _invalid_rows(batch, ["g"], [0]) == [1, 2, 3]


def test_keep_the_drafts_the_bitmask_could_see():
    batch = _input_batch(["g", "p"], [3, 3])
    assert _invalid_rows(batch, ["g"], [2]) == [3]


def test_follow_the_device_layout_under_adaptive_verification():
    # Scheduled 3 drafts each, but adaptive verification admitted 2 for "a" and
    # 3 for "b": the real rows are a=0..2, b=3..6, not the scheduled 0..3, 4..7.
    batch = _input_batch(["a", "b"], [3, 3], num_admitted=[2, 3])
    assert _invalid_rows(batch, ["a", "b"], [1, 0]) == [2, 4, 5, 6]


def test_missing_backfill_metadata_rejects_only_structured_drafts():
    batch = _input_batch(["plain", "json"], [2, 3])
    assert _invalid_rows(batch, ["json"], None) == [4, 5, 6]


def test_complete_backfill_needs_no_mask():
    batch = _input_batch(["json", "plain"], [2, 3])
    assert _invalid_rows(batch, ["json"], [2]) is None
    assert _invalid_rows(batch, [], None) is None
    assert _invalid_rows(_input_batch(["json"], [0]), ["json"], None) is None


def test_grammar_limits_stay_with_owned_requests_after_sharding():
    batch = _input_batch(["plain", "a", "b"], [0, 2, 3])
    batch.cu_num_logits_np = np.array([0, 1, 4, 8], dtype=np.int32)
    bitmask = np.arange(14, dtype=np.int32).reshape(7, 2)
    grammar = GrammarOutput(["a", "b"], bitmask, [1, 2])
    sharded = shard_grammar_output(grammar, batch, ["plain", "b"])
    assert sharded.structured_output_request_ids == ["b"]
    assert sharded.num_acceptable_drafts == [2]
    np.testing.assert_array_equal(sharded.grammar_bitmask, bitmask[3:])
    assert _invalid_rows(
        _input_batch(["plain", "b"], [0, 3]),
        sharded.structured_output_request_ids,
        sharded.num_acceptable_drafts,
    ) == [4]


@pytest.mark.parametrize("chunk_limit", [3, 100])
@pytest.mark.parametrize("invalid_rows", [None, [1, 2]])
def test_verification_copy_is_chunked_but_penalties_keep_real_ids(monkeypatch, chunk_limit, invalid_rows):
    original = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int32)
    batch = SimpleNamespace(
        num_reqs=2,
        input_ids=original.clone(),
        logits_indices=torch.arange(5),
        positions=torch.arange(5),
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        cu_num_logits_np=np.array([0, 3, 5], dtype=np.int32),
        cu_num_logits=torch.tensor([0, 3, 5], dtype=torch.int32),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
        expanded_local_pos=torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32),
    )
    penalty_ids, watermark_ids, verifier_ids = [], [], []

    def apply_sampling_params(logits, expanded_idx, idx, idx_np, pos, drafts, local_pos):
        penalty_ids.append(drafts.clone())
        return logits

    def watermark(drafts, expanded_idx, local_pos):
        watermark_ids.append(drafts.clone())
        return {}

    def device_verify(logits, draft_logits, drafts, cu, *args, **kwargs):
        # The device kernel is the boundary of this CPU routing test.
        verifier_ids.append(drafts.clone())
        num_reqs = cu.shape[0] - 1
        return torch.zeros((num_reqs, 3), dtype=torch.int64), torch.ones(num_reqs, dtype=torch.int32)

    sampler = object.__new__(rs.RejectionSampler)
    state = SimpleNamespace(gpu=torch.zeros(2))
    sampler.sampler = SimpleNamespace(
        compute_nans=False,
        logprobs_mode="raw_logits",
        use_fp64_gumbel=False,
        apply_sampling_params=apply_sampling_params,
        sampling_states=SimpleNamespace(
            max_num_logprobs=lambda indices: rs.NO_LOGPROBS,
            temperature=state,
            seeds=state,
        ),
        req_states=SimpleNamespace(prefill_len=state),
    )
    sampler.num_speculative_steps = 2
    sampler.enable_adaptive_verification = False
    sampler.synthetic_conditional_rates = None
    sampler.use_block_verification = False
    sampler._watermarking_kwargs = watermark
    monkeypatch.setattr(rs, "rejection_sample", device_verify)
    monkeypatch.setattr(rs, "get_max_chunk_logits", lambda vocab: chunk_limit)
    monkeypatch.setattr(
        rs,
        "get_num_sampled_and_rejected",
        lambda counts, *args: (counts, torch.zeros_like(counts)),
    )
    invalid = None
    expected = original.clone()
    if invalid_rows is not None:
        invalid = torch.zeros(5, dtype=torch.bool)
        invalid[invalid_rows] = True
        expected[invalid_rows] = -1

    sampler(torch.zeros((5, 16)), batch, None, invalid)
    assert torch.equal(torch.cat(verifier_ids), expected)
    assert torch.equal(torch.cat(penalty_ids), original)
    assert torch.equal(torch.cat(watermark_ids), original)
    assert torch.equal(batch.input_ids, original)
    assert len(verifier_ids) == (2 if chunk_limit == 3 else 1)
