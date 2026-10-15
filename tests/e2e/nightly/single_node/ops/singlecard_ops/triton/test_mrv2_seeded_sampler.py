# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.sample.sampler import Sampler as MRV1Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS

from vllm_ascend.ops.triton.v2.sample.prepare import prepare_sampling_inputs
from vllm_ascend.sample.sampler import AscendTopKTopPSampler
from vllm_ascend.worker.v2.sample.logprob import compute_topk_scores
from vllm_ascend.worker.v2.sample.sampler import AscendSampler


@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("vocab_size", [32000, 151936])
def test_seeded_stream_matches_mrv1_after_batch_reordering(seed, vocab_size):
    device = torch.device("npu")
    torch.manual_seed(42)
    logits = torch.randn(3, vocab_size, device=device, dtype=torch.float32)
    sampler = AscendSampler.__new__(AscendSampler)
    sampler.generators = {i: torch.Generator(device=device).manual_seed(seed + i) for i in range(3)}
    reference = {i: torch.Generator(device=device).manual_seed(seed + i) for i in range(3)}
    sampler.sampling_states = SimpleNamespace(
        seeds_set=np.ones(3, dtype=bool),
        temperature=SimpleNamespace(np=np.full(3, 0.6), gpu=torch.full((3,), 0.6, device=device)),
    )
    mrv1 = AscendTopKTopPSampler()
    for order in [[0, 1, 2], [2, 0, 1], [1], [0, 2], [1, 0, 2]]:
        mapping = np.array(order)
        batch = logits[order] / 0.6
        top_k = torch.full((len(order),), 20, dtype=torch.int32, device=device)
        top_p = torch.full((len(order),), 0.95, device=device)
        actual, _ = sampler._sample_random(
            batch.clone(), torch.tensor(order, device=device), mapping, None, top_k, top_p, False
        )
        expected, _ = mrv1.forward_native(
            batch.clone(), {row: reference[req] for row, req in enumerate(order)}, top_k, top_p
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(sampler.generators[req].get_offset() == reference[req].get_offset() for req in order)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_unseeded_greedy_matches_mrv1(dtype, batch_size):
    device = torch.device("npu")
    logits = torch.randn(batch_size, 151936, device=device, dtype=dtype)
    # Equal maxima must retain MRV1's first-index tie behavior.
    logits[:, 3] = 100
    logits[:, 7] = 100
    sampler = AscendSampler.__new__(AscendSampler)
    sampler.sampling_states = SimpleNamespace(
        seeds_set=np.zeros(batch_size, dtype=bool),
        temperature=SimpleNamespace(np=np.zeros(batch_size)),
    )
    mapping = np.arange(batch_size)
    actual, processed_logits = sampler._sample_random(logits, None, mapping, None, None, None, False)
    torch.testing.assert_close(actual, logits.argmax(dim=-1), rtol=0, atol=0)
    assert processed_logits is logits


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("all_decode", [False, True])
def test_plain_greedy_preserves_chunked_prefill_counts(dtype, batch_size, strided, all_decode):
    device = torch.device("npu")
    logits = torch.rand(batch_size, 151936 * 2, dtype=dtype, device=device)[:, ::2]
    if not strided:
        logits = logits.contiguous()
    logits[:, 3] = 100
    logits[:, 7] = 100
    mapping = np.arange(batch_size - 1, -1, -1).copy()
    prefill = torch.arange(8, 8 + batch_size, device=device, dtype=torch.int32)
    indices = torch.tensor(mapping, device=device, dtype=torch.int32)
    seq_lens = prefill[indices.long()] + torch.arange(batch_size, device=device) % 3 + (1 if all_decode else -1)
    sampler = AscendSampler.__new__(AscendSampler)
    sampler.generators = {}
    sampler._decode_num_sampled = torch.ones(batch_size, dtype=torch.int32, device=device)
    sampler._decode_num_rejected = torch.zeros(batch_size, dtype=torch.int32, device=device)
    sampler._decode_counts_num_reqs = -1
    sampler._decode_counts = None
    sampler.compute_nans = False
    sampler.trace_replay_state = None
    sampler.return_sampling_mask = False
    sampler.needs_logits_processing = np.zeros(batch_size, dtype=bool)
    sampler.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=prefill))
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.zeros(batch_size)),
        max_num_logprobs=lambda indices: NO_LOGPROBS,
    )
    sampler.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=lambda indices: 0)
    batch = SimpleNamespace(
        num_reqs=batch_size,
        num_draft_tokens=0,
        has_prefill=not all_decode,
        idx_mapping_np=mapping,
        idx_mapping=indices,
        seq_lens=seq_lens.to(torch.int32),
        cu_num_logits=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
    )
    result = sampler(logits, batch)
    expected_tokens = torch.full((batch_size, 1), 3, dtype=torch.int64, device=device)
    torch.testing.assert_close(result.sampled_token_ids, expected_tokens, rtol=0, atol=0)
    expected_counts = (batch.seq_lens >= prefill[indices.long()]).to(torch.int32)
    torch.testing.assert_close(result.num_sampled, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(result.num_rejected, torch.zeros_like(expected_counts), rtol=0, atol=0)
    assert result.logprobs_tensors is None and result.num_nans is None


@pytest.mark.parametrize("batch_size", [1, 4, 16, 129])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("filtered", ["none", "top_k", "top_p", "both"])
def test_sampling_preparation_preserves_gathers_and_prefill_counts(batch_size, strided, filtered):
    device = torch.device("npu")

    def tensor(values, dtype):
        result = torch.as_tensor(values, dtype=dtype, device=device)
        if strided:
            storage = torch.empty(result.numel() * 2, dtype=dtype, device=device)
            storage[::2] = result
            result = storage[::2]
        return result

    state_count = batch_size + 3
    mapping_np = np.arange(batch_size - 1, -1, -1).copy()
    sample_mapping_np = (mapping_np + 2) % state_count
    indices_np = np.arange(batch_size) * 2 + 1
    mapping = tensor(mapping_np, torch.int32)
    sample_mapping = tensor(sample_mapping_np, torch.int32)
    logits_indices = tensor(indices_np, torch.int64)
    positions = tensor(np.arange(batch_size * 2 + 1) + 1000, torch.int64)
    input_ids = tensor(np.arange(batch_size * 2 + 1) + 100, torch.int32)
    prefill_np = np.arange(state_count) + 10
    prefill = tensor(prefill_np, torch.int32)
    lengths = tensor(prefill_np[mapping_np] + np.arange(batch_size) % 3 - 1, torch.int32)
    logits_per_request = np.arange(batch_size) % 2 + 1
    cumulative = tensor(np.concatenate(([0], np.cumsum(logits_per_request))), torch.int32)
    top_k = tensor(np.arange(state_count) + 20, torch.int32) if filtered in ("top_k", "both") else None
    top_p = tensor(np.linspace(0.6, 1.0, state_count), torch.float32) if filtered in ("top_p", "both") else None
    actual = prepare_sampling_inputs(
        positions,
        input_ids,
        logits_indices,
        mapping,
        sample_mapping,
        lengths,
        cumulative,
        prefill,
        top_k,
        top_p,
    )
    torch.testing.assert_close(actual.positions, positions[logits_indices], rtol=0, atol=0)
    torch.testing.assert_close(actual.input_ids, input_ids[logits_indices], rtol=0, atol=0)
    for actual_value, source in [(actual.top_k, top_k), (actual.top_p, top_p)]:
        if source is None:
            assert actual_value is None
        else:
            torch.testing.assert_close(actual_value, source[sample_mapping.long()], rtol=0, atol=0)
    done = lengths >= prefill[mapping.long()]
    expected_sampled = done.to(torch.int32)
    expected_rejected = torch.where(done, cumulative[1:] - cumulative[:-1] - 1, 0)
    torch.testing.assert_close(actual.num_sampled, expected_sampled, rtol=0, atol=0)
    torch.testing.assert_close(actual.num_rejected, expected_rejected, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("vocab_size", [6, 151936])
@pytest.mark.parametrize("num_logprobs", [0, 5])
@pytest.mark.parametrize("logits_mode", [False, True])
def test_topk_scores_match_mrv1_normalization_and_ranks(batch_size, vocab_size, num_logprobs, logits_mode):
    device = torch.device("npu")
    logits = torch.full((batch_size, vocab_size), -120.0, device=device, dtype=torch.float32)
    # The final two logits become equal only after FP32 log_softmax.
    values = torch.tensor([100.0, 90.0, 80.0, 70.0, 0.000001, 0.000002], device=device)
    logits[:, :6] = values
    sampled = torch.full((batch_size,), 5, dtype=torch.int64, device=device)
    expected_scores = logits if logits_mode else logits.log_softmax(-1, dtype=torch.float32)
    topk = expected_scores.topk(num_logprobs, dim=-1)
    expected_ids = torch.cat((sampled[:, None], topk.indices), dim=1)
    selected = expected_scores.gather(-1, sampled[:, None])
    expected_values = torch.cat((selected, topk.values), dim=1)
    expected_ranks = (expected_scores >= selected).sum(-1)
    actual = compute_topk_scores(logits, num_logprobs, sampled, logits_mode=logits_mode)
    torch.testing.assert_close(actual.logprob_token_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual.logprobs, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(actual.selected_token_ranks, expected_ranks, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
def test_cached_seeded_decode_matches_mrv1_across_batch_changes(dtype, filtered, logprobs_mode):
    device = torch.device("npu")
    count, vocab = 3, 151936
    sampler = AscendSampler.__new__(AscendSampler)
    sampler._decode_params_key = None
    sampler._decode_params = None
    sampler.generators = {i: torch.Generator(device=device).manual_seed(17 + i) for i in range(count)}
    reference = {i: torch.Generator(device=device).manual_seed(17 + i) for i in range(count)}
    sampler.req_states = SimpleNamespace(device=device)
    temperatures = np.array([0.6, 0.9, 1.0], dtype=np.float32)
    top_k = np.array([20, 40, vocab] if filtered else [vocab] * count, dtype=np.int32)
    top_p = np.array([0.95, 0.8, 1.0] if filtered else [1.0] * count, dtype=np.float32)
    sampler.sampling_states = SimpleNamespace(
        seeds_set=np.ones(count, dtype=bool),
        vocab_size=vocab,
        temperature=SimpleNamespace(np=temperatures),
        top_k=SimpleNamespace(np=top_k),
        top_p=SimpleNamespace(np=top_p),
        min_p=SimpleNamespace(np=np.zeros(count)),
    )
    sampler.logit_bias_state = SimpleNamespace(use_logit_bias=np.zeros(count, dtype=bool))
    sampler.penalties_state = SimpleNamespace(use_penalty=np.zeros(count, dtype=bool))
    sampler.bad_words_state = SimpleNamespace(num_bad_words=SimpleNamespace(np=np.zeros(count, dtype=np.int32)))
    sampler.thinking_budget_state = SimpleNamespace(enabled=False)
    sampler.logprob_token_ids_state = None
    sampler.logprobs_mode = logprobs_mode
    sampler.get_logprobs_dims = lambda indices: (5, 0)
    sampler._decode_num_sampled = torch.ones(count, dtype=torch.int32, device=device)
    sampler._decode_num_rejected = torch.zeros(count, dtype=torch.int32, device=device)
    sampler._decode_counts_num_reqs = -1
    sampler._decode_counts = None
    mrv1 = AscendTopKTopPSampler(logprobs_mode=logprobs_mode)
    logits = torch.randn(count, vocab, dtype=dtype, device=device)
    for order in [[0, 1, 2], [0, 1, 2], [2, 0], [1], [2, 0, 1], [2, 0, 1]]:
        indices = np.array(order)
        params = sampler._get_seeded_decode_params(indices)
        assert params is not None
        batch = SimpleNamespace(num_reqs=len(order), expanded_idx_mapping=torch.tensor(order, device=device))
        raw = logits[order]
        raw_before = raw.clone()
        actual = sampler._sample_seeded_decode(raw, batch, params)
        scaled = raw.float() / torch.tensor(temperatures[order], device=device)[:, None]
        k = torch.tensor(top_k[order], device=device) if filtered else None
        p = torch.tensor(top_p[order], device=device) if filtered else None
        expected_tokens, processed = mrv1.forward_native(
            scaled, {row: reference[req] for row, req in enumerate(order)}, k, p
        )
        expected_scores = (
            processed if logprobs_mode == "processed_logprobs" else raw.log_softmax(-1, dtype=torch.float32)
        )
        expected = MRV1Sampler.gather_logprobs(expected_scores, 5, expected_tokens)
        torch.testing.assert_close(raw, raw_before, rtol=0, atol=0)
        torch.testing.assert_close(actual.sampled_token_ids.flatten(), expected_tokens, rtol=0, atol=0)
        for field in ["logprob_token_ids", "logprobs", "selected_token_ranks"]:
            torch.testing.assert_close(
                getattr(actual.logprobs_tensors, field), getattr(expected, field), rtol=0, atol=0, check_dtype=False
            )
        assert all(sampler.generators[req].get_offset() == reference[req].get_offset() for req in order)
        torch.testing.assert_close(actual.num_sampled, torch.ones_like(actual.num_sampled), rtol=0, atol=0)
        torch.testing.assert_close(actual.num_rejected, torch.zeros_like(actual.num_rejected), rtol=0, atol=0)
