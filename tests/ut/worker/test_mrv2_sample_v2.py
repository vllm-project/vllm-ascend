#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.worker.gpu.sample import logprob as upstream_logprob
from vllm.v1.worker.gpu.sample import prompt_logprob as upstream_prompt_logprob
from vllm.v1.worker.gpu.sample import sampler as upstream_sampler
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as upstream_rejection_sampler

import vllm_ascend.patch.worker.patch_sampler  # noqa: F401
import vllm_ascend.patch.worker.patch_triton  # noqa: F401
from vllm_ascend.worker.v2.sample.logprob import compute_topk_logprobs, compute_topk_scores
from vllm_ascend.worker.v2.sample.sampler import (
    _get_pure_decode_counts,
    ascend_sample_random,
    ascend_sampler_call,
)


def _make_fake_sampler(
    temperatures: list[float],
    *,
    needs_processing: list[bool] | None = None,
    num_logprobs: list[int] | None = None,
    compute_nans: bool = False,
    trace_replay: bool = False,
    return_sampling_mask: bool = False,
    use_fp64_gumbel: bool = False,
    logprobs_mode: str = "raw_logprobs",
    subclass: bool = False,
):
    max_reqs = max(len(temperatures), 4)
    temps_np: np.ndarray = np.zeros(max_reqs, dtype=np.float32)
    temps_np[: len(temperatures)] = temperatures
    top_k_np: np.ndarray = np.full(max_reqs, 16, dtype=np.int32)
    top_p_np: np.ndarray = np.ones(max_reqs, dtype=np.float32)
    lp_np: np.ndarray = np.full(max_reqs, -1, dtype=np.int32)
    if num_logprobs is not None:
        lp_np[: len(num_logprobs)] = num_logprobs

    sampling_states = SimpleNamespace(
        max_num_reqs=max_reqs,
        vocab_size=16,
        temperature=SimpleNamespace(np=temps_np, gpu=torch.from_numpy(temps_np)),
        top_k=SimpleNamespace(np=top_k_np, gpu=torch.from_numpy(top_k_np)),
        top_p=SimpleNamespace(np=top_p_np, gpu=torch.from_numpy(top_p_np)),
        seeds=SimpleNamespace(gpu=torch.zeros(max_reqs, dtype=torch.int64)),
        num_logprobs=lp_np,
        max_num_logprobs=lambda idx: int(np.max(lp_np[idx])),
        any_greedy=lambda idx: bool(np.any(temps_np[idx] == 0.0)),
        any_explicit_seed=lambda _idx: False,
        get_top_k_top_p=lambda _eidx, _idx_np, _v: (None, None, False),
    )
    cls = type("CustomSampler", (upstream_sampler.Sampler,), {}) if subclass else upstream_sampler.Sampler
    sampler = object.__new__(cls)
    sampler.logprobs_mode = logprobs_mode
    sampler.compute_nans = compute_nans
    sampler.use_fp64_gumbel = use_fp64_gumbel
    sampler.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.ones(max_reqs, dtype=torch.int32)))
    sampler.sampling_states = sampling_states
    sampler.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=lambda _idx: 0)
    sampler.trace_replay_state = MagicMock() if trace_replay else None
    proc: np.ndarray = np.zeros(max_reqs, dtype=bool)
    if needs_processing is not None:
        proc[: len(needs_processing)] = needs_processing
    sampler.needs_logits_processing = proc
    sampler.return_sampling_mask = return_sampling_mask
    sampler.use_flashinfer = False
    sampler.use_xpu_sampler = False
    sampler.logits_processors = []
    return sampler


def _make_fake_batch(num_reqs: int, *, has_prefill: bool = False, num_draft_tokens: int = 0):
    idx_np: np.ndarray = np.arange(num_reqs, dtype=np.int32)
    idx_gpu = torch.from_numpy(idx_np)
    return SimpleNamespace(
        num_reqs=num_reqs,
        has_prefill=has_prefill,
        num_draft_tokens=num_draft_tokens,
        idx_mapping_np=idx_np,
        idx_mapping=idx_gpu,
        expanded_idx_mapping=idx_gpu,
        cu_num_logits_np=np.arange(num_reqs + 1, dtype=np.int32),
        cu_num_logits=torch.arange(num_reqs + 1, dtype=torch.int32),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32),
        positions=torch.arange(num_reqs, dtype=torch.int32),
        input_ids=torch.arange(num_reqs, dtype=torch.int32),
        logits_indices=torch.arange(num_reqs, dtype=torch.int32),
        seq_lens=torch.full((num_reqs,), 4, dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 4, dtype=torch.int32),
    )


def test_patch_triton_rebinds_compute_topk_scores():
    assert upstream_logprob.compute_topk_scores is compute_topk_scores
    assert upstream_prompt_logprob.compute_topk_scores is compute_topk_scores
    assert upstream_sampler.compute_topk_scores is compute_topk_scores
    assert upstream_rejection_sampler.compute_topk_scores is compute_topk_scores


def test_patch_sampler_installs_ascend_hooks():
    assert upstream_sampler.Sampler.__call__ is ascend_sampler_call
    assert upstream_sampler.Sampler._sample_random is ascend_sample_random


def test_compute_topk_scores_empty_batch_and_cu_num_logits_forms():
    logits = torch.empty((0, 16), dtype=torch.float32)
    sampled = torch.empty((0,), dtype=torch.int64)

    res_list = compute_topk_scores(logits, 2, sampled, cu_num_logits=[0], logits_mode=True)
    assert res_list.logprob_token_ids.shape == (0, 3)
    assert res_list.logprobs.shape == (0, 3)
    assert res_list.selected_token_ranks.shape == (0,)
    assert res_list.cu_num_generated_tokens == [0]
    assert res_list.cu_num_generated_tokens_tensor is None

    cu_tensor = torch.tensor([0], dtype=torch.int32)
    res_tensor = compute_topk_logprobs(logits, 0, sampled, cu_num_logits=cu_tensor)
    assert res_tensor.logprob_token_ids.shape == (0, 1)
    assert res_tensor.cu_num_generated_tokens is None
    assert res_tensor.cu_num_generated_tokens_tensor is cu_tensor


def test_plain_greedy_decode_fast_path_skips_gathers_and_kernels():
    sampler = _make_fake_sampler([0.0, 0.0, 0.0])
    batch = _make_fake_batch(3, has_prefill=False, num_draft_tokens=0)
    # Replace positions/input_ids with sentinels that would fail if indexed.
    batch.positions = MagicMock(side_effect=AssertionError("positions should not be indexed"))
    batch.positions.__getitem__ = MagicMock(side_effect=AssertionError("positions should not be indexed"))
    batch.input_ids = MagicMock(side_effect=AssertionError("input_ids should not be indexed"))
    batch.input_ids.__getitem__ = MagicMock(side_effect=AssertionError("input_ids should not be indexed"))

    logits = torch.tensor(
        [
            [0.1, 2.5, -1.0, 0.4],
            [-3.0, -2.0, -1.0, -5.0],
            [4.0, 4.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    with patch("vllm_ascend.worker.v2.sample.sampler.get_num_sampled_and_rejected") as mock_counts:
        out = sampler(logits, batch)

    mock_counts.assert_not_called()
    assert torch.equal(out.sampled_token_ids, torch.tensor([[1], [2], [0]], dtype=torch.int64))
    assert out.logprobs_tensors is None
    assert out.num_nans is None
    assert torch.equal(out.num_sampled, torch.ones(3, dtype=torch.int32))
    assert torch.equal(out.num_rejected, torch.zeros(3, dtype=torch.int32))


def test_greedy_prefill_still_computes_chunked_prefill_counts():
    sampler = _make_fake_sampler([0.0, 0.0])
    batch = _make_fake_batch(2, has_prefill=True, num_draft_tokens=0)
    logits = torch.tensor([[1.0, 3.0], [4.0, 2.0]], dtype=torch.float32)
    expected_sampled = torch.tensor([0, 1], dtype=torch.int32)
    expected_rejected = torch.tensor([0, 0], dtype=torch.int32)

    with patch(
        "vllm_ascend.worker.v2.sample.sampler.get_num_sampled_and_rejected",
        return_value=(expected_sampled, expected_rejected),
    ) as mock_counts:
        out = sampler(logits, batch)

    mock_counts.assert_called_once()
    assert out.num_sampled is expected_sampled
    assert out.num_rejected is expected_rejected
    assert torch.equal(out.sampled_token_ids, torch.tensor([[1], [0]], dtype=torch.int64))


def test_sampler_subclass_bypasses_plain_greedy_shortcut():
    sampler = _make_fake_sampler([0.0, 0.0], subclass=True)
    batch = _make_fake_batch(2, has_prefill=False, num_draft_tokens=0)
    logits = torch.tensor([[0.2, 1.2], [3.0, 1.0]], dtype=torch.float32)
    custom_tokens = torch.tensor([0, 1], dtype=torch.int64)
    sampler.sample = MagicMock(return_value=(custom_tokens, logits))

    out = sampler(logits, batch)
    sampler.sample.assert_called_once()
    assert torch.equal(out.sampled_token_ids, custom_tokens.view(-1, 1))


def test_sample_random_all_greedy_uses_argmax_and_mixed_delegates():
    greedy_sampler = _make_fake_sampler([0.0, 0.0])
    logits = torch.tensor([[0.1, 0.9, 0.2], [3.0, 1.0, 2.0]], dtype=torch.float32)
    idx_np = np.array([0, 1], dtype=np.int32)
    idx_gpu = torch.from_numpy(idx_np)
    pos = torch.zeros(2, dtype=torch.int32)

    with patch("vllm_ascend.worker.v2.sample.sampler._ORIG_SAMPLE_RANDOM") as mock_orig:
        sampled, out_logits = greedy_sampler._sample_random(logits, idx_gpu, idx_np, pos, None, None, False, False)
    mock_orig.assert_not_called()
    assert torch.equal(sampled, torch.tensor([1, 0], dtype=torch.int64))
    assert out_logits is logits

    mixed_sampler = _make_fake_sampler([0.0, 0.7])
    fallback_ret = (torch.tensor([1, 2], dtype=torch.int64), logits)
    with patch("vllm_ascend.worker.v2.sample.sampler._ORIG_SAMPLE_RANDOM", return_value=fallback_ret) as mock_orig:
        sampled_mixed, _ = mixed_sampler._sample_random(logits, idx_gpu, idx_np, pos, None, None, False, False)
    mock_orig.assert_called_once()
    assert torch.equal(sampled_mixed, fallback_ret[0])


def test_pure_decode_counts_reuses_and_resizes_buffer():
    sampler = _make_fake_sampler([0.0, 0.0])
    ref = torch.zeros(2, dtype=torch.int32)
    ones_a, zeros_a = _get_pure_decode_counts(sampler, 2, ref)
    ones_b, zeros_b = _get_pure_decode_counts(sampler, 3, ref)
    assert ones_a.data_ptr() == ones_b.data_ptr()
    assert zeros_a.data_ptr() == zeros_b.data_ptr()

    ones_big, zeros_big = _get_pure_decode_counts(sampler, 8, ref)
    assert ones_big.shape == (8,)
    assert zeros_big.shape == (8,)
    assert torch.all(ones_big == 1)
    assert torch.all(zeros_big == 0)
