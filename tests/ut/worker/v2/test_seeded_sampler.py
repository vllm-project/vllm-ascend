# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_ascend.sample.sampler import random_sample
from vllm_ascend.worker.v2.sample.sampler import AscendSampler


def _sampler(seeds, temperatures):
    sampler = AscendSampler.__new__(AscendSampler)
    sampler.req_states = SimpleNamespace(
        device=torch.device("cpu"),
        req_id_to_index={str(i): i for i in range(len(seeds))},
        index_to_req_id={i: str(i) for i in range(len(seeds))},
    )
    sampler.preempted_generators = {}
    sampler.generators = {i: torch.Generator().manual_seed(seed) for i, seed in enumerate(seeds)}
    sampler.sampling_states = SimpleNamespace(
        seeds_set=np.ones(len(seeds), dtype=bool),
        temperature=SimpleNamespace(np=np.array(temperatures), gpu=torch.tensor(temperatures)),
    )
    return sampler


@pytest.mark.parametrize("seed", [0, 17, 1001])
def test_seeded_sampling_preserves_mrv1_stream_across_batch_permutations(seed):
    sampler = _sampler([seed, seed + 1], [0.6, 0.6])
    reference = {i: torch.Generator().manual_seed(s) for i, s in enumerate([seed, seed + 1])}
    logits = torch.tensor([[2.0, 1.9, 1.8, 1.7], [1.0, 1.1, 1.2, 1.3]])
    with (
        patch("vllm_ascend.sample.sampler.npu_stream_switch", return_value=nullcontext()),
        patch("vllm_ascend.sample.sampler.global_stream", return_value=MagicMock()),
        patch("vllm_ascend.sample.sampler.torch.npu.current_stream", return_value=MagicMock()),
        patch.object(torch.Tensor, "record_stream", lambda self, stream: None),
    ):
        for order in [[0, 1], [1, 0], [0], [1], [1, 0]]:
            indices = np.array(order)
            batch = logits[order]
            actual, _ = sampler._sample_random(batch.clone(), torch.tensor(order), indices, None, None, None, False)
            expected = random_sample(batch.softmax(-1), {row: reference[req] for row, req in enumerate(order)})
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("seeded", [False, True])
def test_all_greedy_does_not_advance_seeded_generators(seeded):
    sampler = _sampler([0, 17], [0.0, 0.0])
    sampler.sampling_states.seeds_set[:] = seeded
    before = [g.get_state().clone() for g in sampler.generators.values()]
    tokens, _ = sampler._sample_random(
        torch.tensor([[1.0, 2.0], [3.0, 1.0]]), torch.tensor([0, 1]), np.array([0, 1]), None, None, None, False
    )
    assert tokens.tolist() == [1, 0]
    assert all(torch.equal(state, g.get_state()) for state, g in zip(before, sampler.generators.values()))


def test_reused_request_slot_resets_seed_and_clears_unseeded_state():
    sampler = _sampler([42], [0.6])
    with patch.object(Sampler, "add_request"):
        sampler.add_request(0, 10, SamplingParams(seed=17))
        assert sampler.generators[0].initial_seed() == 17
        sampler.add_request(0, 10, SamplingParams())
        assert 0 not in sampler.generators


def test_discarded_prefill_does_not_advance_request_rng():
    sampler = AscendSampler.__new__(AscendSampler)
    sampler.generators = {0: MagicMock(), 1: MagicMock()}
    sampler.generators[0].get_offset.return_value = 12
    batch = SimpleNamespace(
        num_computed_prefill_tokens_np=np.array([0, 10]),
        num_scheduled_tokens=np.array([4, 1]),
        prefill_len_np=np.array([10, 10]),
        idx_mapping_np=np.array([0, 1]),
    )
    with patch.object(AscendSampler, "_sample_with_preparation", return_value="sampled"):
        assert sampler(None, batch) == "sampled"
    sampler.generators[0].set_offset.assert_called_once_with(12)
    sampler.generators[1].set_offset.assert_not_called()


def test_preempted_request_retains_rng_when_resumed_in_another_slot():
    sampler = _sampler([17, 42], [0.6, 0.6])
    generator = sampler.generators[0]
    torch.rand(8, generator=generator)
    state = generator.get_state().clone()
    sampler.finish_requests(set(), {"0"})
    assert 0 not in sampler.generators
    sampler.req_states.req_id_to_index = {"0": 1}
    sampler.req_states.index_to_req_id = {1: "0"}
    with patch.object(Sampler, "add_request"):
        sampler.add_request(1, 10, SamplingParams(seed=17))
    assert sampler.generators[1] is generator
    assert torch.equal(sampler.generators[1].get_state(), state)
    assert not sampler.preempted_generators


def test_finished_preempted_request_does_not_leak_rng_to_reused_id():
    sampler = _sampler([17], [0.6])
    sampler.finish_requests(set(), {"0"})
    sampler.req_states.req_id_to_index.clear()
    sampler.finish_requests({"0"}, set())
    assert not sampler.preempted_generators
    sampler.req_states.req_id_to_index = {"0": 0}
    with patch.object(Sampler, "add_request"):
        sampler.add_request(0, 10, SamplingParams(seed=42))
    assert sampler.generators[0].initial_seed() == 42


def test_finished_request_wins_over_preemption():
    sampler = _sampler([17], [0.6])
    sampler.finish_requests({"0"}, {"0"})
    assert not sampler.generators
    assert not sampler.preempted_generators


@pytest.mark.parametrize("finished", [False, True])
def test_absent_preemption_list_preserves_live_generators(finished):
    sampler = _sampler([17], [0.6])
    generator = sampler.generators[0]
    sampler.finish_requests({"0"} if finished else set(), None)
    assert not sampler.preempted_generators
    if finished:
        assert not sampler.generators
    else:
        assert sampler.generators[0] is generator


@pytest.mark.parametrize("feature", ["nans", "trace", "mask", "flashinfer", "expanded", "padded", "drafts"])
def test_sampling_preparation_preserves_upstream_feature_paths(feature):
    sampler = AscendSampler.__new__(AscendSampler)
    sampler.compute_nans = feature == "nans"
    sampler.trace_replay_state = object() if feature == "trace" else None
    sampler.return_sampling_mask = feature == "mask"
    sampler.use_flashinfer = feature == "flashinfer"
    batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=int(feature == "drafts"),
        idx_mapping_np=np.array([-1 if feature == "padded" else 0]),
    )
    logits = torch.empty((2 if feature == "expanded" else 1, 4))
    with patch.object(Sampler, "__call__", return_value="upstream") as upstream:
        assert sampler._sample_with_preparation(logits, batch) == "upstream"
    upstream.assert_called_once_with(logits, batch)


def _decode_sampler():
    sampler = _sampler([17, 42], [0.6, 0.9])
    sampler._decode_params_key = None
    sampler._decode_params = None
    states = sampler.sampling_states
    states.vocab_size = 128
    states.temperature.np = states.temperature.np.astype(np.float32)
    states.top_k = SimpleNamespace(np=np.array([20, 40], dtype=np.int32))
    states.top_p = SimpleNamespace(np=np.array([0.95, 0.8], dtype=np.float32))
    states.min_p = SimpleNamespace(np=np.zeros(2, dtype=np.float32))
    sampler.logit_bias_state = SimpleNamespace(use_logit_bias=np.zeros(2, dtype=bool))
    sampler.penalties_state = SimpleNamespace(use_penalty=np.zeros(2, dtype=bool))
    sampler.bad_words_state = SimpleNamespace(num_bad_words=SimpleNamespace(np=np.zeros(2, dtype=np.int32)))
    sampler.thinking_budget_state = SimpleNamespace(enabled=False, use_thinking_budget=np.zeros(2, dtype=bool))
    sampler.get_logprobs_dims = lambda indices: None
    return sampler


def test_decode_parameter_cache_tracks_batch_reordering():
    sampler = _decode_sampler()
    first = sampler._get_seeded_decode_params(np.array([0, 1]))
    assert sampler._get_seeded_decode_params(np.array([0, 1])) is first
    reordered = sampler._get_seeded_decode_params(np.array([1, 0]))
    torch.testing.assert_close(reordered.temperature.flatten(), torch.tensor([0.9, 0.6]))
    assert reordered.top_k.tolist() == [40, 20]
    assert reordered.generators == {0: sampler.generators[1], 1: sampler.generators[0]}
    assert sampler._get_seeded_decode_params(np.array([1, 0])) is reordered


@pytest.mark.parametrize("event", ["finish", "preempt", "replace"])
def test_decode_parameter_cache_invalidates_on_request_lifecycle(event):
    sampler = _decode_sampler()
    sampler._get_seeded_decode_params(np.array([0, 1]))
    if event == "replace":
        with patch.object(Sampler, "add_request"):
            sampler.add_request(0, 10, SamplingParams(seed=1001))
    else:
        sampler.finish_requests({"0"} if event == "finish" else set(), {"0"} if event == "preempt" else set())
    assert sampler._decode_params is None and sampler._decode_params_key is None


@pytest.mark.parametrize("feature", ["unseeded", "greedy", "bias", "penalty", "bad_words", "thinking_budget", "min_p"])
def test_decode_parameter_cache_retains_contextual_sampling_paths(feature):
    sampler = _decode_sampler()
    if feature == "unseeded":
        sampler.sampling_states.seeds_set[0] = False
    elif feature == "greedy":
        sampler.sampling_states.temperature.np[0] = 0
    elif feature == "bias":
        sampler.logit_bias_state.use_logit_bias[0] = True
    elif feature == "penalty":
        sampler.penalties_state.use_penalty[0] = True
    elif feature == "bad_words":
        sampler.bad_words_state.num_bad_words.np[0] = 1
    elif feature == "thinking_budget":
        sampler.thinking_budget_state.enabled = True
        sampler.thinking_budget_state.use_thinking_budget[0] = True
    else:
        sampler.sampling_states.min_p.np[0] = 0.1
    assert sampler._get_seeded_decode_params(np.array([0, 1])) is None


def test_decode_counts_keep_inflight_views_across_batch_size_changes():
    sampler = AscendSampler.__new__(AscendSampler)
    sampler._decode_num_sampled = torch.ones(4, dtype=torch.int32)
    sampler._decode_num_rejected = torch.zeros(4, dtype=torch.int32)
    sampler._decode_counts_num_reqs = -1
    sampler._decode_counts = None
    previous = sampler._get_decode_counts(4)
    assert sampler._get_decode_counts(4) is previous
    for num_reqs in (1, 3, 4):
        sampled, rejected = sampler._get_decode_counts(num_reqs)
        torch.testing.assert_close(sampled, torch.ones(num_reqs, dtype=torch.int32))
        torch.testing.assert_close(rejected, torch.zeros(num_reqs, dtype=torch.int32))
    torch.testing.assert_close(previous[0], torch.ones(4, dtype=torch.int32))
    torch.testing.assert_close(previous[1], torch.zeros(4, dtype=torch.int32))


def test_decode_parameter_cache_owns_snapshot_of_request_values():
    sampler = _decode_sampler()
    params = sampler._get_seeded_decode_params(np.array([0, 1]))
    sampler.sampling_states.temperature.np[:] = 2.0
    sampler.sampling_states.top_k.np[:] = 1
    sampler.sampling_states.top_p.np[:] = 0.1
    torch.testing.assert_close(params.temperature.flatten(), torch.tensor([0.6, 0.9]))
    assert params.top_k.tolist() == [20, 40]
    torch.testing.assert_close(params.top_p, torch.tensor([0.95, 0.8]))
