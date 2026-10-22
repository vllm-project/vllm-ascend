# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run the production DFlash2 candidate sampler inside an actual ACL graph."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as upstream

import vllm_ascend.patch.worker.patch_v2.patch_dflash_speculator  # noqa: F401
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import _selector_walk_kernel_ascend

_CAPTURE_REQUESTS = 3
_REQUEST_STATES = 5
_NUM_STEPS = 2
_TOP_K = 3
_VOCAB_SIZE = 64
_TOKEN_SENTINEL = -777
_SCORE_SENTINEL = -123.0


@pytest.mark.e2e_coverage(
    arch="dense",
    feature="spec_decode,aclgraph",
    parallel="",
    deploy="",
    hardware="",
    quantization="",
    graph_mode="full_graph",
)
@pytest.mark.parametrize("probabilistic", [False, True], ids=["greedy", "probabilistic"])
@torch.inference_mode()
def test_candidate_sampler_graph_replay_refreshes_requests_and_logits(probabilistic):
    sampler_cls = getattr(upstream, "CandidateSampler", None)
    if sampler_cls is None:
        pytest.skip("This regression targets vLLM's shared DFlash2 CandidateSampler API.")
    assert upstream._selector_walk_kernel is _selector_walk_kernel_ascend
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    graph_sampler = sampler_cls(_REQUEST_STATES, _NUM_STEPS, _TOP_K, device)
    eager_sampler = sampler_cls(_REQUEST_STATES, _NUM_STEPS, _TOP_K, device)
    candidates = torch.zeros((_CAPTURE_REQUESTS, _NUM_STEPS, _TOP_K), dtype=torch.int64, device=device)
    scores = torch.zeros((_CAPTURE_REQUESTS, _NUM_STEPS, _TOP_K, _TOP_K), dtype=torch.float32, device=device)
    sample_positions = torch.zeros(_CAPTURE_REQUESTS * _NUM_STEPS, dtype=torch.int64, device=device)
    request_mapping = torch.full_like(sample_positions, -1)
    temperature = torch.tensor([1.0, 0.7, 0.5, 0.0, 1.2], dtype=torch.float32, device=device)
    seeds = torch.tensor([7, 13, 29, 37, 43], dtype=torch.int64, device=device)
    graph_tokens = torch.full((_CAPTURE_REQUESTS, _NUM_STEPS), _TOKEN_SENTINEL, dtype=torch.int64, device=device)
    eager_tokens = torch.full_like(graph_tokens, _TOKEN_SENTINEL)
    logits_shape = (_REQUEST_STATES, _NUM_STEPS, _VOCAB_SIZE)
    graph_logits = (
        torch.full(logits_shape, -float("inf"), dtype=torch.float32, device=device) if probabilistic else None
    )
    eager_logits = torch.full_like(graph_logits, -float("inf")) if graph_logits is not None else None
    graph_sampler.scores.fill_(_SCORE_SENTINEL)
    eager_sampler.scores.fill_(_SCORE_SENTINEL)

    def sample(sampler, tokens, logits):
        sampler.sample(
            candidates,
            scores,
            _CAPTURE_REQUESTS,
            sample_positions,
            request_mapping,
            temperature,
            seeds,
            tokens,
            logits,
            False,
        )

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        # Compile the actual Ascend kernels before capture. Inert mappings
        # must leave all outputs and the proposal cache untouched.
        sample(graph_sampler, graph_tokens, graph_logits)
        torch.npu.synchronize()
        with torch.npu.graph(graph, stream=stream):
            sample(graph_sampler, graph_tokens, graph_logits)
        torch.npu.synchronize()
        assert torch.all(graph_tokens.cpu() == _TOKEN_SENTINEL)
        assert torch.all(graph_sampler.scores.cpu() == _SCORE_SENTINEL)
        assert torch.count_nonzero(graph_sampler.cached_candidate_ids.cpu()).item() == 0
        if graph_logits is not None:
            assert torch.isneginf(graph_logits.cpu()).all()

        # State 1 moves from batch row 1 to row 0 as the live batch shrinks
        # from three requests to one. The captured launch retains three rows.
        for iteration, state_ids in enumerate(([4, 1, 3], [1], [])):
            num_active = len(state_ids)
            cpu_candidates = (
                torch.arange(_CAPTURE_REQUESTS * _NUM_STEPS * _TOP_K)
                .reshape(_CAPTURE_REQUESTS, _NUM_STEPS, _TOP_K)
                .add(1 + iteration * 20)
            )
            cpu_scores = (
                torch.arange(_CAPTURE_REQUESTS * _NUM_STEPS * _TOP_K * _TOP_K, dtype=torch.float32)
                .reshape(_CAPTURE_REQUESTS, _NUM_STEPS, _TOP_K, _TOP_K)
                .div(17)
                .roll(iteration, dims=-1)
            )
            cpu_mapping = torch.full((_CAPTURE_REQUESTS, _NUM_STEPS), -1, dtype=torch.int64)
            if state_ids:
                cpu_mapping[:num_active] = torch.tensor(state_ids).unsqueeze(1)
            old_candidates = graph_sampler.cached_candidate_ids.cpu().clone()
            old_logits = graph_logits.cpu().clone() if graph_logits is not None else None
            candidates.copy_(cpu_candidates.to(device))
            scores.copy_(cpu_scores.to(device))
            request_mapping.copy_(cpu_mapping.flatten().to(device))
            sample_positions.copy_(
                torch.arange(1 + iteration * 100, 1 + iteration * 100 + sample_positions.numel(), device=device)
            )
            graph_tokens.fill_(_TOKEN_SENTINEL)
            eager_tokens.fill_(_TOKEN_SENTINEL)
            graph_sampler.scores.fill_(_SCORE_SENTINEL)
            eager_sampler.scores.fill_(_SCORE_SENTINEL)

            graph.replay()
            sample(eager_sampler, eager_tokens, eager_logits)
            torch.npu.synchronize()
            torch.testing.assert_close(graph_tokens.cpu(), eager_tokens.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(graph_sampler.scores.cpu(), eager_sampler.scores.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(
                graph_sampler.cached_candidate_ids.cpu(), eager_sampler.cached_candidate_ids.cpu(), rtol=0, atol=0
            )
            assert torch.all(graph_tokens[num_active:].cpu() == _TOKEN_SENTINEL)
            assert torch.all(graph_sampler.scores[num_active:_CAPTURE_REQUESTS].cpu() == _SCORE_SENTINEL)
            if graph_logits is None:
                continue

            actual_logits = graph_logits.cpu()
            torch.testing.assert_close(actual_logits, eager_logits.cpu(), rtol=0, atol=0)
            for state in set(range(_REQUEST_STATES)) - set(state_ids):
                torch.testing.assert_close(actual_logits[state], old_logits[state], rtol=0, atol=0)
                torch.testing.assert_close(
                    graph_sampler.cached_candidate_ids[state].cpu(), old_candidates[state], rtol=0, atol=0
                )
            for row, state in enumerate(state_ids):
                torch.testing.assert_close(
                    graph_sampler.cached_candidate_ids[state].cpu(), cpu_candidates[row], rtol=0, atol=0
                )
                for step in range(_NUM_STEPS):
                    ids = cpu_candidates[row, step]
                    torch.testing.assert_close(
                        actual_logits[state, step, ids], graph_sampler.scores[row, step].cpu(), rtol=0, atol=0
                    )
                    removed_ids = [token for token in old_candidates[state, step].tolist() if token not in ids.tolist()]
                    if removed_ids:
                        assert torch.isneginf(actual_logits[state, step, removed_ids]).all()
