# SPDX-License-Identifier: Apache-2.0
"""Dense, unquantized MLA Q replication A/B acceptance on four NPUs."""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("graph_mode", [None, "FULL_DECODE_ONLY", "PIECEWISE"])
def test_dense_mla_q_replication(dcp_size, graph_mode, monkeypatch, dcp_qrep_model):
    # Ensure the legacy override cannot turn both A/B runs into the same mode.
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    prompts = [
        "The capital of France is",
        "Explain how context parallel attention works. " * 40,
        "A computer scientist studies algorithms. " * 70,
    ]
    results = []
    for enabled in (False, True):
        with VllmRunner(
            dcp_qrep_model,
            dtype="bfloat16",
            tensor_parallel_size=4,
            decode_context_parallel_size=dcp_size,
            dcp_q_replicate=enabled,
            enforce_eager=graph_mode is None,
            compilation_config=(
                {"cudagraph_mode": graph_mode, "cudagraph_capture_sizes": [1, 2, 4, 8, 16]} if graph_mode else {}
            ),
            max_model_len=2048,
            max_num_seqs=3,
            max_num_batched_tokens=128,
            block_size=128,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            additional_config={"enable_mlapo": False},
        ) as runner:
            # Fixed teacher-forced input: compare each supplied token's score,
            # rather than conditioning later comparisons on sampled tokens.
            scored = runner.model.generate(prompts, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
            greedy = runner.generate_greedy(prompts, 16)
            cached = runner.generate_greedy(prompts, 16)
            assert [x[0] for x in greedy] == [x[0] for x in cached]
            results.append((scored, greedy))
    baseline, replicated = results
    for left, right in zip(baseline[0], replicated[0]):
        assert left.prompt_token_ids == right.prompt_token_ids
        for token, lp, rp in zip(left.prompt_token_ids, left.prompt_logprobs, right.prompt_logprobs):
            if lp is None:
                assert rp is None
                continue
            # Fixed BF16 acceptance threshold; do not relax after a failure.
            assert lp[token].logprob == pytest.approx(rp[token].logprob, abs=0.05, rel=0)
    assert [x[0] for x in baseline[1]] == [x[0] for x in replicated[1]]


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("graph_mode", [None, "FULL_DECODE_ONLY"])
@pytest.mark.parametrize("method", ["ngram"])
def test_speculative_q_replication(dcp_size, graph_mode, method, monkeypatch, dcp_qrep_model):
    model = dcp_qrep_model
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    # Repeated text helps ngram propose candidates. Unequal prompt lengths and
    # three requests also exercise graph padding and mixed scheduling.
    prompts = [
        "Continue this sequence: " + "one two three four " * 20,
        "Explain context parallel attention. " * 40,
        "The capital of France is",
    ]
    spec = {"method": method, "num_speculative_tokens": 3}
    spec.update(prompt_lookup_min=2, prompt_lookup_max=5)
    outputs = []
    # Independent ordinary decode reference, then the same speculative path
    # with Q replication off/on; each graph run performs repeated generation.
    for enabled, speculative in [(False, None), (False, spec), (True, spec)]:
        with VllmRunner(
            model,
            dtype="bfloat16",
            tensor_parallel_size=4,
            decode_context_parallel_size=dcp_size,
            dcp_q_replicate=enabled,
            enforce_eager=graph_mode is None,
            compilation_config=(
                {"cudagraph_mode": graph_mode, "cudagraph_capture_sizes": [1, 2, 4, 8, 16]} if graph_mode else {}
            ),
            speculative_config=speculative,
            max_model_len=2048,
            max_num_seqs=3,
            max_num_batched_tokens=128,
            block_size=128,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            additional_config={"enable_mlapo": False},
        ) as runner:
            first = runner.generate_greedy(prompts, 24)
            replay = runner.generate_greedy(prompts, 24)
            assert [x[0] for x in first] == [x[0] for x in replay]
            outputs.append([x[0] for x in first])
    assert outputs[0] == outputs[1] == outputs[2]
