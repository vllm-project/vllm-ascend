# SPDX-License-Identifier: Apache-2.0
"""Dense and sparse MLA Q replication A/B acceptance on four NPUs."""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize(
    "model_kind,dcp_size,graph_mode,interleave",
    [("dense", dcp, graph, 1) for dcp in (2, 4) for graph in (None, "FULL_DECODE_ONLY", "PIECEWISE")]
    + [("sparse", 4, graph, interleave) for graph in (None, "FULL_DECODE_ONLY") for interleave in (1, 128)],
)
def test_dcp_q_replication(model_kind, dcp_size, graph_mode, interleave, monkeypatch, dcp_q_replicate_model):
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    sparse = model_kind == "sparse"
    prompts = (
        [
            "Continue the sequence: " + "one two three four " * 30,
            "Explain how sparse attention selects tokens. " * 60,
            "The capital of France is",
        ]
        if sparse
        else [
            "The capital of France is",
            "Explain how context parallel attention works. " * 40,
            "A computer scientist studies algorithms. " * 70,
        ]
    )
    results = []
    for enabled in (False, True):
        with VllmRunner(
            dcp_q_replicate_model,
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
            additional_config={"enable_mlapo": False, "enable_dsa_cp": False},
            **({"cp_kv_cache_interleave_size": interleave} if sparse else {}),
        ) as runner:
            # Teacher forcing keeps both runs on identical input tokens.
            scored = runner.model.generate(prompts, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
            greedy = [x[0] for x in runner.generate_greedy(prompts, 24 if sparse else 16)]
            cached = [x[0] for x in runner.generate_greedy(prompts, 24 if sparse else 16)]
            assert greedy == cached
            results.append((scored, greedy))
    (baseline, baseline_tokens), (replicated, replicated_tokens) = results
    assert baseline_tokens == replicated_tokens
    for left, right in zip(baseline, replicated, strict=True):
        assert left.prompt_token_ids == right.prompt_token_ids
        for token, lp, rp in zip(left.prompt_token_ids, left.prompt_logprobs, right.prompt_logprobs, strict=True):
            if lp is None:
                assert rp is None
            else:
                assert lp[token].logprob == pytest.approx(rp[token].logprob, abs=0.05, rel=0)
