# SPDX-License-Identifier: Apache-2.0
"""Four-NPU acceptance for unquantized, head-sharded SFA/DSA Q replication."""

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner


@pytest.mark.parametrize("graph_mode", [None, "FULL_DECODE_ONLY"])
@pytest.mark.parametrize("interleave", [1, 128])
@pytest.mark.parametrize("method", [None, "ngram", "mtp"])
def test_sparse_dcp_q_replication(graph_mode, interleave, method, request, monkeypatch):
    fixture = "dcp_qrep_sparse_mtp_model" if method == "mtp" else "dcp_qrep_sparse_model"
    model = request.getfixturevalue(fixture)
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    prompts = [
        "Continue the sequence: " + "one two three four " * 30,
        "Explain how sparse attention selects tokens. " * 60,
        "The capital of France is",
    ]
    spec = None if method is None else {"method": method, "num_speculative_tokens": 3}
    if method == "ngram":
        spec.update(prompt_lookup_min=2, prompt_lookup_max=5)
    outputs, scores = [], []
    runs = [(False, None), (True, None)] if spec is None else [(False, None), (False, spec), (True, spec)]
    for enabled, speculative in runs:
        with VllmRunner(
            model,
            dtype="bfloat16",
            tensor_parallel_size=4,
            decode_context_parallel_size=4,
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
            cp_kv_cache_interleave_size=interleave,
            kv_cache_dtype="auto",
            attention_config={"indexer_kv_dtype": "bf16"},
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            additional_config={"enable_mlapo": False, "enable_dsa_cp": False},
        ) as runner:
            if spec is None:
                scores.append(
                    runner.model.generate(prompts, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1))
                )
            first = runner.generate_greedy(prompts, 24)
            replay = runner.generate_greedy(prompts, 24)
            assert [x[0] for x in first] == [x[0] for x in replay]
            outputs.append([x[0] for x in first])
    assert all(out == outputs[0] for out in outputs[1:])
    if scores:
        for left, right in zip(*scores):
            assert left.prompt_token_ids == right.prompt_token_ids
            for token, lp, rp in zip(left.prompt_token_ids, left.prompt_logprobs, right.prompt_logprobs):
                if lp is None:
                    assert rp is None
                else:
                    assert lp[token].logprob == pytest.approx(rp[token].logprob, abs=0.05, rel=0)
