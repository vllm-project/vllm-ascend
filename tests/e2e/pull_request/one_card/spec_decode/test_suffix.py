# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Model-level correctness and corpus reuse tests for MRV2 suffix decoding."""

from __future__ import annotations

from typing import Any

import pytest
from vllm import SamplingParams
from vllm.outputs import RequestOutput

from tests.e2e.conftest import VllmRunner

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# Match compare_logprobs()'s default decode tolerance in pull_request/utils.py.
_DECODE_LOGPROB_ATOL = 2 * 0.0689


def _assert_greedy_outputs_close(baseline: list[RequestOutput], speculative: list[RequestOutput]) -> None:
    """Compare distributions at the first divergence of a shared history."""
    assert len(baseline) == len(speculative)
    for prompt_idx, (base_output, spec_output) in enumerate(zip(baseline, speculative, strict=True)):
        for name, output in (("baseline", base_output), ("suffix", spec_output)):
            assert output.finished and len(output.outputs) == 1, (prompt_idx, name)
            completion = output.outputs[0]
            assert completion.token_ids, (prompt_idx, name, "empty completion")
            assert completion.logprobs is not None, (prompt_idx, name, "missing logprobs")
            assert len(completion.token_ids) == len(completion.logprobs), (prompt_idx, name)

        base = base_output.outputs[0]
        spec = spec_output.outputs[0]
        assert base.logprobs is not None and spec.logprobs is not None
        for token_idx, (base_id, spec_id) in enumerate(zip(base.token_ids, spec.token_ids)):
            if base_id == spec_id:
                continue
            base_topk = base.logprobs[token_idx]
            spec_topk = spec.logprobs[token_idx]
            detail = (prompt_idx, token_idx, base_id, spec_id, base_topk, spec_topk)
            # Only the first mismatch conditions on the same token history.
            assert base_id in spec_topk and spec_id in base_topk, detail
            for topk in (base_topk, spec_topk):
                gap = abs(topk[base_id].logprob - topk[spec_id].logprob)
                assert gap <= _DECODE_LOGPROB_ATOL, detail
            for token_id in (base_id, spec_id):
                delta = abs(base_topk[token_id].logprob - spec_topk[token_id].logprob)
                assert delta <= _DECODE_LOGPROB_ATOL, detail
            print(f"BF16 near-tie: prompt={prompt_idx}, token={token_idx}, baseline={base_id}, suffix={spec_id}")
            break


def _speculation_counts(runner: VllmRunner) -> tuple[int, int]:
    """Return cumulative draft and accepted-token counts."""
    names = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens")
    values = {name: 0 for name in names}
    found = set()
    for metric in runner.model.get_metrics():
        if metric.name in values:
            found.add(metric.name)
            values[metric.name] += int(metric.value)
    assert found == set(names), f"Missing suffix metrics: {set(names) - found}"
    return values[names[0]], values[names[1]]


@pytest.mark.parametrize(
    ("enforce_eager", "async_scheduling", "adaptive"),
    [
        pytest.param(True, False, False, id="eager-sync"),
        pytest.param(False, True, False, id="graph-async"),
        pytest.param(False, True, True, id="graph-async-adaptive"),
    ],
)
def test_mrv2_suffix_greedy_accuracy(monkeypatch, enforce_eager: bool, async_scheduling: bool, adaptive: bool) -> None:
    """Compare greedy distributions with and without suffix decoding."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    sampling_params = SamplingParams(temperature=0, max_tokens=64, logprobs=5)
    common: dict[str, Any] = dict(
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        async_scheduling=async_scheduling,
        gpu_memory_utilization=0.5,
        compilation_config={
            "cudagraph_mode": "NONE" if enforce_eager else "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
    )
    requests = (
        "Continue the pattern for twenty more words: red blue green red blue green red blue green",
        "Repeat the words 'sun moon star' ten times, without any explanation.",
        "Continue the sequence for ten more groups: one two three, one two three, one two three,",
        "Explain how a rainbow forms, describing the path of sunlight through a raindrop.",
    )
    prompts = [[{"role": "user", "content": request}] for request in requests] * 2
    with VllmRunner(MODEL, **common) as runner:
        baseline = runner.model.chat(prompts, sampling_params)
    with VllmRunner(
        MODEL,
        speculative_config={
            "method": "suffix",
            "num_speculative_tokens": 3,
            "suffix_decoding_corpus_tokens": 65536,
            "enable_adaptive_verification": adaptive,
        },
        disable_log_stats=False,
        **common,
    ) as runner:
        speculative = runner.model.chat(prompts, sampling_params)
        num_drafts, num_accepted = _speculation_counts(runner)

    assert len(speculative) == len(baseline) == len(prompts)
    _assert_greedy_outputs_close(baseline, speculative)
    assert num_drafts > 0
    assert num_accepted > 0


def test_mrv2_suffix_cross_request_acceptance(monkeypatch) -> None:
    """Compare serial corpus warmup with the same workload and corpus disabled."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    requests = (
        "Explain how a rainbow forms, describing the path of sunlight through a raindrop.",
        "Describe how to prepare a vegetable soup, from choosing ingredients to serving it.",
        "Explain the differences between a compiler and an interpreter, with examples.",
        "Describe the journey of a river from its source in the mountains to the sea.",
    )
    k = 8
    output_len = 96
    sampling_params = SamplingParams(temperature=0, min_tokens=output_len, max_tokens=output_len, ignore_eos=True)
    # Reuse identical input token IDs across both engines and passes.
    prompt_ids = None
    counts = {}
    lengths = {}
    for corpus_tokens in (0, 65536):
        with VllmRunner(
            MODEL,
            dtype="bfloat16",
            seed=0,
            tensor_parallel_size=1,
            max_model_len=1024,
            max_num_seqs=1,
            max_num_batched_tokens=128,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            async_scheduling=False,
            gpu_memory_utilization=0.5,
            disable_log_stats=False,
            speculative_config={
                "method": "suffix",
                "num_speculative_tokens": k,
                "suffix_decoding_max_tree_depth": 24,
                "suffix_decoding_corpus_tokens": corpus_tokens,
                "enable_adaptive_verification": False,
            },
        ) as runner:
            if prompt_ids is None:
                tokenizer = runner.model.get_tokenizer()
                prompt_ids = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": request}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                    )
                    for request in requests
                ]
            before_drafts, before_accepted = 0, 0
            for phase in ("cold", "warm"):
                for tokens in prompt_ids:
                    outputs = runner.model.generate([{"prompt_token_ids": tokens}], sampling_params)
                    assert len(outputs) == 1 and outputs[0].finished
                    assert len(outputs[0].outputs) == 1
                    assert len(outputs[0].outputs[0].token_ids) == output_len
                after_drafts, after_accepted = _speculation_counts(runner)
                drafts = after_drafts - before_drafts
                accepted = after_accepted - before_accepted
                assert drafts > 0, f"No drafts: corpus_tokens={corpus_tokens}, phase={phase}"
                assert 0 <= accepted <= k * drafts, (corpus_tokens, phase, drafts, accepted)
                counts[corpus_tokens, phase] = (drafts, accepted)
                # Acceptance length from this pass, not cumulative counters.
                lengths[corpus_tokens, phase] = 1 + accepted / drafts
                before_drafts, before_accepted = after_drafts, after_accepted

    off_cold = lengths[0, "cold"]
    off_warm = lengths[0, "warm"]
    on_cold = lengths[65536, "cold"]
    on_warm = lengths[65536, "warm"]
    summary = f"acceptance lengths={lengths}, (drafts, accepted)={counts}"
    print(summary)
    assert on_warm > on_cold, summary
    assert on_warm > off_warm, summary
    assert on_warm - on_cold > off_warm - off_cold, summary
