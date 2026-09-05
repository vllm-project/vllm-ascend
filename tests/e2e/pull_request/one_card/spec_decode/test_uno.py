from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import UNO, calculate_acceptance_per_pos

MAX_NUM_SEQS = 32
FORWARD_WIDTH = 8
# Comparing full sequences would be comparing the numerics of two different
# kernel shapes; a bounded greedy prefix is the contract that actually holds.
PARITY_TOKENS = 32


class UnoGraphParity:
    def enable_uno_graph_parity(self):
        drafter = self.model_runner.drafter
        assert drafter._draft_graph_batch_sizes == {1, 2}
        original = drafter._forward
        self._uno_graph_parity_stats = {"calls": 0, "values": 0, "request_counts": []}

        def compare(*args, **kwargs):
            assert kwargs.get("compute_logits") and not kwargs.get("capture", False)
            num_reqs = args[3]
            assert num_reqs in drafter._draft_graph_batch_sizes
            # Clone before the reference forward can reuse graph-pool storage.
            actual = original(*args, **kwargs).clone()
            buckets = drafter._draft_graph_batch_sizes
            drafter._draft_graph_batch_sizes = set()
            try:
                expected = original(*args, **kwargs)
            finally:
                drafter._draft_graph_batch_sizes = buckets
            # Both forwards overwrite the same draft window, with the same
            # input noise, positions and committed prefix. No sampling occurs.
            assert torch.equal(actual, expected), (
                f"UNO same-input graph/eager logits differ: requests={num_reqs}, "
                f"max_abs={(actual.float() - expected.float()).abs().max().item()}"
            )
            self._uno_graph_parity_stats["calls"] += 1
            self._uno_graph_parity_stats["values"] += actual.numel()
            self._uno_graph_parity_stats["request_counts"].append(num_reqs)
            return actual

        drafter._forward = compare

    def get_uno_graph_parity(self):
        return self._uno_graph_parity_stats


@pytest.mark.parametrize("method", UNO.keys())
def test_uno_full_decode_graph_matches_eager_logits_across_batch_changes(method: str):
    """Compare replay/eager over identical noise, positions and KV windows.

    Fresh UNO noise can change verification shapes and break a BF16 argmax
    tie differently. A same-input full-vocabulary comparison isolates graph
    computation without relaxing tolerance or truncating generated outputs.
    """
    prompts = ["The capital of France is", "List the first five prime numbers:"]
    params = SamplingParams(temperature=0, max_tokens=32, logprobs=5)
    with VllmRunner(
        UNO[method]["main"],
        seed=0,
        max_model_len=512,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [9, 18]},
        speculative_config={"method": "uno", "model": UNO[method]["adapter"], "num_speculative_tokens": FORWARD_WIDTH},
        worker_extension_cls="tests.e2e.pull_request.one_card.spec_decode.test_uno.UnoGraphParity",
    ) as llm:
        # String RPC uses the default serializer; no callable/pickle fallback.
        llm.model.collective_rpc("enable_uno_graph_parity")
        for batch in (prompts, prompts[:1], prompts):
            outputs = llm.model.generate(batch, params)
            assert all(len(output.outputs[0].token_ids) == 32 for output in outputs)
        metrics = llm.model.get_metrics()
        (stats,) = llm.model.collective_rpc("get_uno_graph_parity")
    assert stats["calls"] == len(stats["request_counts"]) > 0
    assert stats["values"] > 0 and set(stats["request_counts"]) == {1, 2}
    assert any(left == 1 and right == 2 for left, right in zip(stats["request_counts"], stats["request_counts"][1:]))
    acceptance = calculate_acceptance_per_pos(metrics, FORWARD_WIDTH, Counter, Vector)
    assert acceptance[0] > 0.95
    assert acceptance[1] > 0.0, "A base-only draft can pass output checks while losing its acceleration."
    print(f"UNO_GRAPH_PARITY_PASS {stats}")


def _prompt(model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain speculative decoding in two sentences."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@pytest.mark.parametrize("method", UNO.keys())
def test_uno_greedy_output_matches_autoregressive(method: str):
    """UNO is lossless: greedy UNO must reproduce greedy AR token for token.

    Every emitted token is either the target's own argmax (the accepted or
    corrected positions) or the clean token drawn from the base row, so the two
    engines have to agree until kernel-shape numerics diverge.
    """
    main_model_name = UNO[method]["main"]
    adapter_name = UNO[method]["adapter"]
    prompts = [_prompt(main_model_name)]
    sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=PARITY_TOKENS)

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
    ) as ar_llm:
        ar_outputs = ar_llm.model.generate(prompts, sampling_params)

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        speculative_config={
            "method": "uno",
            "model": adapter_name,
            "num_speculative_tokens": FORWARD_WIDTH,
        },
    ) as uno_llm:
        uno_outputs = uno_llm.model.generate(prompts, sampling_params)

    for ar_output, uno_output in zip(ar_outputs, uno_outputs, strict=True):
        ar_tokens = list(ar_output.outputs[0].token_ids)[:PARITY_TOKENS]
        uno_tokens = list(uno_output.outputs[0].token_ids)[:PARITY_TOKENS]
        assert uno_tokens == ar_tokens, f"UNO diverged from AR decoding.\n  AR : {ar_tokens}\n  UNO: {uno_tokens}"


@pytest.mark.parametrize("method", UNO.keys())
def test_uno_clean_token_is_always_accepted(method: str):
    """Position 0 acceptance is a structural invariant, not a tuned number.

    The first proposal comes from the draft forward's base-weight row, which
    computes exactly the distribution the verify forward's first row evaluates.
    Greedy acceptance of that token is therefore guaranteed up to floating-point
    ties, so a rate below ~1.0 means the seed row is receiving a LoRA delta or
    is attending to the noise rows -- both of which leave the *output* correct
    and are invisible in generated text.

    The remaining positions are the ones that depend on the adapter's quality
    and on hardware; they are only asserted to be nonzero, which is what
    distinguishes "the gated routing reached the model" from "the draft ran on
    base weights".
    """
    main_model_name = UNO[method]["main"]
    adapter_name = UNO[method]["adapter"]
    prompts = [_prompt(main_model_name)]
    sampling_params = SamplingParams(temperature=0, ignore_eos=False, max_tokens=256)

    with VllmRunner(
        main_model_name,
        max_model_len=4096,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=MAX_NUM_SEQS,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        speculative_config={
            "method": "uno",
            "model": adapter_name,
            "num_speculative_tokens": FORWARD_WIDTH,
        },
    ) as llm:
        llm.model.generate(prompts, sampling_params)
        metrics = llm.model.get_metrics()

    acceptance_per_pos = calculate_acceptance_per_pos(metrics, FORWARD_WIDTH, Counter, Vector)
    assert len(acceptance_per_pos) == FORWARD_WIDTH
    assert acceptance_per_pos[0] > 0.95, (
        f"the clean token must be accepted almost always, got {acceptance_per_pos[0]:.3f}; "
        "the draft forward's seed row is not running on base weights"
    )
    assert acceptance_per_pos[1] > 0.0, (
        f"no diffusion proposal was ever accepted ({acceptance_per_pos}); "
        "the gated LoRA routing most likely never reached the model"
    )
