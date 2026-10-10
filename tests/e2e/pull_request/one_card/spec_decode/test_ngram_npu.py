from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.one_card.spec_decode.utils import calculate_acceptance_per_pos
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.states import AscendRequestState


@pytest.mark.parametrize("max_model_len", [32, 257, 4096])
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("staged_history", [False, True])
def test_mrv2_ngram_kernel_matches_reference(max_model_len, k, staged_history):
    """Check matching, sparse request slots, fallback, and block boundaries on NPU."""
    device = torch.device("npu")
    generator = torch.Generator().manual_seed(40704)
    rows = [
        [1, 2, 3, 4, 1, 2, 5, 1, 2],
        torch.randint(0, 8, (max_model_len,), generator=generator).tolist(),
        [7],
        [1, 2, 3, 1, 2],
        [21, 22, 23, 24],
    ]
    # Force a longest match across a scan block boundary and at the context cap.
    boundary = 255 if max_model_len >= 1024 else (127 if max_model_len >= 256 else 15)
    suffix = [10, 11, 12, 13, 14]
    rows[1][boundary : boundary + len(suffix)] = suffix
    rows[1][-len(suffix) :] = suffix
    slots = [5, 1, 6, 3, 0]
    num_sampled = [3, 1, 1, 0, 1]
    last_tokens = [2, rows[1][-1], 7, 2, 24]
    token_ids = torch.full((8, max_model_len + 7), -1, dtype=torch.int32, device=device)
    total_len = torch.zeros(8, dtype=torch.int32, device=device)
    last_sampled = torch.zeros((8, 1), dtype=torch.int64, device=device)
    states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=token_ids),
        total_len=SimpleNamespace(gpu=total_len),
    )
    if staged_history:
        states = AscendRequestState(8, max_model_len, 128, k, 32, device, use_dense_all_token_ids=True)
        token_ids = states.all_token_ids.gpu
        total_len = states.total_len.gpu
        token_ids.fill_(-1)
    for row, slot, last in zip(rows, slots, last_tokens):
        if staged_history:
            states.all_token_ids.stage_write(slot, 0, row)
            states.total_len.stage_write_elem(slot, len(row))
        else:
            token_ids[slot, : len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
            total_len[slot] = len(row)
        last_sampled[slot, 0] = last
    if staged_history:
        states.apply_staged_writes()
    history_before = token_ids.clone()
    lengths_before = total_len.clone()
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="ngram_gpu", prompt_lookup_min=2, prompt_lookup_max=5, num_speculative_tokens=k
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )
    speculator = init_speculator(config, device, states)
    batch = SimpleNamespace(num_reqs=len(rows), idx_mapping=torch.tensor(slots, device=device))
    args = dict(
        input_batch=batch,
        attn_metadata=None,
        slot_mappings=None,
        last_hidden_states=torch.empty(0, device=device),
        aux_hidden_states=None,
        num_sampled=torch.tensor(num_sampled, dtype=torch.int32, device=device),
        num_rejected=torch.zeros(len(rows), dtype=torch.int32, device=device),
        last_sampled=last_sampled,
        next_prefill_tokens=torch.empty(0, device=device),
        temperature=torch.empty(0, device=device),
        seeds=torch.empty(0, device=device),
        num_speculative_tokens=k,
    )
    expected = []
    for row, sampled, last in zip(rows, num_sampled, last_tokens):
        matches = [
            (n, pos) for n in range(2, 6) for pos in range(len(row) - n) if sampled and row[pos : pos + n] == row[-n:]
        ]
        continuation = []
        if matches:
            n, pos = max(matches)
            continuation = row[pos + n : pos + n + k]
        expected.append(continuation + [last] * (k - len(continuation)))
    assert speculator.propose(**args).cpu().tolist() == expected
    # A later step with no eligible rows must not reuse scratch from the first call.
    args["num_sampled"].zero_()
    assert speculator.propose(**args).cpu().tolist() == [[last] * k for last in last_tokens]
    assert speculator.propose(**args, dummy_run=True).shape == (len(rows), k)
    # Unlike MRV1 #11925, MRV2 postprocess has already appended accepted tokens.
    torch.testing.assert_close(token_ids, history_before)
    torch.testing.assert_close(total_len, lengths_before)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("enforce_eager", [False, True])
def test_mrv2_ngram_greedy_accuracy(monkeypatch, test_prompts, model_name, async_scheduling, enforce_eager):
    """Compare target token IDs, including chunked prefill and request slot reuse."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    sampling_params = SamplingParams(temperature=0, max_tokens=64)
    common = dict(
        max_model_len=2048,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        enforce_eager=enforce_eager,
        async_scheduling=async_scheduling,
        gpu_memory_utilization=0.7,
    )
    prompts = test_prompts[:8] * 2
    with VllmRunner(model_name, **common) as runner:
        baseline = runner.model.chat(prompts, sampling_params)
    with VllmRunner(
        model_name,
        speculative_config={
            "method": "ngram_gpu",
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5,
            "num_speculative_tokens": 3,
        },
        **common,
    ) as runner:
        speculative = runner.model.chat(prompts, sampling_params)
    assert [o.outputs[0].token_ids for o in speculative] == [o.outputs[0].token_ids for o in baseline]


@pytest.mark.parametrize("num_speculative_tokens", [3])
def test_ngram_npu_async_acceptance(
    test_prompts: list[list[dict[str, Any]]],
    num_speculative_tokens: int,
    model_name: str,
):
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        max_tokens=256,
    )

    speculative_config = {
        "method": "ngram_gpu",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
        "num_speculative_tokens": num_speculative_tokens,
    }

    compilation_config = CompilationConfig(
        cudagraph_mode="PIECEWISE",
        cudagraph_capture_sizes=[12],
    )

    with VllmRunner(
        model_name,
        max_model_len=2048,
        disable_log_stats=False,
        tensor_parallel_size=1,
        max_num_seqs=256,
        distributed_executor_backend="mp",
        gpu_memory_utilization=0.7,
        speculative_config=speculative_config,
        compilation_config=compilation_config,
        async_scheduling=True,
    ) as llm:
        outputs = llm.model.chat(test_prompts, sampling_params)
        metrics = llm.model.get_metrics()

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_tokens = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"Output tokens: {output_tokens}")

    acceptance_per_pos = calculate_acceptance_per_pos(metrics, num_speculative_tokens, Counter, Vector)
    golden = [0.50, 0.30, 0.20]

    match = all(abs(a - b) < 1.0 for a, b in zip(acceptance_per_pos, golden))
    assert match, f"acceptance_per_pos {acceptance_per_pos} does not match golden {golden}"
