# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend coverage for the upstream Model Runner V2 thinking-budget path."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.thinking_budget import ThinkingBudgetState
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as upstream_rejection_sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.ascend_config import RejectionSamplerConfig
from vllm_ascend.worker.v2.spec_decode import rejection_sampler_utils
from vllm_ascend.worker.v2.states import AscendRequestState

DEVICE = torch.device("npu")
START = 90
END = 91
END_A = 92
END_B = 93
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]
    natural_reasoning_end_token_ids = [END]


class MockMultiTokenEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END_A, END_B]


class MockDistinctEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END]


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> RequestState:
    req_states = AscendRequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 1),
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    return req_states


def _apply(
    state: ThinkingBudgetState,
    logits: torch.Tensor,
    input_ids: list[int],
    local_pos: list[int],
) -> torch.Tensor:
    idx_mapping = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.tensor([3] * len(input_ids), dtype=torch.int32, device=DEVICE)
    input_ids_t = torch.tensor(input_ids, dtype=torch.int32, device=DEVICE)
    local_pos_t = torch.tensor(local_pos, dtype=torch.int32, device=DEVICE)
    # vLLM #56497 passes the step's batch layout to apply() via LogitsContext.
    # pos / seq_lens_upper_bound_np are required fields but unused by the
    # thinking-budget op.
    ctx = LogitsContext(
        expanded_idx_mapping=expanded_idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping.cpu().numpy(),
        expanded_local_pos=local_pos_t,
        input_ids=input_ids_t,
        pos=torch.zeros(len(input_ids), dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.zeros(1, dtype=np.int32),
    )
    state.apply(logits, ctx)
    return logits.cpu()


def _apply_batch(state, names, inputs):
    """Construct verification rows in batch order, independently of state slots."""
    slots = [state.req_states.req_id_to_index[name] for name in names]
    lengths = [len(tokens) for tokens in inputs]
    ctx = LogitsContext(
        idx_mapping=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        idx_mapping_np=np.array(slots, dtype=np.int32),
        expanded_idx_mapping=torch.tensor(
            [slot for slot, length in zip(slots, lengths) for _ in range(length)], dtype=torch.int32, device=DEVICE
        ),
        expanded_local_pos=torch.tensor(
            [pos for length in lengths for pos in range(length)], dtype=torch.int32, device=DEVICE
        ),
        input_ids=torch.tensor([token for tokens in inputs for token in tokens], dtype=torch.int32, device=DEVICE),
        pos=torch.zeros(sum(lengths), dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.zeros(len(names), dtype=np.int32),
    )
    logits = torch.zeros(sum(lengths), VOCAB_SIZE, device=DEVICE)
    logits[:, 10] = 1
    state.apply(logits, ctx)
    return logits.argmax(-1).cpu().tolist()


@pytest.mark.parametrize("reverse", [False, True])
def test_v2_budget_lifecycle_reorder_eos_and_resumption(reverse):
    requests = _make_req_states([1, START, 10], prompt_len=2)
    state = ThinkingBudgetState(requests, MockReasoningConfig())
    state.add_request(requests.req_id_to_index["req"], SamplingParams(thinking_token_budget=1))
    for name, budget in [("zero", 0), ("long", 8), ("plain", None)]:
        requests.add_request(name, 2, [1, START, 10], 3, 32)
        state.add_request(requests.req_id_to_index[name], SamplingParams(thinking_token_budget=budget))
    requests.apply_staged_writes()
    state.apply_staged_writes()
    names = ["req", "zero", "long", "plain"]
    if reverse:
        names.reverse()
    expected = {"req": END, "zero": END, "long": 10, "plain": 10}
    assert _apply_batch(state, names, [[10]] * 4) == [expected[name] for name in names]
    # Remove an EOS-finished request and a preempted request. Fill one old slot
    # with a different budget before resuming from committed history.
    requests.remove_request("zero")
    requests.remove_request("req")
    requests.add_request("replacement", 2, [1, START, 10, END, 20], 5, 32)
    state.add_request(requests.req_id_to_index["replacement"], SamplingParams(thinking_token_budget=0))
    requests.add_request("req", 2, [1, START, 10], 3, 32)
    state.add_request(requests.req_id_to_index["req"], SamplingParams(thinking_token_budget=1))
    requests.apply_staged_writes()
    state.apply_staged_writes()
    assert _apply_batch(state, ["long", "replacement", "req", "plain"], [[10]] * 4) == [10, 10, END, 10]


@pytest.mark.parametrize("committed", [1, 2, 3])
@pytest.mark.parametrize("draft_length", [0, 1, 3, 5])
def test_v2_rejected_draft_prefix_does_not_consume_next_step_budget(committed, draft_length):
    requests = _make_req_states([1, START, 10], prompt_len=2)
    state = ThinkingBudgetState(requests, MockReasoningConfig())
    slot = requests.req_id_to_index["req"]
    state.add_request(slot, SamplingParams(thinking_token_budget=3))
    state.apply_staged_writes()
    # A speculative natural end must not leak from in-flight tokens into the
    # committed marker cache if that draft is rejected.
    _apply_batch(state, ["req"], [[10, END, 20, 21]])
    assert state.cached_last_end[slot].item() == -1
    history = [1, START] + [10] * committed
    requests.all_token_ids.stage_write(slot, 0, history)
    requests.total_len.stage_write_elem(slot, len(history))
    requests.apply_staged_writes()
    actual = _apply_batch(state, ["req"], [[10] * (draft_length + 1)])
    assert actual == [10 if committed + pos < 3 else END for pos in range(draft_length + 1)]


@pytest.mark.parametrize("progress", [0, 1])
def test_v2_preempted_multi_token_end_continues_after_slot_reuse(progress):
    history = [1, START, 10, 11] + [END_A][:progress]
    requests = _make_req_states(history, prompt_len=2)
    state = ThinkingBudgetState(requests, MockMultiTokenEndReasoningConfig())
    slot = requests.req_id_to_index["req"]
    state.add_request(slot, SamplingParams(thinking_token_budget=2))
    state.apply_staged_writes()
    assert _apply_batch(state, ["req"], [[history[-1]]]) == [[END_A, END_B][progress]]
    requests.remove_request("req")
    requests.add_request("req", 2, history, len(history), 32)
    state.add_request(requests.req_id_to_index["req"], SamplingParams(thinking_token_budget=2))
    requests.apply_staged_writes()
    state.apply_staged_writes()
    assert _apply_batch(state, ["req"], [[history[-1]]]) == [[END_A, END_B][progress]]


@pytest.mark.parametrize("temperature", [0.0, 0.6])
@pytest.mark.parametrize("budget", [0, 2, 4, None])
@pytest.mark.parametrize("verification", ["standard", "entropy", "block"])
@pytest.mark.parametrize("draft_logits_mode", ["none", "tensor", "cache"])
def test_v2_rejection_sampler_budget_boundary_and_plain_neighbor(
    monkeypatch, temperature, budget, verification, draft_logits_mode
):
    # Direct sampler tests have no engine to initialize Ascend configuration.
    config = RejectionSamplerConfig(enable_entropy_verify=verification == "entropy")
    monkeypatch.setattr(
        rejection_sampler_utils, "get_ascend_config", lambda: SimpleNamespace(rejection_sampler_config=config)
    )
    requests = _make_req_states([1, START, 10], prompt_len=2)
    requests.add_request("plain", 2, [1, START, 10], 3, 32)
    requests.apply_staged_writes()
    sampler = Sampler(
        SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=requests,
        num_speculative_tokens=3,
    )
    slots = [requests.req_id_to_index[name] for name in ["plain", "req"]]
    for slot, value in zip(slots, [None, budget]):
        sampler.add_request(slot, SamplingParams(temperature=temperature, seed=42, thinking_token_budget=value))
    sampler.apply_staged_writes()
    rejection = RejectionSampler.__new__(RejectionSampler)
    rejection.sampler = sampler
    rejection.num_speculative_steps = 3
    rejection.synthetic_conditional_rates = None
    rejection.use_block_verification = verification == "block"
    rejection.fly_window_size = 0
    rejection.fly_entropy_threshold = 0.3
    rejection.fly_entropy_top_k = 3
    rejection.watermark_key = None
    mapping = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
    expanded = mapping.repeat_interleave(4)
    local = torch.tensor([0, 1, 2, 3] * 2, dtype=torch.int32, device=DEVICE)
    inputs = torch.full((8,), 10, dtype=torch.int32, device=DEVICE)
    logits = torch.full((8, VOCAB_SIZE), -torch.inf, device=DEVICE)
    logits[:, 10] = 0
    draft_logits = None
    if draft_logits_mode != "none":
        values = torch.full((4, 3, VOCAB_SIZE), -torch.inf, device=DEVICE)
        values[:, :, 10] = 0
        values[:, :, END] = 0
        if draft_logits_mode == "tensor":
            draft_logits = values
        else:
            cache_type = getattr(upstream_rejection_sampler, "DraftLogitsCache", None)
            if cache_type is None:
                pytest.skip("This upstream release does not use DraftLogitsCache")
            assert cache_type is not None
            draft_logits = cache_type(4, 3, VOCAB_SIZE, dtype=torch.float32, fill=-torch.inf, device=DEVICE)
            draft_logits.values.copy_(values)
    # v0.33.0 separates the verification tokens from the history tokens.
    # All drafts in this fixture are valid, so both inputs are identical.
    # The verified main revision uses draft_sampled directly for verification.
    verify_kwargs = {}
    if "verify_draft_sampled" in inspect.signature(rejection._verify).parameters:
        verify_kwargs["verify_draft_sampled"] = inputs
    _, sampled, counts = rejection._verify(
        logits,
        draft_logits,
        inputs,
        local + 2,
        torch.tensor([0, 4, 8], dtype=torch.int32, device=DEVICE),
        mapping,
        np.array(slots, dtype=np.int32),
        expanded,
        local,
        np.array([6, 6], dtype=np.int32),
        **verify_kwargs,
    )
    sampled, counts = sampled.cpu(), counts.cpu().tolist()
    assert counts[0] == 4
    assert sampled[0, :4].tolist() == [10] * 4
    expected = [10] * 4 if budget is None else [10] * max(budget - 1, 0) + [END]
    assert sampled[1, : counts[1]].tolist() == expected


@pytest.mark.parametrize("temperature", [0.0, 0.6])
def test_v2_sampler_eos_removal_and_budgeted_slot_reuse(temperature):
    eos = 94
    requests = _make_req_states([1, START, 10], prompt_len=2)
    requests.add_request("finishing", 2, [1, START, 10], 3, 32)
    requests.apply_staged_writes()
    sampler = Sampler(
        SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=requests,
    )
    slots = [requests.req_id_to_index[name] for name in ["finishing", "req"]]
    for slot, budget in zip(slots, [None, 1]):
        sampler.add_request(slot, SamplingParams(temperature=temperature, seed=42, thinking_token_budget=budget))
    sampler.apply_staged_writes()

    def sample():
        mapping = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
        logits = torch.full((2, VOCAB_SIZE), -torch.inf, device=DEVICE)
        logits[:, eos] = 0
        sampled, _ = sampler.sample(
            logits,
            mapping,
            mapping,
            np.array(slots, dtype=np.int32),
            torch.full((2,), 2, dtype=torch.int32, device=DEVICE),
            torch.full((2,), 10, dtype=torch.int32, device=DEVICE),
            torch.zeros(2, dtype=torch.int32, device=DEVICE),
            np.array([3, 3], dtype=np.int32),
        )
        return sampled.cpu().tolist()

    assert sample() == [eos, END]
    requests.remove_request("finishing")
    requests.add_request("replacement", 2, [1, START, 10], 3, 32)
    assert requests.req_id_to_index["replacement"] == slots[0]
    sampler.add_request(slots[0], SamplingParams(temperature=temperature, seed=42, thinking_token_budget=0))
    requests.apply_staged_writes()
    sampler.apply_staged_writes()
    assert sample() == [END, END]


@pytest.mark.parametrize("verification_rows", [1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_v2_mixed_budget_preserves_unlimited_logits_bitwise(verification_rows, reverse):
    requests = _make_req_states([1, START, 10], prompt_len=2)
    requests.add_request("unlimited", 2, [1, START, 10], 3, 32)
    requests.apply_staged_writes()
    state = ThinkingBudgetState(requests, MockReasoningConfig())
    for name, budget in [("req", 0), ("unlimited", None)]:
        state.add_request(requests.req_id_to_index[name], SamplingParams(thinking_token_budget=budget))
    state.apply_staged_writes()
    names = ["req", "unlimited"]
    if reverse:
        names.reverse()
    slots = np.array([requests.req_id_to_index[name] for name in names], dtype=np.int32)
    mapping = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
    rows = len(names) * verification_rows
    ctx = LogitsContext(
        idx_mapping=mapping,
        idx_mapping_np=slots,
        expanded_idx_mapping=mapping.repeat_interleave(verification_rows),
        expanded_local_pos=torch.arange(verification_rows, dtype=torch.int32, device=DEVICE).repeat(len(names)),
        input_ids=torch.full((rows,), 10, dtype=torch.int32, device=DEVICE),
        pos=torch.zeros(rows, dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.full(len(names), 3, dtype=np.int32),
    )
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(rows, VOCAB_SIZE, generator=generator).to(DEVICE)
    before = logits.clone()
    state.apply(logits, ctx)
    first = names.index("unlimited") * verification_rows
    assert torch.equal(logits[first : first + verification_rows], before[first : first + verification_rows])
    forced_row = names.index("req") * verification_rows
    assert logits[forced_row].argmax().item() == END
