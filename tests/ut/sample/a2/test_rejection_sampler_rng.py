# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import accumulate
from types import SimpleNamespace

import torch

from vllm_ascend.sample.rejection_sampler import sample_recovered_tokens

DEVICE = torch.device("npu:0")
VOCAB_SIZE = 128


def make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def expected_state_after_draw(seed: int) -> torch.Tensor:
    generator = make_generator(seed)
    torch.empty(VOCAB_SIZE, dtype=torch.float32, device=DEVICE).exponential_(generator=generator)
    torch.npu.synchronize()
    return generator.get_state()


def run_sample(
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    total_num_draft_tokens = sum(num_draft_tokens)
    cu_num_draft_tokens = torch.tensor(list(accumulate(num_draft_tokens)), dtype=torch.int32, device=DEVICE)
    draft_token_ids = torch.zeros(total_num_draft_tokens, dtype=torch.int64, device=DEVICE)
    weights = torch.arange(1, VOCAB_SIZE + 1, dtype=torch.float32, device=DEVICE)
    target_probs = (weights / weights.sum()).repeat(total_num_draft_tokens, 1)

    output = sample_recovered_tokens(
        max(max(num_draft_tokens), 1),
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        None,
        target_probs,
        SimpleNamespace(generators=generators),
        DEVICE,
    )
    torch.npu.synchronize()
    return output


def test_zero_draft_preserves_generator_state() -> None:
    generator = make_generator(101)
    state_before = generator.get_state().clone()

    output = run_sample([0], {0: generator})

    assert output.numel() == 0
    assert torch.equal(generator.get_state(), state_before)


def test_active_draft_advances_generator_state() -> None:
    seed = 202
    generator = make_generator(seed)
    state_before = generator.get_state().clone()

    run_sample([1], {0: generator})

    assert not torch.equal(generator.get_state(), state_before)
    assert torch.equal(generator.get_state(), expected_state_after_draw(seed))


def test_mixed_batch_advances_only_active_generators() -> None:
    num_draft_tokens = [0, 1, 0, 2]
    seeds = [301, 302, 303, 304]
    generators = {i: make_generator(seed) for i, seed in enumerate(seeds)}
    states_before = {i: generator.get_state().clone() for i, generator in generators.items()}

    run_sample(num_draft_tokens, generators)

    for i, num_drafts in enumerate(num_draft_tokens):
        state_after = generators[i].get_state()
        if num_drafts == 0:
            assert torch.equal(state_after, states_before[i])
        else:
            assert torch.equal(state_after, expected_state_after_draw(seeds[i]))


def test_request_generators_are_independent() -> None:
    generator_a = make_generator(401)
    generator_b = make_generator(402)
    mixed_output = run_sample([1, 2], {0: generator_a, 1: generator_b})

    control_a = make_generator(401)
    control_b = make_generator(402)
    output_a = run_sample([1], {0: control_a})
    output_b = run_sample([2], {0: control_b})

    assert torch.equal(generator_a.get_state(), control_a.get_state())
    assert torch.equal(generator_b.get_state(), control_b.get_state())
    assert torch.equal(mixed_output[:1], output_a)
    assert torch.equal(mixed_output[1:], output_b)


def test_zero_draft_round_matches_skipped_round() -> None:
    after_zero_round = make_generator(501)
    skipped_zero_round = make_generator(501)

    run_sample([0], {0: after_zero_round})
    actual = run_sample([1], {0: after_zero_round})
    control = run_sample([1], {0: skipped_zero_round})

    assert torch.equal(actual, control)
    assert torch.equal(after_zero_round.get_state(), skipped_zero_round.get_state())


def test_partial_generator_dict_honors_draft_activity() -> None:
    zero_draft_generator = make_generator(601)
    active_generator = make_generator(602)
    zero_state_before = zero_draft_generator.get_state().clone()

    run_sample([0, 1, 2], {0: zero_draft_generator, 2: active_generator})

    assert torch.equal(zero_draft_generator.get_state(), zero_state_before)
    assert torch.equal(active_generator.get_state(), expected_state_after_draw(602))


def test_without_generators_remains_seeded() -> None:
    torch.npu.manual_seed(701)
    first = run_sample([1, 2], {})
    torch.npu.manual_seed(701)
    second = run_sample([1, 2], {})

    assert torch.equal(first, second)
