# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Request-owned CPU RNG must advance across MRV2 sampling paths."""

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_ascend._310p.worker.v2 import sample_utils
from vllm_ascend._310p.worker.v2.sampler import Ascend310PSampler
from vllm_ascend._310p.worker.v2.spec_utils import probabilistic_rejection_sample_cpu

pytestmark = pytest.mark.usefixtures("sampling_kernel_overrides")


@pytest.fixture(autouse=True)
def cpu_allocations(monkeypatch):
    original_rand = torch.rand

    def rand(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", rand)
    sample_utils._CPU_GENERATOR_CACHE_310P.clear()
    yield
    sample_utils._CPU_GENERATOR_CACHE_310P.clear()


def make_sampler():
    return Ascend310PSampler(max_num_reqs=4, device="cpu", vocab_size=4)


def add(sampler, slot, seed, temperature=1.0):
    sampler.add_request(slot, SamplingParams(temperature=temperature, seed=seed))


def draw(sampler, mapping):
    generators = sampler._build_row_generators(np.asarray(mapping))
    return sample_utils._generate_request_uniforms_310p(len(mapping), generators, torch.device("cpu"))


def reference(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def test_seeded_sampling_chain_advances_every_step():
    sampler = make_sampler()
    add(sampler, 0, 42)
    expected = reference(42)
    actual = [draw(sampler, [0])[0] for _ in range(10)]
    torch.testing.assert_close(torch.stack(actual), torch.rand(10, generator=expected), rtol=0, atol=0)


def test_reordering_and_temporary_absence_preserve_request_state():
    sampler = make_sampler()
    add(sampler, 0, 11)
    add(sampler, 2, 22)
    expected = {0: reference(11), 2: reference(22)}
    for mapping in ([0, 2], [2, 0], [2], [2, 0]):
        actual = draw(sampler, mapping)
        for row, slot in enumerate(mapping):
            assert actual[row].item() == torch.rand((), generator=expected[slot]).item()


def test_cpu_generator_is_reused_including_external_advancement():
    source = reference(42)
    prepared = sample_utils._prepare_cpu_generators_310p({0: source})
    assert prepared[0] is source
    torch.rand(7, generator=source)
    expected = reference(42)
    torch.rand(7, generator=expected)
    actual = sample_utils._generate_request_uniforms_310p(1, {0: source}, torch.device("cpu"))
    assert actual[0].item() == torch.rand((), generator=expected).item()


def test_slot_reuse_creates_new_request_rng():
    sampler = make_sampler()
    add(sampler, 2, 11)
    previous = sampler._source_generators[2]
    draw(sampler, [2])
    add(sampler, 2, 22)
    assert sampler._source_generators[2] is not previous
    assert draw(sampler, [2])[0].item() == torch.rand((), generator=reference(22)).item()


def test_greedy_and_unseeded_rows_do_not_advance_dedicated_rng():
    sampler = make_sampler()
    add(sampler, 0, 11, temperature=0.0)
    add(sampler, 1, None)
    add(sampler, 2, 22)
    before = sampler._source_generators[0].get_state().clone()
    assert set(sampler._build_row_generators(np.array([2, 0, 1, -1]))) == {0}
    draw(sampler, [2, 0, 1, -1])
    assert torch.equal(before, sampler._source_generators[0].get_state())


def test_duplicate_generator_reference_consumes_one_sequence():
    source = reference(42)
    actual = sample_utils._generate_request_uniforms_310p(2, {0: source, 1: source}, torch.device("cpu"))
    torch.testing.assert_close(actual, torch.rand(2, generator=reference(42)), rtol=0, atol=0)


def test_baseline_and_mtp_rejection_share_request_rng_state():
    sampler = make_sampler()
    add(sampler, 0, 42)
    expected = reference(42)
    assert draw(sampler, [0])[0].item() == torch.rand((), generator=expected).item()
    probabilistic_rejection_sample_cpu(
        target_logits=torch.zeros(2, 4),
        draft_sampled_cpu=torch.tensor([-1, 0], dtype=torch.int32),
        cu_num_logits_np=np.array([0, 2]),
        num_speculative_steps=1,
        temperature_np=np.array([1.0]),
        idx_mapping_np=np.array([0]),
        source_generators=sampler._source_generators,
    )
    # Second seed-42 draw exceeds p(draft)=0.25: rejection + recovery use two draws.
    torch.rand(2, generator=expected)
    assert draw(sampler, [0])[0].item() == torch.rand((), generator=expected).item()
