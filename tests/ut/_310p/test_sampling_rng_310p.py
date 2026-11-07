# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Request-owned RNG and shared MTP sampling contracts."""

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_ascend._310p.worker.v2 import categorical
from vllm_ascend._310p.worker.v2.sampler import Ascend310PSampler
from vllm_ascend._310p.worker.v2.spec_utils import _sample_from_probs_row, probabilistic_rejection_sample_cpu

pytestmark = pytest.mark.usefixtures("sampling_kernel_overrides")


def reference(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


@pytest.fixture(autouse=True)
def cpu_record_stream(monkeypatch):
    # CPU tensors do not accept NPU streams; exercise RNG and tensor math here.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)


def test_mtp_reuses_shared_sampler_without_mutating_probability_view():
    source, expected = reference(42), reference(42)
    batch = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    before = batch.clone()
    for row in [0, 1, 0]:
        noise = torch.empty_like(batch[row]).exponential_(generator=expected)
        token = _sample_from_probs_row(batch[row], source)
        assert token == int((batch[row] / noise).argmax())
        assert torch.equal(source.get_state(), expected.get_state())
    torch.testing.assert_close(batch, before, rtol=0, atol=0)


def test_mtp_acceptance_and_bonus_advance_request_rng():
    source, expected = reference(42), reference(42)
    _, _, sampled, counts = probabilistic_rejection_sample_cpu(
        target_logits=torch.tensor([[0.0, -torch.inf, -torch.inf, -torch.inf], [0.0] * 4]),
        draft_sampled_cpu=torch.tensor([-1, 0], dtype=torch.int32),
        cu_num_logits_np=np.array([0, 2]),
        num_speculative_steps=1,
        temperature_np=np.array([1.0]),
        idx_mapping_np=np.array([0]),
        source_generators={0: source},
    )
    torch.rand((1,), generator=expected)
    bonus = (1 / torch.empty(4).exponential_(generator=expected)).argmax()
    assert sampled.tolist() == [[0, int(bonus)]]
    assert counts.tolist() == [2]
    assert torch.equal(source.get_state(), expected.get_state())


def test_row_generators_keep_request_identity_and_skip_greedy():
    sampler = Ascend310PSampler(max_num_reqs=4, device="cpu", vocab_size=4)
    for slot, seed, temperature in [(0, 11, 0.0), (1, None, 1.0), (2, 22, 1.0)]:
        sampler.add_request(slot, SamplingParams(temperature=temperature, seed=seed))
    source = sampler._source_generators[2]
    assert sampler._build_row_generators(np.array([2, 0, 1, -1, 2])) == {0: source, 4: source}
    assert sampler._build_row_generators(np.array([0, 1])) == {}
    assert sampler._build_row_generators(np.array([2])) == {0: source}
    sampler.add_request(2, SamplingParams(temperature=1.0, seed=33))
    assert sampler._source_generators[2] is not source


@pytest.mark.parametrize("tiled", [False, True])
def test_categorical_handles_degenerate_rows_and_partial_tile(monkeypatch, tiled):
    if tiled:
        monkeypatch.setattr(categorical, "_TILED_MIN_ELEMENTS", 0)
    probs = torch.zeros((4, 129))
    probs[1, 2] = 0.25
    probs[1, 128] = 0.75
    probs[2].fill_(torch.nan)
    uniforms = torch.full((4, 1), 0.5)
    assert categorical.sample_from_uniform_310p(probs, uniforms).tolist() == [0, 128, 0, 0]


class OffsetGenerator:
    def __init__(self, seed, offset=0):
        self.seed = seed
        self.offset = offset

    def initial_seed(self):
        return self.seed

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.offset = offset


def test_categorical_reuses_blocks_and_replays_restored_offset(monkeypatch):
    # CPU torch.Generator can generate the blocks but has no offset API.
    # CPU CI also does not provide a pinned-memory allocator.
    make_tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda data, *, pin_memory=False, **kw: make_tensor(data, **kw))
    device = torch.device("cpu")
    sampler = categorical.CategoricalSampler310()
    source = OffsetGenerator(42, (categorical._UNIFORM_BLOCK_SIZE - 1) * 4)
    offset = source.get_offset()
    expected = sampler._draw(2, device, {0: source, 1: source})
    assert source.get_offset() == offset + 8
    blocks = sampler._blocks.copy()
    assert set(blocks) == {(42, 0), (42, 1)}

    source.set_offset(offset)
    replayed = sampler._draw(2, device, {0: source, 1: source})
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
    assert all(sampler._blocks[key] is block for key, block in blocks.items())
    torch.testing.assert_close(
        categorical.CategoricalSampler310()._draw(
            2, device, {0: OffsetGenerator(42, offset), 1: OffsetGenerator(42, offset + 4)}
        ),
        expected,
        rtol=0,
        atol=0,
    )
