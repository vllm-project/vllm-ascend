from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend._310p.sample import sampler as sampler_310p


class _FakeRow:
    def __init__(self):
        self.generators = []

    def exponential_(self, generator=None):
        self.generators.append(generator)
        return self


class _FakeQ:
    def __init__(self, batch_size):
        self.shape = (batch_size, 4)
        self.default_exponential_called = False
        self.rows = {idx: _FakeRow() for idx in range(batch_size)}

    def cpu(self):
        return self

    def npu(self):
        return self

    def exponential_(self):
        self.default_exponential_called = True
        return self

    def __getitem__(self, idx):
        return self.rows[idx]


class _FakeCPUGenerator:
    def __init__(self, device=None):
        self.device = device
        self.state = None
        self.seed = None

    def set_state(self, state):
        self.state = state

    def manual_seed(self, seed):
        self.seed = seed


def test_random_sample_310p_reuse_cpu_generator_cache():
    sampler_310p._CPU_GENERATOR_CACHE_310P.clear()
    probs = MagicMock()
    probs.div_.return_value = probs
    probs.argmax.return_value = probs
    probs.view.return_value = torch.tensor([0])

    fake_q_first = _FakeQ(batch_size=2)
    fake_q_second = _FakeQ(batch_size=2)

    npu_stream = MagicMock()
    generator = MagicMock()
    generator.get_state.return_value = b"state"
    generator.initial_seed.return_value = 7
    generators = {1: generator}

    with (
        patch.object(sampler_310p, "npu_stream_switch", return_value=nullcontext()),
        patch.object(sampler_310p, "global_stream", return_value=MagicMock()),
        patch.object(sampler_310p.torch, "empty_like", side_effect=[fake_q_first, fake_q_second]),
        patch.object(sampler_310p.torch, "Generator", side_effect=_FakeCPUGenerator) as gen_ctor,
        patch.object(sampler_310p.torch.npu, "current_stream", return_value=npu_stream),
    ):
        sampler_310p._random_sample_310p(probs, generators)
        sampler_310p._random_sample_310p(probs, generators)

    assert gen_ctor.call_count == 1
    assert 1 in sampler_310p._CPU_GENERATOR_CACHE_310P
    cached_cpu_generator = sampler_310p._CPU_GENERATOR_CACHE_310P[1]
    assert fake_q_first.rows[1].generators[0] is cached_cpu_generator
    assert fake_q_second.rows[1].generators[0] is cached_cpu_generator
    assert cached_cpu_generator.state == b"state"
    assert cached_cpu_generator.seed is None
    assert npu_stream.wait_stream.call_count == 2

    sampler_310p._CPU_GENERATOR_CACHE_310P.clear()


def test_random_sample_310p_fallback_to_initial_seed_when_set_state_failed():
    sampler_310p._CPU_GENERATOR_CACHE_310P.clear()
    probs = MagicMock()
    probs.div_.return_value = probs
    probs.argmax.return_value = probs
    probs.view.return_value = torch.tensor([1])

    fake_q = _FakeQ(batch_size=1)
    npu_stream = MagicMock()
    generator = MagicMock()
    generator.get_state.side_effect = RuntimeError("state read failed")
    generator.initial_seed.return_value = 1234
    generators = {0: generator}

    class _FailSetStateCPUGenerator(_FakeCPUGenerator):
        def set_state(self, state):
            raise RuntimeError("state set failed")

    with (
        patch.object(sampler_310p, "npu_stream_switch", return_value=nullcontext()),
        patch.object(sampler_310p, "global_stream", return_value=MagicMock()),
        patch.object(sampler_310p.torch, "empty_like", return_value=fake_q),
        patch.object(sampler_310p.torch, "Generator", side_effect=_FailSetStateCPUGenerator),
        patch.object(sampler_310p.torch.npu, "current_stream", return_value=npu_stream),
    ):
        sampler_310p._random_sample_310p(probs, generators)

    cached_cpu_generator = sampler_310p._CPU_GENERATOR_CACHE_310P[0]
    assert cached_cpu_generator.seed == 1234
    assert fake_q.rows[0].generators[0] is cached_cpu_generator
    assert npu_stream.wait_stream.call_count == 1

    sampler_310p._CPU_GENERATOR_CACHE_310P.clear()
