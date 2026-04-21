import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "vllm" not in sys.modules:
    vllm_module = ModuleType("vllm")
    vllm_module.envs = SimpleNamespace(VLLM_BATCH_INVARIANT=False)
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.envs"] = vllm_module.envs

if "vllm_ascend.sample.sampler" not in sys.modules:
    sample_sampler_module = ModuleType("vllm_ascend.sample.sampler")
    sample_sampler_module.DEFAULT_LOGPROBS_MODE = "raw_logprobs"
    sample_sampler_module.AscendSampler = type("AscendSampler", (), {})
    sample_sampler_module.AscendTopKTopPSampler = type("AscendTopKTopPSampler", (), {})
    sys.modules["vllm_ascend.sample.sampler"] = sample_sampler_module

if "vllm_ascend.utils" not in sys.modules:
    utils_module = ModuleType("vllm_ascend.utils")
    utils_module.global_stream = lambda: MagicMock()
    utils_module.npu_stream_switch = lambda _: nullcontext()
    sys.modules["vllm_ascend.utils"] = utils_module

from vllm_ascend._310p.sample import sampler as sampler_310p  # noqa: E402


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


class TestSampler310pStandalone(unittest.TestCase):
    def tearDown(self):
        sampler_310p._CPU_GENERATOR_CACHE_310P.clear()

    def test_random_sample_310p_reuse_cpu_generator_cache(self):
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
            patch.object(
                sampler_310p.torch,
                "npu",
                SimpleNamespace(current_stream=MagicMock(return_value=npu_stream)),
                create=True,
            ),
        ):
            sampler_310p._random_sample_310p(probs, generators)
            sampler_310p._random_sample_310p(probs, generators)

        self.assertEqual(gen_ctor.call_count, 1)
        self.assertIn(1, sampler_310p._CPU_GENERATOR_CACHE_310P)
        cached_cpu_generator, source_generator_id = sampler_310p._CPU_GENERATOR_CACHE_310P[1]
        self.assertIs(fake_q_first.rows[1].generators[0], cached_cpu_generator)
        self.assertIs(fake_q_second.rows[1].generators[0], cached_cpu_generator)
        self.assertEqual(source_generator_id, id(generator))
        self.assertEqual(cached_cpu_generator.state, b"state")
        self.assertIsNone(cached_cpu_generator.seed)
        self.assertEqual(npu_stream.wait_stream.call_count, 2)

    def test_random_sample_310p_fallback_to_initial_seed_when_set_state_failed(self):
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
            patch.object(
                sampler_310p.torch,
                "npu",
                SimpleNamespace(current_stream=MagicMock(return_value=npu_stream)),
                create=True,
            ),
        ):
            sampler_310p._random_sample_310p(probs, generators)

        cached_cpu_generator, source_generator_id = sampler_310p._CPU_GENERATOR_CACHE_310P[0]
        self.assertEqual(source_generator_id, id(generator))
        self.assertEqual(cached_cpu_generator.seed, 1234)
        self.assertIs(fake_q.rows[0].generators[0], cached_cpu_generator)
        self.assertEqual(npu_stream.wait_stream.call_count, 1)

    def test_random_sample_310p_rebuild_cache_when_generator_identity_changes(self):
        sampler_310p._CPU_GENERATOR_CACHE_310P.clear()
        probs = MagicMock()
        probs.div_.return_value = probs
        probs.argmax.return_value = probs
        probs.view.return_value = torch.tensor([0])

        fake_q_first = _FakeQ(batch_size=1)
        fake_q_second = _FakeQ(batch_size=1)
        npu_stream = MagicMock()

        generator_first = MagicMock()
        generator_first.get_state.return_value = b"state-1"
        generator_first.initial_seed.return_value = 11

        generator_second = MagicMock()
        generator_second.get_state.return_value = b"state-2"
        generator_second.initial_seed.return_value = 22

        with (
            patch.object(sampler_310p, "npu_stream_switch", return_value=nullcontext()),
            patch.object(sampler_310p, "global_stream", return_value=MagicMock()),
            patch.object(sampler_310p.torch, "empty_like", side_effect=[fake_q_first, fake_q_second]),
            patch.object(sampler_310p.torch, "Generator", side_effect=_FakeCPUGenerator) as gen_ctor,
            patch.object(
                sampler_310p.torch,
                "npu",
                SimpleNamespace(current_stream=MagicMock(return_value=npu_stream)),
                create=True,
            ),
        ):
            sampler_310p._random_sample_310p(probs, {0: generator_first})
            sampler_310p._random_sample_310p(probs, {0: generator_second})

        self.assertEqual(gen_ctor.call_count, 2)
        first_cpu_generator = fake_q_first.rows[0].generators[0]
        second_cpu_generator = fake_q_second.rows[0].generators[0]
        self.assertIsNot(first_cpu_generator, second_cpu_generator)
        self.assertEqual(first_cpu_generator.state, b"state-1")
        self.assertEqual(second_cpu_generator.state, b"state-2")
        cached_cpu_generator, source_generator_id = sampler_310p._CPU_GENERATOR_CACHE_310P[0]
        self.assertIs(cached_cpu_generator, second_cpu_generator)
        self.assertEqual(source_generator_id, id(generator_second))


if __name__ == "__main__":
    unittest.main()
