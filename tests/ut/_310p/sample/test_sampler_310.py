import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "vllm" not in sys.modules:
    vllm_module = ModuleType("vllm")
    vllm_envs_module = ModuleType("vllm.envs")
    vllm_envs_module.VLLM_BATCH_INVARIANT = False  # type: ignore[attr-defined]
    vllm_module.envs = vllm_envs_module  # type: ignore[attr-defined]
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.envs"] = vllm_envs_module

if "vllm_ascend.sample.sampler" not in sys.modules:
    sample_sampler_module = ModuleType("vllm_ascend.sample.sampler")
    sample_sampler_module.DEFAULT_LOGPROBS_MODE = "raw_logprobs"  # type: ignore[attr-defined]
    sample_sampler_module.AscendSampler = type("AscendSampler", (), {})  # type: ignore[attr-defined]
    sample_sampler_module.AscendTopKTopPSampler = type(  # type: ignore[attr-defined]
        "AscendTopKTopPSampler", (), {}
    )
    sys.modules["vllm_ascend.sample.sampler"] = sample_sampler_module

if "vllm_ascend.utils" not in sys.modules:
    utils_module = ModuleType("vllm_ascend.utils")
    utils_module.global_stream = lambda: MagicMock()  # type: ignore[attr-defined]
    utils_module.npu_stream_switch = lambda _: nullcontext()  # type: ignore[attr-defined]
    sys.modules["vllm_ascend.utils"] = utils_module

from vllm_ascend._310p.sample import sampler as sampler_310p  # noqa: E402


class _SourceGenerator:
    def __init__(self, seed: int):
        self.seed = seed

    def initial_seed(self) -> int:
        return self.seed


class _RecordingStreamContext:
    def __init__(self, events: list[str]):
        self.events = events

    def __enter__(self):
        self.events.append("enter_global")

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append("exit_global")


class TestSampler310pStandalone(unittest.TestCase):
    def tearDown(self):
        sampler_310p._CPU_GENERATOR_CACHE_310P.clear()

    def test_prepare_cpu_generators_preserves_requests_across_reordering(self):
        source_a = _SourceGenerator(11)
        source_b = _SourceGenerator(22)

        first = sampler_310p._prepare_cpu_generators_310p({0: source_a, 1: source_b})
        second = sampler_310p._prepare_cpu_generators_310p({0: source_b, 1: source_a})

        self.assertIs(second[1], first[0])
        self.assertIs(second[0], first[1])
        self.assertIs(sampler_310p._CPU_GENERATOR_CACHE_310P[0][0], source_b)
        self.assertIs(sampler_310p._CPU_GENERATOR_CACHE_310P[1][0], source_a)

    def test_prepare_cpu_generators_replaces_changed_source(self):
        source_first = _SourceGenerator(11)
        source_second = _SourceGenerator(22)

        first = sampler_310p._prepare_cpu_generators_310p({0: source_first})
        second = sampler_310p._prepare_cpu_generators_310p({0: source_second})

        self.assertIsNot(second[0], first[0])
        expected = torch.Generator(device="cpu")
        expected.manual_seed(22)
        self.assertEqual(
            torch.rand((), generator=second[0]).item(),
            torch.rand((), generator=expected).item(),
        )

    def test_sample_from_cdf_handles_zero_weight_prefix_and_boundaries(self):
        weights = torch.tensor(
            [
                [0.0, 2.0, 3.0, 5.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
            ]
        )
        uniforms = torch.tensor([0.2, 0.999, torch.finfo(torch.float32).tiny])

        sampled = sampler_310p._sample_from_cdf_310p(weights, uniforms)

        self.assertTrue(torch.equal(sampled, torch.tensor([2, 0, 3])))

    def test_fill_exponential_honors_active_mask(self):
        source_active = _SourceGenerator(31)
        source_inactive = _SourceGenerator(41)
        prepared = sampler_310p._prepare_cpu_generators_310p({0: source_active, 1: source_inactive})
        active_state = prepared[0].get_state()
        inactive_state = prepared[1].get_state()
        real_rand = torch.rand

        def rand_without_pinning(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_rand(*args, **kwargs)

        with patch.object(sampler_310p.torch, "rand", side_effect=rand_without_pinning):
            exponential = sampler_310p.fill_exponential_310p(
                torch.empty((2, 4), dtype=torch.float32),
                {0: source_active, 1: source_inactive},
                active_mask=[True, False],
            )

        cached_active = sampler_310p._CPU_GENERATOR_CACHE_310P[0][1]
        cached_inactive = sampler_310p._CPU_GENERATOR_CACHE_310P[1][1]
        self.assertFalse(torch.equal(cached_active.get_state(), active_state))
        self.assertTrue(torch.equal(cached_inactive.get_state(), inactive_state))
        self.assertEqual(exponential.shape, (2, 4))
        self.assertTrue(torch.isfinite(exponential).all())
        self.assertTrue((exponential > 0).all())

    def test_random_sample_waits_before_cdf_reads_probs(self):
        events: list[str] = []
        global_npu_stream = MagicMock()
        current_npu_stream = MagicMock()
        current_npu_stream.wait_stream.side_effect = lambda _: events.append("wait_global")
        fake_npu = ModuleType("torch.npu")
        fake_npu.current_stream = MagicMock(  # type: ignore[attr-defined]
            return_value=current_npu_stream
        )
        probs = MagicMock()
        probs.shape = (1, 4)
        probs.device = torch.device("cpu")
        uniforms = MagicMock()

        def generate_uniforms(*args, **kwargs):
            events.append("generate_uniforms")
            return uniforms

        def sample_from_cdf(actual_probs, actual_uniforms):
            events.append("sample_from_cdf")
            self.assertIs(actual_probs, probs)
            self.assertIs(actual_uniforms, uniforms)
            return torch.tensor([2])

        with (
            patch.object(
                sampler_310p,
                "npu_stream_switch",
                return_value=_RecordingStreamContext(events),
            ),
            patch.object(
                sampler_310p,
                "global_stream",
                return_value=global_npu_stream,
            ),
            patch.object(
                sampler_310p,
                "_generate_request_uniforms_310p",
                side_effect=generate_uniforms,
            ),
            patch.object(
                sampler_310p,
                "_sample_from_cdf_310p",
                side_effect=sample_from_cdf,
            ),
            patch.object(sampler_310p.torch, "npu", fake_npu, create=True),
        ):
            sampled = sampler_310p._random_sample_310p(probs, {})

        self.assertTrue(torch.equal(sampled, torch.tensor([2])))
        self.assertEqual(
            events,
            [
                "enter_global",
                "generate_uniforms",
                "exit_global",
                "wait_global",
                "sample_from_cdf",
            ],
        )
        current_npu_stream.wait_stream.assert_called_once_with(global_npu_stream)
        global_npu_stream.wait_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
