from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, call, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler, random_sample


class TestRandomSample(TestBase):
    def test_stream_dependencies_and_single_current_stream_query(self):
        operations = MagicMock()
        caller_stream = operations.caller_stream
        sampling_stream = operations.sampling_stream
        probabilities = operations.probabilities
        probabilities.shape = (2, 3)
        probabilities.div_.return_value = probabilities
        noise = operations.noise
        operations.current_stream.return_value = caller_stream
        operations.empty_like.return_value = noise

        @contextmanager
        def switch_stream(target_stream):
            self.assertIs(target_stream, sampling_stream)
            operations.enter_sampling_stream()
            try:
                yield
            finally:
                operations.exit_sampling_stream()

        with (
            patch(
                "vllm_ascend.sample.sampler.torch.npu.current_stream", side_effect=operations.current_stream
            ) as query_current_stream,
            patch("vllm_ascend.sample.sampler.global_stream", return_value=sampling_stream),
            patch("vllm_ascend.sample.sampler.npu_stream_switch", side_effect=switch_stream),
            patch("vllm_ascend.sample.sampler.torch.empty_like", side_effect=operations.empty_like),
        ):
            result = random_sample(probabilities, {})

        query_current_stream.assert_called_once_with()
        self.assertEqual(
            operations.mock_calls,
            [
                call.current_stream(),
                call.enter_sampling_stream(),
                call.sampling_stream.wait_stream(caller_stream),
                call.empty_like(probabilities),
                call.noise.exponential_(),
                call.exit_sampling_stream(),
                call.caller_stream.wait_stream(sampling_stream),
                call.probabilities.div_(noise),
                call.probabilities.argmax(dim=-1),
                call.probabilities.argmax().view(-1),
            ],
        )
        self.assertIs(result, probabilities.argmax.return_value.view.return_value)

    def assert_sampling_matches_reference(self, generator_seeds):
        probabilities = torch.tensor([[0.1, 0.6, 0.3], [0.7, 0.2, 0.1]], dtype=torch.float32)
        expected_noise = torch.empty_like(probabilities)
        generators = {index: torch.Generator().manual_seed(seed) for index, seed in generator_seeds.items()}
        reference_generators = {index: torch.Generator().manual_seed(seed) for index, seed in generator_seeds.items()}
        caller_stream = MagicMock()
        sampling_stream = MagicMock()

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            if len(reference_generators) != probabilities.shape[0]:
                expected_noise.exponential_()
            for index, generator in reference_generators.items():
                expected_noise[index].exponential_(generator=generator)
            expected_probabilities = probabilities / expected_noise
            expected_samples = expected_probabilities.argmax(dim=-1).view(-1)
            torch.manual_seed(1234)
            with (
                patch(
                    "vllm_ascend.sample.sampler.torch.npu.current_stream", return_value=caller_stream
                ) as query_current_stream,
                patch("vllm_ascend.sample.sampler.global_stream", return_value=sampling_stream),
                patch("vllm_ascend.sample.sampler.npu_stream_switch", return_value=nullcontext()),
            ):
                actual_samples = random_sample(probabilities, generators)

        query_current_stream.assert_called_once_with()
        sampling_stream.wait_stream.assert_called_once_with(caller_stream)
        caller_stream.wait_stream.assert_called_once_with(sampling_stream)
        torch.testing.assert_close(actual_samples, expected_samples)
        torch.testing.assert_close(probabilities, expected_probabilities)
        self.assertEqual(actual_samples.shape, (2,))
        self.assertEqual(actual_samples.dtype, torch.int64)
        for index, generator in generators.items():
            self.assertTrue(torch.equal(generator.get_state(), reference_generators[index].get_state()))

    def test_sampling_without_request_generators(self):
        self.assert_sampling_matches_reference({})

    def test_sampling_with_some_request_generators(self):
        self.assert_sampling_matches_reference({1: 42})

    def test_sampling_with_all_request_generators(self):
        self.assert_sampling_matches_reference({0: 17, 1: 42})

    def test_current_stream_is_recaptured_between_calls(self):
        caller_streams = [MagicMock(), MagicMock()]
        sampling_stream = MagicMock()
        with (
            patch(
                "vllm_ascend.sample.sampler.torch.npu.current_stream", side_effect=caller_streams
            ) as query_current_stream,
            patch("vllm_ascend.sample.sampler.global_stream", return_value=sampling_stream),
            patch("vllm_ascend.sample.sampler.npu_stream_switch", return_value=nullcontext()),
        ):
            for _ in caller_streams:
                random_sample(torch.tensor([[0.2, 0.8]], dtype=torch.float32), {})

        self.assertEqual(query_current_stream.call_count, 2)
        self.assertEqual(sampling_stream.wait_stream.call_args_list, [call(stream) for stream in caller_streams])
        for caller_stream in caller_streams:
            caller_stream.wait_stream.assert_called_once_with(sampling_stream)


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)
