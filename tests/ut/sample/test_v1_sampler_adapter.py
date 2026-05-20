import unittest
from types import SimpleNamespace

import torch
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from tests.ut.base import TestBase


def _enabled_sampling_config():
    from vllm_ascend.ascend_config import SamplingConfig

    return SamplingConfig({"enable_sampling_optimization": True})


def _metadata(max_num_logprobs=None):
    return SimpleNamespace(
        temperature=torch.ones(3, dtype=torch.float32),
        generators={},
        max_num_logprobs=max_num_logprobs,
    )


class TestV1SamplerAdapter(TestBase):
    def test_builds_decode_context_and_returns_v1_sampler_output(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=4,
            vocab_size=8,
            device=torch.device("cpu"),
            sampling_config=_enabled_sampling_config(),
        )
        logits = torch.arange(24, dtype=torch.float16).reshape(3, 8)
        processed_logits = logits.float() + 1
        sampled = torch.tensor([7, 6, 5], dtype=torch.int64)
        calls = []

        def apply_logits_processor(step_logits, sampling_metadata, ctx, num_speculative_tokens):
            calls.append("logits_processor")
            self.assertIs(step_logits, logits)
            self.assertEqual(num_speculative_tokens, 1)
            self.assertEqual(ctx.num_reqs, 3)
            self.assertFalse(ctx.expanded_logits)
            self.assertEqual(ctx.expanded_idx_mapping.tolist(), [0, 1, 2])
            self.assertEqual(ctx.pos.tolist(), [10, 11, 12])
            self.assertEqual(ctx.input_ids.tolist(), [101, 102, 103])
            return processed_logits

        def sample(step_logits, sampling_metadata, ctx):
            calls.append("sample")
            self.assertIs(step_logits, processed_logits)
            self.assertEqual(ctx.expanded_idx_mapping.tolist(), [0, 1, 2])
            return sampled

        def compute_logprobs(raw_logits, step_logits, sampled_token_ids, sampling_metadata, ctx):
            calls.append("logprobs")
            self.assertIs(raw_logits, logits)
            self.assertIs(step_logits, processed_logits)
            self.assertIs(sampled_token_ids, sampled)
            return None

        adapter._logits_processor.apply = apply_logits_processor
        adapter._sample = sample
        adapter._compute_logprobs = compute_logprobs

        output = adapter(
            logits=logits,
            sampling_metadata=_metadata(),
            num_reqs=3,
            positions=torch.tensor([10, 11, 12], dtype=torch.int64),
            input_ids=torch.tensor([101, 102, 103], dtype=torch.int64),
            req_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            req_ids=("req0", "req1", "req2"),
        )

        self.assertIsInstance(output, SamplerOutput)
        self.assertEqual(output.sampled_token_ids.shape, (3, 1))
        self.assertEqual(output.sampled_token_ids.dtype, torch.int32)
        self.assertEqual(output.sampled_token_ids.tolist(), [[7], [6], [5]])
        self.assertIsNone(output.logprobs_tensors)
        self.assertEqual(calls, ["logits_processor", "sample", "logprobs"])

    def test_preserves_logprobs_tensors_from_logprobs_stage(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=3,
            vocab_size=8,
            device=torch.device("cpu"),
            sampling_config=_enabled_sampling_config(),
        )
        logits = torch.randn(3, 8)
        sampled = torch.tensor([1, 2, 3], dtype=torch.int64)
        logprobs_tensors = LogprobsTensors(
            logprob_token_ids=torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.int32),
            logprobs=torch.tensor([[-0.1, -1.0], [-0.2, -2.0], [-0.3, -3.0]], dtype=torch.float32),
            selected_token_ranks=torch.tensor([0, 1, 2], dtype=torch.int32),
        )

        adapter._logits_processor.apply = lambda step_logits, *_args: step_logits.float()
        adapter._sample = lambda *_args: sampled
        adapter._compute_logprobs = lambda *_args: logprobs_tensors

        output = adapter(
            logits=logits,
            sampling_metadata=_metadata(max_num_logprobs=1),
            num_reqs=3,
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            input_ids=torch.tensor([10, 11, 12], dtype=torch.int64),
            req_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            req_ids=("req0", "req1", "req2"),
        )

        self.assertEqual(output.sampled_token_ids.tolist(), [[1], [2], [3]])
        self.assertIs(output.logprobs_tensors, logprobs_tensors)

    def test_respects_explicit_generator_seeds(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter

        adapter = V1SamplerAdapter(
            max_num_reqs=3,
            vocab_size=8,
            device=torch.device("cpu"),
            sampling_config=_enabled_sampling_config(),
        )
        seeded_generator = torch.Generator(device="cpu")
        seeded_generator.manual_seed(12345)

        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        ctx = V1MappingContext.from_v1_logits(
            num_reqs=3,
            positions_at_logits=torch.tensor([0, 1, 2], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10, 11, 12], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 1, 2], dtype=torch.int32),
            device=torch.device("cpu"),
            req_ids=("req0", "req1", "req2"),
        )
        metadata = SimpleNamespace(generators={1: seeded_generator})
        seeds = adapter._compute_seeds(
            sampling_metadata=metadata,
            ctx=ctx,
        )
        second_seeds = adapter._compute_seeds(
            sampling_metadata=metadata,
            ctx=ctx,
        )

        self.assertEqual(seeds.shape, (3,))
        self.assertEqual(seeds.dtype, torch.int64)
        self.assertEqual(seeds.device.type, "cpu")
        self.assertEqual(seeds[1].item(), 12345)
        torch.testing.assert_close(second_seeds, seeds)

    def test_seed_cache_follows_request_id_when_slot_changes(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        adapter = V1SamplerAdapter(
            max_num_reqs=3,
            vocab_size=8,
            device=torch.device("cpu"),
            sampling_config=_enabled_sampling_config(),
        )
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=3,
            positions_at_logits=torch.tensor([0, 1, 2], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10, 11, 12], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 1, 2], dtype=torch.int32),
            device=torch.device("cpu"),
            req_ids=("req0", "req1", "req2"),
        )
        seeds = adapter._compute_seeds(SimpleNamespace(generators={}), ctx)
        moved_ctx = V1MappingContext.from_v1_logits(
            num_reqs=3,
            positions_at_logits=torch.tensor([3, 4, 5], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([13, 14, 15], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 1, 2], dtype=torch.int32),
            device=torch.device("cpu"),
            req_ids=("req2", "req0", "req1"),
        )

        moved_seeds = adapter._compute_seeds(SimpleNamespace(generators={}), moved_ctx)

        self.assertEqual(moved_seeds[0].item(), seeds[2].item())
        self.assertEqual(moved_seeds[1].item(), seeds[0].item())
        self.assertEqual(moved_seeds[2].item(), seeds[1].item())

    def test_temperature_none_maps_to_greedy_gumbel_temperature(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            vocab_size=8,
            device=torch.device("cpu"),
            sampling_config=_enabled_sampling_config(),
        )
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=2,
            positions_at_logits=torch.tensor([0, 1], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10, 11], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 1], dtype=torch.int32),
            device=torch.device("cpu"),
            req_ids=("req0", "req1"),
        )

        temp = adapter._temperature_for_sampling(SimpleNamespace(temperature=None), ctx)

        self.assertEqual(temp.tolist(), [0.0, 0.0])
        self.assertEqual(temp.dtype, torch.float32)

    def test_formats_expanded_logits_without_assuming_one_row_per_request(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            vocab_size=8,
            device=torch.device("cpu"),
            sampling_config=_enabled_sampling_config(),
        )
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=2,
            positions_at_logits=torch.tensor([0, 1, 0], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10, 11, 20], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 0, 1], dtype=torch.int32),
            device=torch.device("cpu"),
            req_ids=("req0", "req1"),
        )

        sampled = adapter._format_sampled_token_ids(torch.tensor([5, 6, 7], dtype=torch.int32), ctx)

        self.assertEqual(sampled.tolist(), [[5, 6], [7, -1]])

    def test_compute_logprobs_returns_none_when_not_requested(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            vocab_size=4,
            device=torch.device("cpu"),
            sampling_config=_enabled_sampling_config(),
        )
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=2,
            positions_at_logits=torch.tensor([0, 1], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10, 11], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 1], dtype=torch.int32),
            device=torch.device("cpu"),
        )

        logprobs = adapter._compute_logprobs(
            raw_logits=torch.randn(2, 4),
            processed_logits=torch.randn(2, 4),
            sampled=torch.tensor([1, 2], dtype=torch.int64),
            sampling_metadata=_metadata(max_num_logprobs=None),
            ctx=ctx,
        )

        self.assertIsNone(logprobs)

    def test_compute_logprobs_zero_returns_sampled_token_only(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            vocab_size=4,
            device=torch.device("cpu"),
            logprobs_mode="raw_logprobs",
            sampling_config=_enabled_sampling_config(),
        )
        raw_logits = torch.tensor(
            [
                [0.0, 2.0, 1.0, -1.0],
                [3.0, 1.0, 0.0, -2.0],
            ],
            dtype=torch.float32,
        )
        processed_logits = raw_logits + 10.0
        sampled = torch.tensor([1, 0], dtype=torch.int64)
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=2,
            positions_at_logits=torch.tensor([0, 1], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10, 11], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 1], dtype=torch.int32),
            device=torch.device("cpu"),
        )

        logprobs = adapter._compute_logprobs(
            raw_logits,
            processed_logits,
            sampled,
            _metadata(max_num_logprobs=0),
            ctx,
        )

        expected_logprobs = raw_logits.log_softmax(dim=-1).gather(-1, sampled.unsqueeze(-1))
        self.assertEqual(logprobs.logprob_token_ids.tolist(), [[1], [0]])
        torch.testing.assert_close(logprobs.logprobs, expected_logprobs)
        self.assertEqual(logprobs.selected_token_ranks.tolist(), [1, 1])
        self.assertIsNone(logprobs.cu_num_generated_tokens)

    def test_compute_logprobs_topk_can_use_processed_logits(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        adapter = V1SamplerAdapter(
            max_num_reqs=1,
            vocab_size=4,
            device=torch.device("cpu"),
            logprobs_mode="processed_logprobs",
            sampling_config=_enabled_sampling_config(),
        )
        raw_logits = torch.zeros((1, 4), dtype=torch.float32)
        processed_logits = torch.tensor([[0.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
        sampled = torch.tensor([3], dtype=torch.int64)
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=1,
            positions_at_logits=torch.tensor([0], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0], dtype=torch.int32),
            device=torch.device("cpu"),
        )

        logprobs = adapter._compute_logprobs(
            raw_logits,
            processed_logits,
            sampled,
            _metadata(max_num_logprobs=2),
            ctx,
        )

        expected_token_ids = torch.tensor([[3, 1, 2]], dtype=torch.int32)
        expected_logprobs = processed_logits.log_softmax(dim=-1)[:, [3, 1, 2]]
        torch.testing.assert_close(logprobs.logprob_token_ids, expected_token_ids)
        torch.testing.assert_close(logprobs.logprobs, expected_logprobs)
        self.assertEqual(logprobs.selected_token_ranks.tolist(), [3])

    def test_compute_logprobs_full_vocab_preserves_expanded_cu_num_logits(self):
        from vllm_ascend.worker.v1.sample.adapter import V1SamplerAdapter
        from vllm_ascend.worker.v1.sample.context import V1MappingContext

        adapter = V1SamplerAdapter(
            max_num_reqs=2,
            vocab_size=4,
            device=torch.device("cpu"),
            sampling_config=_enabled_sampling_config(),
        )
        logits = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0, 0.0],
                [1.0, 3.0, 0.0, 2.0],
            ],
            dtype=torch.float32,
        )
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=2,
            positions_at_logits=torch.tensor([0, 1, 0], dtype=torch.int64),
            input_ids_at_logits=torch.tensor([10, 11, 20], dtype=torch.int64),
            req_indices_at_logits=torch.tensor([0, 0, 1], dtype=torch.int32),
            device=torch.device("cpu"),
        )

        logprobs = adapter._compute_logprobs(
            raw_logits=logits,
            processed_logits=logits + 1.0,
            sampled=torch.tensor([3, 0, 1], dtype=torch.int64),
            sampling_metadata=_metadata(max_num_logprobs=-1),
            ctx=ctx,
        )

        self.assertEqual(logprobs.logprob_token_ids.numel(), 0)
        self.assertEqual(logprobs.selected_token_ranks.numel(), 0)
        torch.testing.assert_close(logprobs.logprobs, logits.log_softmax(dim=-1))
        self.assertEqual(logprobs.cu_num_generated_tokens, [0, 2, 3])


if __name__ == "__main__":
    unittest.main()
