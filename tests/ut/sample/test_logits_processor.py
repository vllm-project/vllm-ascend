import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase


def _ctx(num_reqs=2):
    from vllm_ascend.worker.v1.sample.context import V1MappingContext

    return V1MappingContext.from_v1_logits(
        num_reqs=num_reqs,
        positions_at_logits=torch.arange(num_reqs, dtype=torch.int64),
        input_ids_at_logits=torch.arange(num_reqs, dtype=torch.int64),
        req_indices_at_logits=torch.arange(num_reqs, dtype=torch.int32),
        device=torch.device("cpu"),
    )


def _expanded_ctx():
    from vllm_ascend.worker.v1.sample.context import V1MappingContext

    return V1MappingContext.from_v1_logits(
        num_reqs=2,
        positions_at_logits=torch.tensor([0, 1, 0], dtype=torch.int64),
        input_ids_at_logits=torch.tensor([10, 11, 20], dtype=torch.int64),
        req_indices_at_logits=torch.tensor([0, 0, 1], dtype=torch.int32),
        device=torch.device("cpu"),
    )


def _metadata(**overrides):
    values = {
        "no_penalties": True,
        "bad_words_token_ids": {},
        "logitsprocs": SimpleNamespace(argmax_invariant=[], non_argmax_invariant=[]),
        "allowed_token_ids_mask": None,
        "min_p": torch.zeros(2, dtype=torch.float32),
        "temperature": torch.ones(2, dtype=torch.float32),
        "top_k": None,
        "top_p": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestLogitsProcessor(TestBase):
    def test_default_runs_full_pipeline_in_order(self):
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        processor = LogitsProcessor("default")
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
        raw_logits = logits.clone()
        processed = torch.empty_like(logits, dtype=torch.float32)
        calls = []

        def expect_working_copy(stage_logits):
            self.assertIsNot(stage_logits, logits)
            self.assertEqual(stage_logits.dtype, torch.float32)
            torch.testing.assert_close(stage_logits, logits.float())

        def allowed(stage_logits, sampling_metadata, ctx):
            expect_working_copy(stage_logits)
            calls.append("allowed")

        def bad_words(stage_logits, sampling_metadata, ctx):
            expect_working_copy(stage_logits)
            calls.append("bad_words")

        def non_argmax(stage_logits, sampling_metadata, ctx):
            expect_working_copy(stage_logits)
            calls.append("non_argmax")
            return stage_logits

        def penalties(stage_logits, sampling_metadata, ctx, num_speculative_tokens):
            expect_working_copy(stage_logits)
            self.assertEqual(num_speculative_tokens, 1)
            calls.append("penalties")

        def temperature(stage_logits, sampling_metadata, ctx):
            expect_working_copy(stage_logits)
            calls.append("temperature")

        def argmax(stage_logits, sampling_metadata, ctx):
            expect_working_copy(stage_logits)
            calls.append("argmax")
            return stage_logits

        def top_k_top_p(stage_logits, sampling_metadata, ctx):
            expect_working_copy(stage_logits)
            calls.append("top_k_top_p")
            processed.copy_(stage_logits)
            return processed

        processor._apply_allowed_token_ids = allowed
        processor._apply_bad_words = bad_words
        processor._apply_non_argmax_invariant = non_argmax
        processor._apply_penalties = penalties
        processor._apply_temperature = temperature
        processor._apply_argmax_invariant = argmax
        processor._apply_top_k_top_p = top_k_top_p

        output = processor.apply(logits, _metadata(), _ctx(), num_speculative_tokens=1)

        self.assertIs(output, processed)
        torch.testing.assert_close(logits, raw_logits)
        self.assertEqual(
            calls,
            [
                "allowed",
                "non_argmax",
                "penalties",
                "bad_words",
                "temperature",
                "argmax",
                "top_k_top_p",
            ],
        )

    def test_default_uses_upstream_sampling_ops_without_bridge_state(self):
        from vllm_ascend.worker.v1.sample import logits_processor as logits_processor_module
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        class MinPProcessor:
            min_p_count = 1

            @staticmethod
            def get_min_p_by_index(req_idx):
                return [0.0, 0.2][req_idx]

        processor = LogitsProcessor("default")
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float16)
        processed = torch.empty_like(logits, dtype=torch.float32)
        ctx = _expanded_ctx()
        metadata = _metadata(
            no_penalties=False,
            prompt_token_ids=torch.tensor([[1], [2]], dtype=torch.int64),
            presence_penalties=torch.tensor([0.0, 0.1], dtype=torch.float32),
            frequency_penalties=torch.tensor([0.0, 0.2], dtype=torch.float32),
            repetition_penalties=torch.tensor([1.0, 1.1], dtype=torch.float32),
            output_token_ids=[[1], [2]],
            bad_words_token_ids={1: [[3, 4]]},
            temperature=torch.tensor([1.0, 0.5], dtype=torch.float32),
            top_k=torch.tensor([2, 2], dtype=torch.int32),
            top_p=torch.tensor([1.0, 0.9], dtype=torch.float32),
            logitsprocs=SimpleNamespace(
                argmax_invariant=[MinPProcessor()],
                non_argmax_invariant=[],
            ),
        )

        self.assertFalse(hasattr(processor, "_apply_sampling_params_bridge"))
        with (
            patch.object(logits_processor_module, "_apply_bad_words_op") as bad_words,
            patch.object(
                logits_processor_module,
                "_apply_penalties_op",
                side_effect=lambda *_: None,
            ) as penalties,
            patch.object(logits_processor_module, "_apply_temperature") as temperature,
            patch.object(logits_processor_module, "_apply_min_p") as min_p,
            patch.object(
                logits_processor_module,
                "_apply_top_k_top_p",
                return_value=processed,
            ) as top_k_top_p,
        ):
            output = processor.apply(logits, metadata, ctx, num_speculative_tokens=3)

        self.assertIs(output, processed)
        bad_words.assert_called_once()
        penalties.assert_called_once()
        temperature.assert_called_once()
        min_p.assert_called_once()
        top_k_top_p.assert_called_once()

    def test_skip_returns_unprocessed_logits_and_warns_once(self):
        from vllm_ascend.worker.v1.sample import logits_processor as logits_processor_module
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        warnings = []
        processor = LogitsProcessor("skip")
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
        metadata = _metadata(
            no_penalties=False,
            bad_words_token_ids={0: [[11, 12]]},
            logitsprocs=SimpleNamespace(argmax_invariant=[object()], non_argmax_invariant=[object()]),
            allowed_token_ids_mask=torch.zeros((2, 4), dtype=torch.bool),
            min_p=torch.tensor([0.0, 0.2], dtype=torch.float32),
        )

        def fail_default(*_args, **_kwargs):
            self.fail("skip mode must not fall back to default processing")

        processor._apply_default = fail_default

        with patch.object(
            logits_processor_module.logger,
            "warning",
            side_effect=lambda _message, category, _details: warnings.append(category),
        ):
            output = processor.apply(logits, metadata, _ctx(), num_speculative_tokens=1)
            second_output = processor.apply(logits, metadata, _ctx(), num_speculative_tokens=1)

        torch.testing.assert_close(output, logits.float())
        torch.testing.assert_close(second_output, logits.float())
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(
            set(warnings),
            {
                "penalties",
                "bad_words",
                "logit_bias",
                "non_argmax_invariant",
                "filtering",
            },
        )
        self.assertEqual(len(warnings), 5)

    def test_skip_without_incompatible_params_does_not_warn(self):
        from vllm_ascend.worker.v1.sample import logits_processor as logits_processor_module
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        warnings = []
        processor = LogitsProcessor("skip")

        with patch.object(
            logits_processor_module.logger,
            "warning",
            side_effect=lambda _message, category, _details: warnings.append(category),
        ):
            output = processor.apply(
                torch.ones((2, 4), dtype=torch.float16),
                _metadata(),
                _ctx(),
                num_speculative_tokens=1,
            )

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(warnings, [])

    def test_skip_reuses_fp32_logits_without_copy(self):
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        processor = LogitsProcessor("skip")
        logits = torch.ones((2, 4), dtype=torch.float32)

        output = processor.apply(logits, _metadata(), _ctx(), num_speculative_tokens=1)

        self.assertIs(output, logits)

    def test_skip_warns_when_top_k_or_top_p_filtering_would_be_skipped(self):
        from vllm_ascend.worker.v1.sample import logits_processor as logits_processor_module
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        warnings = []
        processor = LogitsProcessor("skip")
        metadata = _metadata(
            top_k=torch.tensor([8, 4], dtype=torch.int32),
            top_p=torch.tensor([1.0, 0.8], dtype=torch.float32),
        )

        with patch.object(
            logits_processor_module.logger,
            "warning",
            side_effect=lambda _message, category, _details: warnings.append(category),
        ):
            processor.apply(
                torch.ones((2, 8), dtype=torch.float16),
                metadata,
                _ctx(),
                num_speculative_tokens=1,
            )

        self.assertEqual(warnings, ["filtering"])

    def test_request_row_expansion_supports_mixed_params_for_expanded_logits(self):
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        processor = LogitsProcessor("skip")
        ctx = _expanded_ctx()

        per_request = torch.tensor([0.7, 0.2], dtype=torch.float32)
        expanded = processor._expand_request_rows(
            per_request,
            ctx,
            torch.device("cpu"),
        )
        torch.testing.assert_close(
            expanded,
            torch.tensor([0.7, 0.7, 0.2], dtype=torch.float32),
        )

        scalar = processor._expand_request_rows(
            torch.tensor(1.5),
            ctx,
            torch.device("cpu"),
        )
        torch.testing.assert_close(
            scalar,
            torch.tensor([1.5, 1.5, 1.5]),
        )

    def test_allowed_token_mask_uses_request_mapping_for_expanded_logits(self):
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        processor = LogitsProcessor("default")
        ctx = _expanded_ctx()
        logits = torch.zeros((3, 4), dtype=torch.float32)
        metadata = _metadata(
            allowed_token_ids_mask=torch.tensor(
                [
                    [False, True, False, True],
                    [True, False, True, False],
                ],
                dtype=torch.bool,
            )
        )

        processor._apply_allowed_token_ids(logits, metadata, ctx)

        self.assertTrue(torch.isneginf(logits[0, 1]))
        self.assertTrue(torch.isneginf(logits[0, 3]))
        self.assertTrue(torch.isneginf(logits[1, 1]))
        self.assertTrue(torch.isneginf(logits[1, 3]))
        self.assertTrue(torch.isneginf(logits[2, 0]))
        self.assertTrue(torch.isneginf(logits[2, 2]))
        torch.testing.assert_close(
            logits[
                torch.tensor([0, 0, 1, 1, 2, 2]),
                torch.tensor([0, 2, 0, 2, 1, 3]),
            ],
            torch.zeros(6),
        )

    def test_output_token_history_expands_by_request_for_expanded_logits(self):
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        processor = LogitsProcessor("default")
        ctx = _expanded_ctx()

        expanded = processor._expand_output_token_ids([[100], [200, 201]], ctx)

        self.assertEqual(expanded, [[100], [100, 10], [200, 201]])

    def test_fused_mode_is_valid_but_not_implemented_in_phase_1(self):
        from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

        processor = LogitsProcessor("fused")

        with self.assertRaisesRegex(NotImplementedError, "fused logits_processing_mode"):
            processor.apply(torch.ones((2, 4)), _metadata(), _ctx(), num_speculative_tokens=1)


if __name__ == "__main__":
    unittest.main()
