from types import SimpleNamespace

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessors

from tests.ut.base import TestBase
from vllm_ascend.sample.logits_processor import ReasoningEosLogitsProcessor
from vllm_ascend.sample.reasoning_phase import (
    ReasoningPhaseStateHolder,
    ReasoningProtocolSpec,
)
from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler


class TestAscendSampler(TestBase):
    def test_init_with_raw_logprobs(self):
        sampler = AscendSampler(logprobs_mode="raw_logprobs")
        self.assertEqual(sampler.logprobs_mode, "raw_logprobs")
        self.assertTrue(hasattr(sampler, "topk_topp_sampler"))
        self.assertIsInstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)

    def test_dispatches_reasoning_eos_processor_for_bonus(self):
        processor = object.__new__(ReasoningEosLogitsProcessor)
        processor.phase_state = ReasoningPhaseStateHolder(ReasoningProtocolSpec((90,), ((91,),)))
        params = SamplingParams()
        params.update_from_generation_config({"eos_token_id": 2}, 2)
        processor.update_state(
            BatchUpdate(
                batch_size=1,
                removed=(),
                added=((0, params, [], []),),
                moved=(),
            )
        )
        metadata = SimpleNamespace(
            no_penalties=True,
            bad_words_token_ids=None,
            output_token_ids=[[]],
            allowed_token_ids_mask=None,
            logitsprocs=LogitsProcessors([processor]),
            thinking_budget_state_holder=None,
            spec_token_ids=[[90]],
        )

        logits = AscendSampler().apply_logits_processors(torch.zeros((1, 100)), metadata, predict_bonus_token=True)

        assert torch.isneginf(logits[0, 2])
