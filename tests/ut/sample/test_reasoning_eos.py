# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import BatchUpdate

from vllm_ascend.sample.logits_processor import (
    ReasoningEosLogitsProcessor,
    build_ascend_logitsprocs,
)
from vllm_ascend.sample.reasoning_phase import (
    ReasoningPhaseStateHolder,
    ReasoningPhaseTracker,
    ReasoningProtocolSpec,
)


class MockReasoningConfig:
    premature_eos_policy = "mask_in_reasoning"
    reasoning_start_token_ids = [90, 91]
    reasoning_exit_token_ids = [[92, 93], [94]]


class MockVllmConfig:
    reasoning_config = MockReasoningConfig()


def _params(*eos_token_ids: int) -> SamplingParams:
    params = SamplingParams()
    params.update_from_generation_config({"eos_token_id": list(eos_token_ids)}, eos_token_ids[0])
    return params


def _batch_update(*requests):
    return BatchUpdate(batch_size=len(requests), removed=(), added=requests, moved=())


def test_phase_tracker_handles_multitoken_markers_and_reentry():
    tracker = ReasoningPhaseTracker(ReasoningProtocolSpec((90, 91), ((92, 93), (94,))))

    tracker.extend([90])
    assert not tracker.in_reasoning
    tracker.consume(91)
    assert tracker.in_reasoning
    tracker.extend([92, 93])
    assert not tracker.in_reasoning
    tracker.extend([90, 91, 8, 94])
    assert not tracker.in_reasoning


def test_phase_holder_tracks_moves_swaps_and_async_placeholders():
    holder = ReasoningPhaseStateHolder(ReasoningProtocolSpec((90, 91), ((92, 93),)))
    async_output = [90, -1]
    holder.add_request(0, [], async_output, [2])
    holder.add_request(1, [], [], [3])
    assert holder.normal_mask_entries() == []

    async_output[-1] = 91
    holder.move_request(0, 1, swap=True)

    assert holder.normal_mask_entries() == [(1, (2,))]


def test_phase_holder_only_seeds_from_prompt_suffix():
    protocol = ReasoningProtocolSpec((90, 91), ((92, 93),))
    holder = ReasoningPhaseStateHolder(protocol)
    holder.add_request(0, [8, 90, 91], [], [2])
    holder.add_request(1, [90, 91, 8], [], [3])

    assert holder.normal_mask_entries() == [(0, (2,))]


def test_processor_masks_only_model_eos_inside_reasoning():
    processor = ReasoningEosLogitsProcessor(MockVllmConfig(), torch.device("cpu"), False)
    output_ids = [[90, 91], [8]]
    params = _params(2, 3)
    params.stop_token_ids = [7]
    processor.update_state(
        _batch_update(
            (0, params, [], output_ids[0]),
            (1, _params(2), [], output_ids[1]),
        )
    )
    logits = torch.zeros((2, 100))

    processor.apply(logits)

    assert torch.isneginf(logits[0, 2:4]).all()
    assert logits[0, 7] == 0
    assert logits[1, 2] == 0
    output_ids[0].extend([92, 93])
    logits.zero_()
    processor.apply(logits)
    assert logits[0, 2] == 0


def test_processor_masks_spec_positions_and_bonus():
    processor = ReasoningEosLogitsProcessor(MockVllmConfig(), torch.device("cpu"), False)
    processor.update_state(
        _batch_update(
            (0, _params(2), [], [90, 91]),
            (1, _params(3), [], []),
        )
    )
    drafts = [[8, 92, 93, 9], [90, 91, 10]]
    logits = torch.zeros((7, 100))

    processor.apply_with_spec_decode(logits, drafts, [4, 3])

    assert torch.isneginf(logits[:3, 2]).all()
    assert logits[3, 2] == 0
    assert logits[4, 3] == 0
    assert logits[5, 3] == 0
    assert torch.isneginf(logits[6, 3])

    bonus_logits = torch.zeros((2, 100))
    processor.apply_for_bonus(bonus_logits, drafts)
    assert bonus_logits[0, 2] == 0
    assert torch.isneginf(bonus_logits[1, 3])


def test_processor_fails_open_if_all_logits_would_be_masked():
    processor = ReasoningEosLogitsProcessor(MockVllmConfig(), torch.device("cpu"), False)
    processor.update_state(_batch_update((0, _params(2), [], [90, 91])))
    logits = torch.full((1, 100), -torch.inf)
    logits[0, 2] = 5

    processor.apply(logits)

    assert logits[0, 2] == 5


def test_builder_adds_plugin_processor_for_normal_and_spec_decode():
    config = VllmConfig()
    config.reasoning_config = MockReasoningConfig()

    processors = build_ascend_logitsprocs(config, torch.device("cpu"), False, is_pooling_model=False)
    assert sum(isinstance(processor, ReasoningEosLogitsProcessor) for processor in processors.all) == 1

    config.speculative_config = SimpleNamespace()
    processors = build_ascend_logitsprocs(config, torch.device("cpu"), False, is_pooling_model=False)
    assert sum(isinstance(processor, ReasoningEosLogitsProcessor) for processor in processors.all) == 1


def test_builder_preserves_default_allow_policy():
    config = VllmConfig()
    config.reasoning_config = SimpleNamespace(premature_eos_policy="allow")

    processors = build_ascend_logitsprocs(config, torch.device("cpu"), False, is_pooling_model=False)

    assert not any(isinstance(processor, ReasoningEosLogitsProcessor) for processor in processors.all)
