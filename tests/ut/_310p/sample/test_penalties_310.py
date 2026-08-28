from unittest.mock import MagicMock, patch

import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.sample import rejection_sampler as rejection_sampler_module
from vllm_ascend.sample import sampler as sampler_module
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
from vllm_ascend.sample.sampler import AscendSampler


def test_sampler_uses_vllm_penalties_on_310p_when_triton_is_available():
    logits = MagicMock()
    metadata = MagicMock(no_penalties=False, prompt_token_ids=MagicMock())
    output_token_ids = [[1, 2]]

    with (
        patch.object(sampler_module, "HAS_TRITON", True),
        patch.object(sampler_module, "is_310p", return_value=True),
        patch.object(Sampler, "apply_penalties", return_value=logits) as fallback,
        patch.object(sampler_module, "apply_all_penalties") as triton_penalties,
    ):
        result = AscendSampler.apply_penalties(logits, metadata, output_token_ids)

    assert result is logits
    fallback.assert_called_once_with(logits, metadata, output_token_ids)
    triton_penalties.assert_not_called()


def test_sampler_uses_triton_penalties_off_310p():
    logits = MagicMock()
    metadata = MagicMock(no_penalties=False, prompt_token_ids=MagicMock())
    output_token_ids = [[1, 2]]
    expected = MagicMock()

    with (
        patch.object(sampler_module, "HAS_TRITON", True),
        patch.object(sampler_module, "is_310p", return_value=False),
        patch.object(sampler_module, "apply_all_penalties", return_value=expected) as triton_penalties,
    ):
        result = AscendSampler.apply_penalties(logits, metadata, output_token_ids)

    assert result is expected
    triton_penalties.assert_called_once_with(
        logits,
        metadata.prompt_token_ids,
        metadata.presence_penalties,
        metadata.frequency_penalties,
        metadata.repetition_penalties,
        output_token_ids,
    )


def test_rejection_sampler_uses_vllm_penalties_on_310p_when_triton_is_available():
    logits = torch.empty((3, 4))
    sampling_metadata = SamplingMetadata(
        temperature=None,
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=torch.tensor([[1, 4], [2, 4]]),
        frequency_penalties=torch.tensor([0.1, 0.2]),
        presence_penalties=torch.tensor([0.3, 0.4]),
        repetition_penalties=torch.tensor([1.1, 1.2]),
        output_token_ids=[[1], [2]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=MagicMock(),
    )
    metadata = MagicMock()
    repeat_indices = torch.tensor([0, 0, 1])
    expanded_output_token_ids = [[1], [1, 3], [2]]

    with (
        patch.object(rejection_sampler_module, "HAS_TRITON", True),
        patch.object(rejection_sampler_module, "is_310p", return_value=True),
        patch.object(rejection_sampler_module.Sampler, "apply_penalties", return_value=logits) as fallback,
        patch.object(rejection_sampler_module, "apply_all_penalties") as triton_penalties,
    ):
        result = AscendRejectionSampler.apply_penalties(
            logits,
            sampling_metadata,
            metadata,
            repeat_indices,
            expanded_output_token_ids,
        )

    assert result is logits
    fallback.assert_called_once()
    fallback_logits, repeated_metadata, fallback_output_token_ids = fallback.call_args.args
    assert fallback_logits is logits
    assert repeated_metadata is not sampling_metadata
    torch.testing.assert_close(
        repeated_metadata.prompt_token_ids,
        sampling_metadata.prompt_token_ids[repeat_indices],
    )
    torch.testing.assert_close(
        repeated_metadata.presence_penalties,
        sampling_metadata.presence_penalties[repeat_indices],
    )
    torch.testing.assert_close(
        repeated_metadata.frequency_penalties,
        sampling_metadata.frequency_penalties[repeat_indices],
    )
    torch.testing.assert_close(
        repeated_metadata.repetition_penalties,
        sampling_metadata.repetition_penalties[repeat_indices],
    )
    assert fallback_output_token_ids is expanded_output_token_ids
    triton_penalties.assert_not_called()
