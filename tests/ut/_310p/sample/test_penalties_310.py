from unittest.mock import MagicMock, patch

import torch
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
    logits = torch.empty((1, 4))
    sampling_metadata = MagicMock(no_penalties=False, prompt_token_ids=torch.empty((1, 1), dtype=torch.int64))
    metadata = MagicMock()
    repeat_indices = torch.tensor([0])

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
            [[1]],
        )

    assert result is logits
    fallback.assert_called_once()
    triton_penalties.assert_not_called()
