# SPDX-License-Identifier: Apache-2.0
"""The reference may force a known input token, never any generated answer."""

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.pull_request.pd_reference import InputSuffix, InputSuffixProcessor


def test_only_known_input_suffix_is_forced():
    processor = InputSuffix(2)
    logits = torch.tensor([4.0, 1.0, -3.0, 2.0])
    result = processor([], logits)
    assert result.argmax().item() == 2
    assert torch.isfinite(result).sum().item() == 1
    for generated in ([2], [2, 0], [2, 0, 3]):
        answer_logits = torch.tensor([4.0, 1.0, -3.0, 2.0])
        expected = answer_logits.clone()
        assert processor(generated, answer_logits) is answer_logits
        assert torch.equal(answer_logits, expected)


@pytest.mark.parametrize("token", [-1, 1.5, True])
def test_invalid_suffix_is_rejected(token):
    with pytest.raises(ValueError):
        InputSuffixProcessor.validate_params(SamplingParams(extra_args={"pd_input_suffix": token}))


def test_other_requests_have_no_processor():
    processor = InputSuffixProcessor.__new__(InputSuffixProcessor)
    assert processor.new_req_logits_processor(SamplingParams()) is None
    assert isinstance(
        processor.new_req_logits_processor(SamplingParams(extra_args={"pd_input_suffix": 0})), InputSuffix
    )
