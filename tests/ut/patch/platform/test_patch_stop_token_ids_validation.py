# SPDX-License-Identifier: Apache-2.0
"""Tests for the stop_token_ids vocabulary validation patch.

Out-of-vocab stop token ids crash the engine on CANN 9.1.x (IndexCheck
kernel traps on the OOB logits index). See vllm-ascend issue #15200.
"""

from unittest.mock import MagicMock

import pytest
from vllm.sampling_params import SamplingParams, VLLMValidationError

import vllm_ascend.patch.platform.patch_stop_token_ids_validation  # noqa: F401


def _make_model_config(vocab_size: int) -> MagicMock:
    model_config = MagicMock()
    model_config.get_vocab_size.return_value = vocab_size
    return model_config


@pytest.mark.parametrize(
    ("stop_token_ids", "vocab_size", "should_raise"),
    [
        # valid ids (boundaries included)
        ([0], 129280, False),
        ([127999, 129279], 129280, False),
        ([], 129280, False),
        (None, 129280, False),
        # out-of-vocab high / negative
        ([129280], 129280, True),
        ([151645], 129280, True),  # the id from vllm-ascend issue #15200
        ([-1], 129280, True),
        ([1, 151645, 2], 129280, True),
    ],
)
def test_stop_token_ids_vocab_validation(stop_token_ids, vocab_size, should_raise):
    model_config = _make_model_config(vocab_size)
    params = SamplingParams(stop_token_ids=stop_token_ids)
    params._all_stop_token_ids = set(stop_token_ids or ())

    if should_raise:
        with pytest.raises(VLLMValidationError, match="stop_token_ids"):
            params._validate_stop_token_ids(model_config)
    else:
        params._validate_stop_token_ids(model_config)


def test_verify_calls_validation():
    model_config = _make_model_config(129280)
    params = SamplingParams(stop_token_ids=[151645])
    params._all_stop_token_ids = {151645}

    # verify() delegates through the patched _validate_stop_token_ids; the
    # other validators are no-ops on this params object, so the first real
    # failure must be the stop_token_ids check.
    with pytest.raises(VLLMValidationError, match="out-of-vocab"):
        params.verify(model_config, None, None, None)


def test_patch_is_idempotent_when_upstream_has_fix():
    # When the bundled vLLM already ships _validate_stop_token_ids, the
    # patch must not override it.
    assert hasattr(SamplingParams, "_validate_stop_token_ids")
    model_config = _make_model_config(100)
    params = SamplingParams(stop_token_ids=[150])
    params._all_stop_token_ids = {150}
    with pytest.raises(VLLMValidationError):
        params._validate_stop_token_ids(model_config)
