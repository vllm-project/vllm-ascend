# SPDX-License-Identifier: Apache-2.0
"""Tests for the vocabulary validation patch (stop_token_ids + allowed_token_ids).

Out-of-vocab stop token ids crash the engine on CANN 9.1.x (IndexCheck
kernel traps on the OOB logits index). allowed_token_ids has the same
hazard via the InputBatch mask. See vllm-ascend issue #15200 and the
upstream fix vllm#54196.
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


@pytest.mark.parametrize(
    ("allowed_token_ids", "vocab_size", "should_raise"),
    [
        # valid ids (boundaries included)
        ([0], 129280, False),
        ([127999, 129279], 129280, False),
        (None, 129280, False),
        # empty / out-of-vocab high / negative
        ([], 129280, True),
        ([129280], 129280, True),
        ([151645], 129280, True),
        ([-1], 129280, True),
        ([1, 151645, 2], 129280, True),
        # id within a larger tokenizer vocab but past the model vocab
        ([1500], 1000, True),
    ],
)
def test_allowed_token_ids_vocab_validation(allowed_token_ids, vocab_size, should_raise):
    model_config = _make_model_config(vocab_size)
    params = SamplingParams(allowed_token_ids=allowed_token_ids)

    if should_raise:
        with pytest.raises(VLLMValidationError, match="allowed_token_ids"):
            params._validate_allowed_token_ids(model_config)
    else:
        params._validate_allowed_token_ids(model_config)


@pytest.mark.parametrize(
    ("stop_token_ids", "allowed_token_ids", "match"),
    [
        ([151645], None, "stop_token_ids"),
        (None, [151645], "allowed_token_ids"),
    ],
)
def test_verify_calls_validation(stop_token_ids, allowed_token_ids, match):
    model_config = _make_model_config(129280)
    params = SamplingParams(stop_token_ids=stop_token_ids, allowed_token_ids=allowed_token_ids)

    # verify() delegates through the patched validators; the other
    # validators are no-ops on this params object, so the failure must come
    # from the patched checks.
    with pytest.raises(VLLMValidationError, match=match):
        params.verify(model_config, None, None, None)


def test_verify_accepts_in_vocab_ids():
    model_config = _make_model_config(129280)
    params = SamplingParams(stop_token_ids=[0, 129279], allowed_token_ids=[1, 2])
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
