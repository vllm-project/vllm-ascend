# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

from vllm_ascend.worker.v2.spec_decode.draft_window_policy import (
    DraftWindowPolicyState,
    resolve_draft_window_policy,
)


@pytest.mark.parametrize("method", ["eagle", "mtp", "unknown", "dspark", "eagle3"])
def test_config_off_precedes_all_capability_checks(method):
    result = resolve_draft_window_policy(
        requested_window_size=None,
        method=method,
        target_model_type="deepseek_v4",
        dynamic_k=True,
        is_dflash2=True,
    )
    assert result.state is DraftWindowPolicyState.CONFIG_OFF
    assert not result.enabled
    assert result.effective_window_size is None


@pytest.mark.parametrize("requested", [0, -1, False, "not a number"])
@pytest.mark.parametrize("method,cotrained", [(" MTP ", None), ("dspark", True)])
def test_forced_off_precedes_invalid_number_and_capability(requested, method, cotrained):
    result = resolve_draft_window_policy(
        requested_window_size=requested,
        method=method,
        builtin_v4_dspark=cotrained,
        dynamic_k=True,
        is_dflash2=True,
    )
    assert result.state is DraftWindowPolicyState.FORCED_OFF
    assert result.effective_window_size is None


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41", "deepseek_v4_external_draft"])
def test_v4_name_does_not_prove_cotraining(model_type):
    with pytest.raises(ValueError, match="model-route identity"):
        resolve_draft_window_policy(requested_window_size=1024, method="dspark", target_model_type=model_type)
    independent = resolve_draft_window_policy(
        requested_window_size=1024,
        method="dspark",
        target_model_type=model_type,
        builtin_v4_dspark=False,
    )
    assert independent.enabled


@pytest.mark.parametrize("method", [" Eagle3 ", "dflash", "dspark"])
def test_static_supported_methods(method):
    result = resolve_draft_window_policy(requested_window_size=1024, method=method)
    assert result.enabled
    assert result.normalized_method == method.strip().lower()
    assert result.effective_window_size == 1024


def test_dflash2_uses_the_dflash_policy_without_a_new_method():
    result = resolve_draft_window_policy(requested_window_size=1024, method="dflash", is_dflash2=True)
    assert result.enabled
    assert result.normalized_method == "dflash"
    with pytest.raises(NotImplementedError, match="static"):
        resolve_draft_window_policy(requested_window_size=1024, method="dflash", is_dflash2=True, dynamic_k=True)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"method": "eagle"}, NotImplementedError),
        ({"method": "ngram"}, NotImplementedError),
        ({"method": "eagle3", "is_dflash2": True}, NotImplementedError),
        ({"method": "eagle3", "dynamic_k": True}, NotImplementedError),
        ({"method": "dspark", "dynamic_k": True}, NotImplementedError),
    ],
)
def test_unimplemented_policy_is_explicit(kwargs, error):
    with pytest.raises(error):
        resolve_draft_window_policy(requested_window_size=1024, **kwargs)


@pytest.mark.parametrize("value", [False, True, "1024", 1.5])
def test_noninteger_on_rejected(value):
    with pytest.raises(TypeError, match="integer"):
        resolve_draft_window_policy(requested_window_size=value, method="dspark")


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_on_rejected(value):
    with pytest.raises(ValueError, match="greater than zero"):
        resolve_draft_window_policy(requested_window_size=value, method="dspark")
