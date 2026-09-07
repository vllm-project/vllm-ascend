# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config.vllm import VllmConfig

from vllm_ascend.patch.platform.patch_v1_dflash2_support import (
    DFLASH2_UNSUPPORTED_FEATURE,
    _original_get_v1_unsupported_features,
    _patched_get_v1_model_runner_unsupported_features,
)


@pytest.fixture
def config():
    return SimpleNamespace()


def _patch_original(monkeypatch, features):
    monkeypatch.setattr(
        "vllm_ascend.patch.platform.patch_v1_dflash2_support._original_get_v1_unsupported_features",
        lambda _self: list(features),
    )


def test_dflash2_blocker_is_dropped(monkeypatch, config):
    _patch_original(monkeypatch, [DFLASH2_UNSUPPORTED_FEATURE])

    assert _patched_get_v1_model_runner_unsupported_features(config) == []


def test_other_blockers_are_preserved(monkeypatch, config):
    _patch_original(
        monkeypatch,
        ["prefill context parallel", DFLASH2_UNSUPPORTED_FEATURE, "diffusion models"],
    )

    assert _patched_get_v1_model_runner_unsupported_features(config) == [
        "prefill context parallel",
        "diffusion models",
    ]


def test_list_without_dflash2_is_unchanged(monkeypatch, config):
    _patch_original(monkeypatch, ["dspark speculative decoding"])

    assert _patched_get_v1_model_runner_unsupported_features(config) == [
        "dspark speculative decoding",
    ]


@pytest.mark.skipif(
    _original_get_v1_unsupported_features is None,
    reason="vLLM does not gate v1 model runner features yet, so there is nothing to patch",
)
def test_patch_is_installed_on_vllm_config():
    assert VllmConfig._get_v1_model_runner_unsupported_features is _patched_get_v1_model_runner_unsupported_features
