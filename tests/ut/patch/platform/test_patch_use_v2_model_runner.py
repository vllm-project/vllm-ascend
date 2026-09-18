import pytest
from vllm.config.vllm import VllmConfig

from vllm_ascend.patch.platform import patch_use_v2_model_runner


def test_use_v2_model_runner_is_driven_by_ascend_whitelist():
    assert isinstance(VllmConfig.use_v2_model_runner, property)
    from vllm_ascend.mrv2_utils import use_v2_model_runner

    assert VllmConfig.use_v2_model_runner.fget is use_v2_model_runner


def test_ascend_v1_supported_features_are_not_rejected(monkeypatch):
    if not hasattr(VllmConfig, "_get_v1_model_runner_unsupported_features"):
        pytest.skip("V1 model runner validation is only present on vLLM main")

    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "_original_get_v1_model_runner_unsupported_features",
        lambda _: [
            "prefill context parallel",
            "dspark speculative decoding",
            "dflash2 drafts",
            "diffusion models",
        ],
    )

    unsupported = patch_use_v2_model_runner._patched_get_v1_model_runner_unsupported_features(object())

    assert unsupported == ["prefill context parallel", "diffusion models"]


def test_release_pcp_is_not_rejected_as_v2_unsupported_feature(monkeypatch):
    monkeypatch.setattr(patch_use_v2_model_runner, "vllm_version_is", lambda version: version == "0.28.0")
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "_original_get_unsupported_features",
        lambda _: ["prefill context parallelism", "diffusion models"],
    )
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "resolve_spec_pp_support",
        lambda _: None,
    )

    unsupported = patch_use_v2_model_runner._patched_get_unsupported_features(object())

    assert unsupported == ["diffusion models"]
