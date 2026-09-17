from vllm.config.vllm import VllmConfig

from vllm_ascend.patch.platform import patch_use_v2_model_runner
from vllm_ascend.utils import vllm_version_is


def test_ascend_v1_supported_features_are_not_rejected(monkeypatch):
    if vllm_version_is("0.28.0"):
        assert "_get_v1_model_runner_unsupported_features" not in VllmConfig.__dict__
        return

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
