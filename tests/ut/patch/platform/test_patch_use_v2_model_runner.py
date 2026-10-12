from types import SimpleNamespace

import pytest
from vllm.config.compilation import CompilationMode
from vllm.config.vllm import VllmConfig

from vllm_ascend.patch.platform import patch_use_v2_model_runner


def test_ascend_v1_supported_features_are_not_rejected(monkeypatch):
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


@pytest.mark.parametrize("method", ["eagle3", "mtp", "dspark"])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_native_v2_spec_pp_validation(method, enable_sp):
    config = SimpleNamespace(
        model_config=None,
        speculative_config=SimpleNamespace(method=method, parallel_drafting=False),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE, pass_config=SimpleNamespace(enable_sp=enable_sp)),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            tensor_parallel_size=4,
            distributed_executor_backend="mp",
            use_ubatching=False,
            enable_elastic_ep=False,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)
    # Spec+PP is natively supported; unrelated upstream guards still apply.
    assert unsupported == (["sequence parallelism"] if enable_sp else [])
