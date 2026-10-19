from types import SimpleNamespace

import pytest
from vllm.config.compilation import CompilationMode
from vllm.config.vllm import VllmConfig

from vllm_ascend.patch.platform import patch_use_v2_model_runner


def test_use_v2_model_runner_is_driven_by_ascend_default():
    assert isinstance(VllmConfig.use_v2_model_runner, property)
    from vllm_ascend.mrv2_utils import use_v2_model_runner

    assert VllmConfig.use_v2_model_runner.fget is use_v2_model_runner


def test_ascend_v1_supported_features_are_not_rejected(monkeypatch):
    monkeypatch.setattr(
        patch_use_v2_model_runner,
        "_original_get_v1_model_runner_unsupported_features",
        lambda _: [
            "prefill context parallel",
            "dspark speculative decoding",
            "dflash2 drafts",
            "pipeline parallelism with async scheduling",
            "diffusion models",
        ],
    )

    unsupported = patch_use_v2_model_runner._patched_get_v1_model_runner_unsupported_features(object())

    assert unsupported == ["prefill context parallel", "diffusion models"]


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("prefill_context_parallel_size", [1, 2])
def test_native_v1_pp_async_validation(pipeline_parallel_size, async_scheduling, prefill_context_parallel_size):
    config = SimpleNamespace(
        model_config=None,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            enable_batch_sharded_sampling=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        _dflash_needs_multi_kv_group=lambda: False,
        _is_dflash2_draft=lambda: False,
        _is_dflash_candidate_draft=lambda: False,
    )
    config._get_v1_model_runner_unsupported_features = lambda: (
        patch_use_v2_model_runner._patched_get_v1_model_runner_unsupported_features(config)
    )

    # Exercise the real upstream helper and validator on both supported versions.
    # The Ascend PP exception must not suppress unrelated V1 restrictions.
    if prefill_context_parallel_size > 1:
        with pytest.raises(ValueError, match="prefill context parallel"):
            VllmConfig._validate_v1_model_runner(config)
    else:
        VllmConfig._validate_v1_model_runner(config)


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
