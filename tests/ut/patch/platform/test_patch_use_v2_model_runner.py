from types import SimpleNamespace

import pytest
from vllm.config.compilation import CompilationMode
from vllm.config.vllm import VllmConfig

# Importing the module applies the V2-only runner selection at collection time.
from vllm_ascend.patch.platform import patch_use_v2_model_runner  # noqa: F401


def test_use_v2_model_runner_is_driven_by_ascend_default():
    assert isinstance(VllmConfig.use_v2_model_runner, property)
    from vllm_ascend.mrv2_utils import use_v2_model_runner

    assert VllmConfig.use_v2_model_runner.fget is use_v2_model_runner


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("prefill_context_parallel_size", [1, 2])
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
    # vLLM main (#60517) dropped the PassConfig.enable_sp guard, so the V2
    # runner no longer rejects sequence parallelism there.
    expected_sp: list[str] = []
    assert unsupported == expected_sp
