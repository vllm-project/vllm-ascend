import vllm.envs as envs
from vllm.config.vllm import VllmConfig

from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    defaults_model_runner_v2,
    get_current_hardware_profile,
)

_original_validate_v2_model_runner = VllmConfig._validate_v2_model_runner

_ASCEND_V1_SUPPORTED_FEATURES = frozenset(
    {
        "dspark speculative decoding",
        "dflash2 drafts",
    }
)


def _patched_use_v2_model_runner(self) -> bool:
    """Select Ascend Model Runner V2 from env, with 310P defaulting to V2.

    The upstream use_v2_model_runner gate-keeps the v2 runner with
    per-model architecture whitelists, Triton availability checks, and
    feature-support inspections. On Ascend the v2 runner is controlled
    by VLLM_USE_V2_MODEL_RUNNER when set; model-compatibility decisions
    are deferred to the NPU runner itself.

    When the env var is unset:
    - Compatibility stacks without STANDARD_WORKER_PATCHES (Ascend 310P)
      default to Model Runner V2 (MRV1 is no longer maintained there).
    - Standard Ascend platforms keep Model Runner V1 until they opt in with
      VLLM_USE_V2_MODEL_RUNNER=1.
    Explicit 0 / 1 overrides always win.
    """
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2
    return defaults_model_runner_v2()


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)


def _patched_validate_v2_model_runner(self) -> None:
    if not get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
        return
    _original_validate_v2_model_runner(self)


VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner

# Both supported vLLM versions expose this helper.
_original_get_v1_model_runner_unsupported_features = VllmConfig._get_v1_model_runner_unsupported_features


def _patched_get_v1_model_runner_unsupported_features(self) -> list[str]:
    unsupported = _original_get_v1_model_runner_unsupported_features(self)
    return [feature for feature in unsupported if feature not in _ASCEND_V1_SUPPORTED_FEATURES]


VllmConfig._get_v1_model_runner_unsupported_features = _patched_get_v1_model_runner_unsupported_features
