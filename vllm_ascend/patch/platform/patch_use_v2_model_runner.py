import vllm.envs as envs
from vllm.config.vllm import VllmConfig

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

_original_validate_v2_model_runner = VllmConfig._validate_v2_model_runner

_ASCEND_V1_SUPPORTED_FEATURES = frozenset(
    {
        "dspark speculative decoding",
        "dflash2 drafts",
    }
)


def _patched_use_v2_model_runner(self) -> bool:
    """Return VLLM_USE_V2_MODEL_RUNNER env directly.

    The upstream use_v2_model_runner gate-keeps the v2 runner with
    per-model architecture whitelists, Triton availability checks, and
    feature-support inspections. On Ascend the v2 runner is controlled
    purely by the VLLM_USE_V2_MODEL_RUNNER environment variable;
    model-compatibility decisions are deferred to the NPU runner itself.
    """
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2
    return False


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
