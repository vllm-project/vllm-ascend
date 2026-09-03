import vllm.envs as envs
from vllm.config.vllm import VllmConfig

from vllm_ascend.device.hardware_profile import (
    ModelRunnerV2ImplementationFamily,
    get_current_hardware_profile,
)
from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support

_original_validate_v2_model_runner = VllmConfig._validate_v2_model_runner
_original_get_unsupported_features = VllmConfig._get_v2_model_runner_unsupported_features
_HARDWARE_PROFILE = get_current_hardware_profile()


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


def _patched_get_unsupported_features(self) -> list[str]:
    unsupported = _original_get_unsupported_features(self)
    support = resolve_spec_pp_support(self)
    unsupported_feature = support.unsupported_feature if support is not None else None
    if unsupported_feature is not None and unsupported_feature in unsupported:
        unsupported.remove(unsupported_feature)
    return unsupported


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
VllmConfig._get_v2_model_runner_unsupported_features = _patched_get_unsupported_features


def _patched_validate_v2_model_runner(self) -> None:
    if (
        _HARDWARE_PROFILE.model_runner_v2_implementation_family
        is ModelRunnerV2ImplementationFamily.TRITON_FREE_HOST_METADATA
    ):
        return
    _original_validate_v2_model_runner(self)


VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner
