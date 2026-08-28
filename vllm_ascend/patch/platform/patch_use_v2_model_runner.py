import vllm.envs as envs
from vllm.config.vllm import VllmConfig

from vllm_ascend.utils import is_310p

_original_validate_v2_model_runner = VllmConfig._validate_v2_model_runner
_original_validate_v1_model_runner = getattr(VllmConfig, "_validate_v1_model_runner", None)


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
    if is_310p():
        return
    _original_validate_v2_model_runner(self)


VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner


def _patched_validate_v1_model_runner(self) -> None:
    # Upstream main validates that V1 is not used with features such as PCP,
    # dspark, dflash2, and diffusion. On Ascend, model/framework compatibility
    # is deferred to the NPU runner itself (see _patched_use_v2_model_runner);
    # the NPU V1 runner supports these features on its own, so the upstream
    # gate must not reject them. PCP on V1 is still rejected by
    # vllm_ascend.platform._validate_parallel_config.
    pass


if _original_validate_v1_model_runner is not None:
    VllmConfig._validate_v1_model_runner = _patched_validate_v1_model_runner
