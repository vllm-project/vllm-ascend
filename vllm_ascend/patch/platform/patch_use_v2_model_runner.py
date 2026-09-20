from collections.abc import Iterator
from contextlib import contextmanager

import vllm.envs as envs
from pydantic.dataclasses import rebuild_dataclass
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.platforms import current_platform

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support

_original_validate_v2_model_runner = VllmConfig._validate_v2_model_runner
_original_get_unsupported_features = VllmConfig._get_v2_model_runner_unsupported_features

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


def _patched_get_unsupported_features(self) -> list[str]:
    unsupported = _original_get_unsupported_features(self)
    if vllm_version_is("0.28.0") and "prefill context parallelism" in unsupported:
        # The release GPU runner rejects non-MLA PCP. AscendPCPManager owns
        # PCP execution and validates its model, graph and speculator limits.
        unsupported.remove("prefill context parallelism")
    support = resolve_spec_pp_support(self)
    unsupported_feature = support.unsupported_feature if support is not None else None
    if unsupported_feature is not None and unsupported_feature in unsupported:
        unsupported.remove(unsupported_feature)
    return unsupported


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
VllmConfig._get_v2_model_runner_unsupported_features = _patched_get_unsupported_features


def _patched_validate_v2_model_runner(self) -> None:
    if not get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
        return
    _original_validate_v2_model_runner(self)


VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner

# vLLM main exposes this helper; the supported v0.28.0 lane does not.
if not vllm_version_is("0.28.0"):
    _original_get_v1_model_runner_unsupported_features = VllmConfig._get_v1_model_runner_unsupported_features

    def _patched_get_v1_model_runner_unsupported_features(self) -> list[str]:
        unsupported = _original_get_v1_model_runner_unsupported_features(self)
        return [feature for feature in unsupported if feature not in _ASCEND_V1_SUPPORTED_FEATURES]

    VllmConfig._get_v1_model_runner_unsupported_features = _patched_get_v1_model_runner_unsupported_features


if vllm_version_is("0.28.0"):
    _original_validate_parallel_config = ParallelConfig._validate_parallel_config

    @contextmanager
    def _temporarily_disable_pcp_validation(config: ParallelConfig) -> Iterator[None]:
        pcp_size = config.prefill_context_parallel_size
        try:
            config.prefill_context_parallel_size = 1
            yield
        finally:
            config.prefill_context_parallel_size = pcp_size

    def _patched_validate_parallel_config(self: ParallelConfig) -> ParallelConfig:
        if (
            current_platform.device_name == "npu"
            and envs.VLLM_USE_V2_MODEL_RUNNER is True
            and self.data_parallel_size > 1
            and self.prefill_context_parallel_size > 1
            and self.decode_context_parallel_size == 1
        ):
            # __post_init__ computed world_size with the real PCP size. Keep
            # DP checks intact; DCP=1 is valid with either PCP value. Remove
            # this release-only workaround once vLLM #54523 is available.
            with _temporarily_disable_pcp_validation(self):
                return _original_validate_parallel_config(self)
        return _original_validate_parallel_config(self)

    ParallelConfig._validate_parallel_config = _patched_validate_parallel_config
    ParallelConfig.__pydantic_decorators__.model_validators[
        "_validate_parallel_config"
    ].func = _patched_validate_parallel_config
    # Rebuild dependencies before VllmConfig: SpeculativeConfig can retain
    # the old ParallelConfig schema even when its fields use SkipValidation.
    rebuild_dataclass(ParallelConfig, force=True)
    rebuild_dataclass(SpeculativeConfig, force=True)
    rebuild_dataclass(VllmConfig, force=True)
