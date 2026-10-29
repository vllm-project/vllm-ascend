from vllm.config.vllm import VllmConfig

from vllm_ascend.mrv2_utils import apply_v2_model_runner_config_patch

# Default to the Ascend V2 runner unless the environment explicitly selects
# V1, or the config hits the V2 feature blacklist.
apply_v2_model_runner_config_patch()

_ASCEND_V1_SUPPORTED_FEATURES = frozenset(
    {
        "dspark speculative decoding",
        "dflash2 drafts",
        # Ascend V1 has its own PP communication and scheduler adaptations.
        "pipeline parallelism with async scheduling",
    }
)


<<<<<<< HEAD
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
    if is_310p():
        return
    _original_validate_v2_model_runner(self)


VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner

# vLLM main exposes this helper; v0.28.0 does not. Prefer hasattr over
# vllm_version_is(): CI installs from a commit SHA can report __version__="dev"
# and would otherwise apply the main-only patch on a release-lane checkout.
if hasattr(VllmConfig, "_get_v1_model_runner_unsupported_features"):
    _original_get_v1_model_runner_unsupported_features = VllmConfig._get_v1_model_runner_unsupported_features

    def _patched_get_v1_model_runner_unsupported_features(self) -> list[str]:
        unsupported = _original_get_v1_model_runner_unsupported_features(self)
        return [feature for feature in unsupported if feature not in _ASCEND_V1_SUPPORTED_FEATURES]
=======
# Both supported vLLM versions expose this helper.
# Runner selection and upstream V2 validation stay in mrv2_utils.
_original_get_v1_model_runner_unsupported_features = VllmConfig._get_v1_model_runner_unsupported_features
>>>>>>> ff37dbaab5c58d567daaecfbb0337447e7e90274

    VllmConfig._get_v1_model_runner_unsupported_features = _patched_get_v1_model_runner_unsupported_features
