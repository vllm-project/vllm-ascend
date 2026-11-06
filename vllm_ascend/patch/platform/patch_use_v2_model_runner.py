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


# Both supported vLLM versions expose this helper.
# Runner selection and upstream V2 validation stay in mrv2_utils.
_original_get_v1_model_runner_unsupported_features = VllmConfig._get_v1_model_runner_unsupported_features


def _patched_get_v1_model_runner_unsupported_features(self) -> list[str]:
    unsupported = _original_get_v1_model_runner_unsupported_features(self)
    return [feature for feature in unsupported if feature not in _ASCEND_V1_SUPPORTED_FEATURES]


VllmConfig._get_v1_model_runner_unsupported_features = _patched_get_v1_model_runner_unsupported_features
