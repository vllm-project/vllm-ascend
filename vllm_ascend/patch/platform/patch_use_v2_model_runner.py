from vllm.config.vllm import VllmConfig

from vllm_ascend.mrv2_utils import apply_v2_model_runner_config_patch
from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support

# Default to the Ascend V2 runner unless the environment explicitly selects
# V1, or the config hits the V2 feature blacklist.
apply_v2_model_runner_config_patch()

_original_get_unsupported_features = VllmConfig._get_v2_model_runner_unsupported_features

_ASCEND_V1_SUPPORTED_FEATURES = frozenset(
    {
        "dspark speculative decoding",
        "dflash2 drafts",
    }
)


def _patched_get_unsupported_features(self) -> list[str]:
    unsupported = _original_get_unsupported_features(self)
    support = resolve_spec_pp_support(self)
    unsupported_feature = support.unsupported_feature if support is not None else None
    if unsupported_feature is not None and unsupported_feature in unsupported:
        unsupported.remove(unsupported_feature)
    return unsupported


VllmConfig._get_v2_model_runner_unsupported_features = _patched_get_unsupported_features

# Both supported vLLM versions expose this helper.
# Runner selection and upstream V2 validation stay in mrv2_utils.
_original_get_v1_model_runner_unsupported_features = VllmConfig._get_v1_model_runner_unsupported_features


def _patched_get_v1_model_runner_unsupported_features(self) -> list[str]:
    unsupported = _original_get_v1_model_runner_unsupported_features(self)
    return [feature for feature in unsupported if feature not in _ASCEND_V1_SUPPORTED_FEATURES]


VllmConfig._get_v1_model_runner_unsupported_features = _patched_get_v1_model_runner_unsupported_features
