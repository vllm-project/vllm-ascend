from vllm.config.vllm import VllmConfig

from vllm_ascend.mrv2_utils import apply_v2_model_runner_config_patch
from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support

# Drive use_v2_model_runner from the Ascend model/feature whitelist instead of
# the env-only override. Also neutralize upstream V2 validation: Ascend owns
# that decision in mrv2_utils (see apply_v2_model_runner_config_patch).
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

# vLLM main exposes this helper; v0.28.0 does not. Prefer hasattr over
# vllm_version_is(): CI installs from a commit SHA can report __version__="dev"
# and would otherwise apply the main-only patch on a release-lane checkout.
if hasattr(VllmConfig, "_get_v1_model_runner_unsupported_features"):
    _original_get_v1_model_runner_unsupported_features = VllmConfig._get_v1_model_runner_unsupported_features

    def _patched_get_v1_model_runner_unsupported_features(self) -> list[str]:
        unsupported = _original_get_v1_model_runner_unsupported_features(self)
        return [feature for feature in unsupported if feature not in _ASCEND_V1_SUPPORTED_FEATURES]

    VllmConfig._get_v1_model_runner_unsupported_features = _patched_get_v1_model_runner_unsupported_features
