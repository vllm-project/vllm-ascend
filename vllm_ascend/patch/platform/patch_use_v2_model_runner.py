from vllm_ascend.mrv2_utils import apply_v2_model_runner_config_patch

# Ascend only ships Model Runner V2, so pin the runner selection on VllmConfig.
# Upstream still computes its own default from env and feature lists; this
# override keeps every process on V2 without carrying a V1 fallback.
apply_v2_model_runner_config_patch()
