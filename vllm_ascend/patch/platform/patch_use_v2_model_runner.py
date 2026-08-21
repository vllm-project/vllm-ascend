import vllm.envs as envs
from vllm.config.vllm import VllmConfig

_original_get_unsupported_features = VllmConfig._get_v2_model_runner_unsupported_features


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
    speculative_config = self.speculative_config
    if (
        speculative_config is not None
        and speculative_config.method == "eagle3"
        and self.model_config.architecture == "Qwen3ForCausalLM"
        and self.parallel_config.pipeline_parallel_size > 1
        and "EAGLE3 with pipeline parallelism" in unsupported
    ):
        unsupported.remove("EAGLE3 with pipeline parallelism")
    return unsupported


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
VllmConfig._get_v2_model_runner_unsupported_features = _patched_get_unsupported_features
