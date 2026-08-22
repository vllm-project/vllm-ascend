import vllm.envs as envs
from vllm.config.vllm import VllmConfig


def _patched_use_v2_model_runner(self) -> bool:
    """Return VLLM_USE_V2_MODEL_RUNNER env directly.

    The upstream use_v2_model_runner gate-keeps the v2 runner with
    per-model architecture whitelists, Triton availability checks, and
    feature-support inspections. Ascend normally requires the environment
    variable, except for features such as DFlash2 that have no correct V1
    implementation.
    """
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2

    # DFlash2's candidate selector is implemented only by the V2 speculator.
    # Let the upstream capability helper select V2 when available; otherwise a
    # DFlash2 checkpoint would silently run through the V1 DFlash1 proposer.
    is_dflash2_draft = getattr(self, "_is_dflash2_draft", None)
    return is_dflash2_draft is not None and is_dflash2_draft()


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
