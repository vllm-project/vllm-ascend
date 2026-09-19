import vllm.v1.worker.gpu.spec_decode.adaptive_verification
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import _assign_draft_token_budget

vllm.v1.worker.gpu.spec_decode.adaptive_verification._assign_draft_token_budget_compiled = _assign_draft_token_budget

# vLLM #52228 adds a Triton-backed acceptance estimator for adaptive
# verification without a confidence head. Replace the constructor used by all
# draft-model speculators with the NPU implementation when that upstream module
# is available. vLLM 0.28 does not have the estimator and keeps the existing
# DSpark confidence-head path.
try:
    import vllm.v1.worker.gpu.spec_decode.acceptance_estimator
except ImportError:
    pass
else:
    import vllm.v1.worker.gpu.spec_decode.speculator as speculator_module

    from vllm_ascend.worker.v2.spec_decode.acceptance_estimator import (
        AscendOnlineAcceptanceEstimator,
    )

    speculator_module.OnlineAcceptanceEstimator = AscendOnlineAcceptanceEstimator
