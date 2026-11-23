# Qwen-family batch-invariance cases: every model runs the same BS=1 vs BS=N check.
import pytest

from tests.e2e.common.rlhf.batch_invariant.common import (
    assert_bitwise_batch_invariance,
    model_kwargs,
)
from tests.e2e.conftest import ModelName

DEFAULT_MODEL = ModelName.QWEN3_30B_A3B

# Shrink Qwen3-30B-A3B so it fits in CI: only the first 2 of the 48 hidden
# layers are instantiated, keeping memory and runtime small.
SMALL_QWEN3_OVERRIDES = {
    "num_hidden_layers": 2,
}

# Qwen3.5-35B-A3B is a hybrid MoE: 30 gated-delta-net layers plus 10
# full-attention layers, 256 experts top-8. It adds the GDN kernels and the
# linear/full interleave to what the Qwen3 case covers; the real checkpoint is
# already cached on the four-card runners.
QWEN35_MODEL = "Qwen/Qwen3.5-35B-A3B"


@pytest.mark.e2e_model("Qwen/Qwen3-30B-A3B")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="batch_invariant,logprobs",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_and_piecewise",
)
@pytest.mark.model(**model_kwargs(DEFAULT_MODEL, utilization=0.95, overrides=SMALL_QWEN3_OVERRIDES))
def test_qwen3_moe_logprobs_bitwise_batch_invariance_bs1_vs_bsN(vllm_runner):
    """Verify bitwise token/logprob invariance across batch sizes with the default graph configuration."""
    assert_bitwise_batch_invariance(vllm_runner)


@pytest.mark.e2e_model("Qwen/Qwen3.5-35B-A3B")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="batch_invariant,logprobs",
    parallel="TP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_and_piecewise",
)
@pytest.mark.model(**model_kwargs(QWEN35_MODEL, utilization=0.9))
def test_qwen35_moe_logprobs_bitwise_batch_invariance_bs1_vs_bsN(vllm_runner):
    """Same bitwise check as above, for the hybrid linear/full attention MoE model."""
    assert_bitwise_batch_invariance(vllm_runner)
