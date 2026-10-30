# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Muse Spark Open bring-up on one NPU: a reduced model with dummy weights.

Functional smoke, not accuracy: no checkpoint or tokenizer is downloaded. The
reduced text config keeps every Muse Spark per-layer pattern (3:1 sliding to
full attention, NoPE on the full layer, dense and latent-MoE feed-forwards)
at small widths, and the sliding window is shorter than the prompt so the
windowed path is exercised.
"""

import json
import math
from typing import Any

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.models import _upstream_has_module

# The same gate the registration uses, so the test skips exactly when the adapter
# is inert. A marker rather than ``importorskip``: the release pin carries no Muse
# Spark, and a module-level skip there collects no tests at all, which the job
# reports as a failure instead of a skip.
pytestmark = pytest.mark.skipif(
    not _upstream_has_module("vllm.models.muse_spark.amd.model"),
    reason="vLLM without Muse Spark",
)

NUM_LAYERS = 4
VOCAB_SIZE = 4096
SLIDING_WINDOW = 16
PROMPT_LENGTH = 64
OUTPUT_TOKENS = 8
NUM_LOGPROBS = 5
LOGPROB_ATOL = 1e-5
GATE_ATOL = 2e-2

LAYER_TYPES = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
LAYER_ROPE_THETA = [10000.0, 10000.0, 10000.0, 0.0]
MLP_LAYER_TYPES = ["dense", "sparse", "sparse", "sparse"]

TEXT_CONFIG = {
    "model_type": "muse_spark_text",
    "architectures": ["MuseSpark1xForCausalLM"],
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "hidden_act": "silu",
    "attention_bias": False,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "post_norm_eps": 1e-5,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "layer_types": LAYER_TYPES,
    "layer_rope_theta": LAYER_ROPE_THETA,
    "mlp_layer_types": MLP_LAYER_TYPES,
    "sliding_window": SLIDING_WINDOW,
    "qk_scale_factor": 1.0,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "num_shared_experts": 0,
    "routed_scaling_factor": 1.0,
    "output_multiplier": 1.0,
    "moe_hidden_size": 256,
    "moe_intermediate_size": 256,
    "residual_gate_temperature": 0.3,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


class MuseSparkWorker:
    """Read-only checks of which Muse Spark paths the loaded model takes."""

    model_runner: Any  # Supplied by the worker extension host at runtime.

    def check_muse_spark_paths(self) -> bool:
        model = self.model_runner.model
        names = [type(module).__name__ for module in model.modules()]
        assert names.count("AscendMuseSpark1xAttention") == NUM_LAYERS
        assert names.count("AscendMuseSpark1xMoE") == MLP_LAYER_TYPES.count("sparse")
        assert "MuseSpark1xRMSNorm" not in names
        assert "MuseSpark1xRMSNormWithInputScale" not in names

        layers = model.model.layers
        windows = [layer.self_attn.attn.sliding_window for layer in layers]
        assert windows == [SLIDING_WINDOW if kind == "sliding_attention" else None for kind in LAYER_TYPES]
        has_rope = [layer.self_attn.rotary_emb is not None for layer in layers]
        assert has_rope == [theta > 0 for theta in LAYER_ROPE_THETA]
        assert not any(layer.self_attn.fused_qk_norm_rope_kv_insert for layer in layers)
        return True


def _write_model(destination, config: dict[str, Any] = TEXT_CONFIG) -> None:
    destination.mkdir()
    (destination / "config.json").write_text(json.dumps(config))


def _generate(runner: VllmRunner) -> list[tuple[list[int], list[dict]]]:
    params = SamplingParams(
        temperature=0,
        max_tokens=OUTPUT_TOKENS,
        logprobs=NUM_LOGPROBS,
        ignore_eos=True,
        detokenize=False,
    )
    prompts = [{"prompt_token_ids": [(7 * i + offset) % VOCAB_SIZE for i in range(PROMPT_LENGTH)]} for offset in (1, 2)]
    outputs = runner.model.generate(prompts, params, use_tqdm=False)
    return [(list(output.outputs[0].token_ids), output.outputs[0].logprobs) for output in outputs]


def _assert_replay_matches(baseline, replay) -> None:
    for (baseline_ids, baseline_logprobs), (replay_ids, replay_logprobs) in zip(baseline, replay):
        assert len(baseline_ids) == OUTPUT_TOKENS
        assert all(0 <= token < VOCAB_SIZE for token in baseline_ids)
        assert baseline_ids == replay_ids
        for step, (baseline_topk, replay_topk) in enumerate(zip(baseline_logprobs, replay_logprobs)):
            assert baseline_topk.keys() == replay_topk.keys(), f"top-k token mismatch at decode step {step}"
            for token_id, baseline_value in baseline_topk.items():
                assert math.isfinite(baseline_value.logprob)
                assert replay_topk[token_id].logprob == pytest.approx(baseline_value.logprob, abs=LOGPROB_ATOL)


@wait_until_npu_memory_free()
def test_muse_spark_dummy_forward_and_logprob_replay(tmp_path):
    model = tmp_path / "model"
    _write_model(model)
    with VllmRunner(
        str(model),
        load_format="dummy",
        skip_tokenizer_init=True,
        worker_extension_cls=f"{__name__}.MuseSparkWorker",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        seed=0,
    ) as runner:
        assert runner.model.collective_rpc("check_muse_spark_paths") == [True]
        baseline = _generate(runner)
        replay = _generate(runner)
    _assert_replay_matches(baseline, replay)


def _torch_rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * torch.sigmoid(gate.float())).to(x.dtype)


def test_muse_spark_output_gate_kernel_matches_the_weightless_reference():
    """Muse Spark's output norm has no learned weight; GLM's KDA norm does."""
    from vllm_ascend.ops.triton.kda.kda import rms_norm_gated

    torch.manual_seed(0)
    num_tokens, num_heads, head_dim, eps = 37, 8, 64, 1e-5
    x = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device="npu")
    gate = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device="npu")

    output = rms_norm_gated(x, gate, None, None, "sigmoid", eps=eps)

    torch.testing.assert_close(output, _torch_rms_norm_gated(x, gate, eps), atol=GATE_ATOL, rtol=GATE_ATOL)
