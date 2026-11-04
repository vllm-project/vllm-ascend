# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Mistral Large 4 bring-up on one NPU: a reduced model with dummy weights.

Functional smoke, not accuracy: no checkpoint or tokenizer is downloaded. The
reduced config keeps every per-layer pattern the full model has, at small
widths — NoPE on every fourth layer, a shared-indexer schedule, and a layer
that reads another layer's latent KV instead of owning one — because those
three are exactly what the Ascend adapter changes.
"""

import json
import math
from typing import Any

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.models import _upstream_has_module

# The same gate the registration uses, so the test skips exactly when the
# adapter is inert. A marker rather than ``importorskip``: a module-level skip
# collects no tests at all, which the job reports as a failure.
pytestmark = pytest.mark.skipif(
    not _upstream_has_module("vllm.models.mistral.mistral_large4"),
    reason="vLLM without Mistral Large 4 (vllm-project/vllm#64534)",
)

NUM_LAYERS = 8
VOCAB_SIZE = 4096
PROMPT_LENGTH = 64
OUTPUT_TOKENS = 8
NUM_LOGPROBS = 5
LOGPROB_ATOL = 1e-5

# NoPE on every fourth layer, as in the full model (idx % 4 == 3).
NOPE_FLAGS = [idx % 4 == 3 for idx in range(NUM_LAYERS)]
# Layer 7 reads layer 3's latent; both are NoPE layers, which is the pairing
# the real checkpoint uses (39..71 all read 35).
KV_SOURCE_LAYERS = [0, 1, 2, 3, 4, 5, 6, 3]
# Two layers reuse a neighbour's top-k rather than running their own indexer.
INDEX_SHARING = ["full", "full", "shared", "full", "full", "shared", "full", "full"]

SHARING_LAYERS = [i for i, src in enumerate(KV_SOURCE_LAYERS) if src != i]
INDEXER_LAYERS = [i for i, kind in enumerate(INDEX_SHARING) if kind == "full"]

CONFIG = {
    "model_type": "mistral_large4",
    "architectures": ["MistralLarge4ForCausalLM"],
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": NUM_LAYERS,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    # MLA: no q-LoRA, which is what switches the fused prolog off.
    "q_lora_rank": None,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 32,
    "v_head_dim": 64,
    "head_dim": 32,
    # DSA indexer. head_dim stays a power of two: the Ascend top-k kernel
    # reads a full row with tl.arange.
    "index_topk": 16,
    "index_n_heads": 4,
    "index_head_dim": 32,
    "indexer_norm_eps": 1e-5,
    "indexer_ssmax_scaling": True,
    "ssmax_scaling": True,
    # Per-layer patterns, in the field names the upstream draft reads.
    "nope_layer_flags": NOPE_FLAGS,
    "indexer_nope_layer_flags": NOPE_FLAGS,
    "kv_source_layers": KV_SOURCE_LAYERS,
    "index_sharing": INDEX_SHARING,
    # MoE: one dense layer, then routed experts with an always-on shared one.
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "moe_intermediate_size": 256,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "scoring_func": "softmax",
    "topk_method": None,
    "n_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 4096,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "softmax_tanh_gating": 30.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


class MistralLarge4Worker:
    """Read-only checks of which paths the loaded model actually takes."""

    model_runner: Any  # Supplied by the worker extension host at runtime.

    def check_mistral_large4_paths(self) -> bool:
        from vllm_ascend.models.mistral_large4 import (
            AscendMistralMLAAttention,
            AscendSharedKVSlotMappingMixin,
        )

        model = self.model_runner.model
        layers = model.model.layers
        attentions = [layer.self_attn for layer in layers]

        # Every attention module was switched, so none of them runs upstream's
        # CUDA-stream forward.
        assert all(isinstance(attn, AscendMistralMLAAttention) for attn in attentions)
        assert all(attn.indexer_stream is None for attn in attentions)
        assert all(attn.indexer_events == [] for attn in attentions)

        # The HIP-only decode runtime is never built.
        assert model.model.specialized_decode is None
        assert model.model.specialized_decode_config is None

        # The write skip is on the borrowing layers and nowhere else: a missed
        # layer would overwrite the latent cache it reads, silently.
        shared = [i for i, attn in enumerate(attentions) if isinstance(attn.attn.impl, AscendSharedKVSlotMappingMixin)]
        assert shared == SHARING_LAYERS, f"{shared} != {SHARING_LAYERS}"

        # The per-layer patterns survived the reduced config.
        assert [attn.spec.kv_source_layer for attn in attentions] == KV_SOURCE_LAYERS
        assert [attn.spec.use_rope for attn in attentions] == [not nope for nope in NOPE_FLAGS]
        assert [i for i, attn in enumerate(attentions) if attn.indexer is not None] == INDEXER_LAYERS

        # Borrowing layers hold no KV compression weights at all.
        for i in SHARING_LAYERS:
            assert not hasattr(attentions[i], "kv_a_proj_with_mqa")
        return True


def _write_model(destination, config: dict[str, Any] = CONFIG) -> None:
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
def test_mistral_large4_dummy_forward_and_logprob_replay(tmp_path):
    model = tmp_path / "model"
    _write_model(model)
    with VllmRunner(
        str(model),
        load_format="dummy",
        skip_tokenizer_init=True,
        worker_extension_cls=f"{__name__}.MistralLarge4Worker",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        seed=0,
    ) as runner:
        assert runner.model.collective_rpc("check_mistral_large4_paths") == [True]
        baseline = _generate(runner)
        replay = _generate(runner)
    _assert_replay_matches(baseline, replay)
