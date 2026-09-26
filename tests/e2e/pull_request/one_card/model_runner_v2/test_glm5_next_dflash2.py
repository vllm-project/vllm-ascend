#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Synthetic GLM-5.3-Flash + DFlash2 end-to-end test on model runner V2.

Runs on a single NPU: builds local configs for a reduced ``glm5_next`` target
and a reduced ``DFlash2DraftModel`` drafter, initializes dummy weights, and
exercises the whole speculative chain -- target aux hidden-state capture ->
``fc`` combine -> draft context-KV precompute -> parallel DFlash2 draft ->
rejection sampling.

The target config stays faithful to the shipping ``zai-org/GLM-5.3-Flash``
``text_config``: one full attention/MLP period (3 KDA + 1 sparse-MLA layer,
first 3 MLPs dense and the last one MoE), mHC with ``hc_mult=4``, and
production MLA/KDA/indexer/mHC widths. Only layer count, expert count, MoE
width and vocabulary are reduced, following the same recipe as
``tests/e2e/pull_request/four_card/test_kimi_k3.py``.

Greedy speculative decoding is lossless, so the drafted output must match the
non-speculative output token for token regardless of the (random) weights;
that equality is the end-to-end check of the capture/draft/verify plumbing.
Real-weight acceptance-rate baselines belong in nightly tests.
"""

import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter

from tests.e2e.conftest import VllmRunner

# One production attention/MLP period: 3 KDA layers + 1 sparse-MLA layer.
NUM_LAYERS = 4
FIRST_K_DENSE_REPLACE = 3
# Names/logits stay small; everything shape-critical keeps its trained width.
VOCAB_SIZE = 32768
HIDDEN_SIZE = 4096
NUM_ROUTED_EXPERTS = 16
MOE_INTERMEDIATE_SIZE = 512
# The drafter's block is 1 bonus + 7 mask queries, and the AscendC KDA kernels
# of the target accept at most 7 speculative tokens (KDA_MAX_RECURRENT_TOKENS).
NUM_SPECULATIVE_TOKENS = 7
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 2
OUTPUT_TOKENS = 8
PROMPT_LEN = 64
# Capture before layer k+1 yields the completed output of layer k, so taps on
# the 4-layer target land on aux layers [1, 2, 3].
DRAFT_TARGET_LAYER_IDS = (0, 1, 2)
MASK_TOKEN_ID = VOCAB_SIZE - 2


def _target_config() -> dict:
    return {
        "architectures": ["Glm5NextForCausalLM"],
        "model_type": "glm5_next_text",
        "torch_dtype": "bfloat16",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": 12288,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": False,
        "max_position_embeddings": 8192,
        # NoPE MLA skips the main RoPE path, but the v32 indexer builds its own
        # rotary embedding from these parameters.
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        # MoE: production expert width, reduced expert count.
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "n_routed_experts": NUM_ROUTED_EXPERTS,
        "num_experts_per_token": 4,
        "n_shared_experts": 1,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "first_k_dense_replace": FIRST_K_DENSE_REPLACE,
        "moe_layer_freq": 1,
        "use_grouped_topk": True,
        "n_group": 1,
        "topk_group": 1,
        "moe_renormalize": True,
        # MLA (production widths, NoPE).
        "mla": True,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "mla_use_nope": True,
        # KDA (production widths; the AscendC kernels require conv kernel 4).
        "linear_attn_config": {
            "head_dim": 128,
            "num_heads": 64,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        # Sparse indexer / k-pool (production).
        "index_head_dim": 128,
        "index_topk": 2048,
        "index_n_heads": 32,
        "index_kpool": 4,
        "index_dsa_use_layernorm": True,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        "indexer_rope_interleave": True,
        # mHC (production).
        "mhc": True,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "hres_vwnstyle": True,
        "mhc_no_norm_weight": False,
        "mlp_layer_types": ["dense"] * FIRST_K_DENSE_REPLACE + ["sparse"] * (NUM_LAYERS - FIRST_K_DENSE_REPLACE),
        "swiglu_limit": 10.0,
        "num_nextn_predict_layers": 0,
        "logit_scale": 1.0,
    }


def _draft_config() -> dict:
    return {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "torch_dtype": "bfloat16",
        # Must equal the target hidden size: the drafter shares the target's
        # embed_tokens/lm_head and its fc consumes the target's aux states.
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": 2048,
        "num_hidden_layers": 1,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB_SIZE,
        "tie_word_embeddings": False,
        "max_position_embeddings": 8192,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        # Same draft attention shape as the shipping drafter: all sliding-window
        # layers, non-causal (needs a non-causal-capable backend).
        "sliding_window": 2048,
        "use_sliding_window": True,
        "max_window_layers": 1,
        "layer_types": ["sliding_attention"],
        "is_causal": False,
        "dflash_config": {
            "block_size": NUM_SPECULATIVE_TOKENS + 1,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "mask_token_id": MASK_TOKEN_ID,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": list(DRAFT_TARGET_LAYER_IDS),
        },
    }


def _write_config(path: Path, config: dict) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def _write_tokenizer(path: Path) -> None:
    special_tokens = {0: "<unk>", 1: "</s>", 2: "<s>"}
    vocabulary = {special_tokens.get(i, f"token_{i}"): i for i in range(VOCAB_SIZE)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=list(special_tokens.values()),
    ).save_pretrained(path)


@pytest.fixture(scope="module")
def glm5_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    tmp_path = tmp_path_factory.mktemp("glm5-next-dflash2")
    target = _write_config(tmp_path / "target", _target_config())
    _write_tokenizer(tmp_path / "target")
    draft = _write_config(tmp_path / "draft", _draft_config())
    return {"target": target, "draft": draft}


@pytest.fixture(autouse=True)
def glm5_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _engine_args(draft: str | None) -> dict:
    args: dict = {
        # Dummy weights: the test validates the plumbing, not the checkpoint.
        "load_format": "dummy",
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": 256,
        "block_size": 128,
        "gpu_memory_utilization": 0.7,
        "enable_prefix_caching": True,
        "disable_log_stats": False,
        "seed": 0,
        # Left eager on purpose: MRV2 hybrid (KDA + sparse MLA) ACL graph capture
        # has no CI coverage yet, so this test isolates the DFlash2 chain. Graph
        # mode is validated separately (capture sizes must be multiples of
        # num_speculative_tokens + 1 = 8).
        "enforce_eager": True,
        "additional_config": {"enable_cpu_binding": False},
    }
    if draft is not None:
        args["speculative_config"] = {
            "method": "dflash",
            "model": draft,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            # The DFlash2 candidate selector is eager-only on model runner V2.
            "enforce_eager": True,
            "draft_load_config": {"load_format": "dummy"},
        }
    return args


def _prompt(salt: int = 0) -> dict:
    return {"prompt_token_ids": [10 + (i + salt) % 1000 for i in range(PROMPT_LEN)]}


def _generate(llm, prompts: list[dict]):
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.finished
        assert len(output.outputs) == 1
        assert len(output.outputs[0].token_ids) == OUTPUT_TOKENS
        assert all(0 <= token < VOCAB_SIZE for token in output.outputs[0].token_ids)
    return outputs


def test_glm5_next_dflash2_spec_decode_runs(glm5_models: dict[str, str]) -> None:
    """The target capture + DFlash2 draft + verify chain must actually run."""

    prompts = [_prompt(salt=0), _prompt(salt=17)]
    with VllmRunner(glm5_models["target"], **_engine_args(glm5_models["draft"])) as runner:
        _generate(runner.model, prompts)
        drafts = [m for m in runner.model.get_metrics() if m.name == "vllm:spec_decode_num_drafts"]

    assert drafts and all(isinstance(metric, Counter) for metric in drafts)
    assert sum(metric.value for metric in drafts) > 0, "requests bypassed DFlash2 speculative decoding"


def test_glm5_next_dflash2_greedy_is_lossless(glm5_models: dict[str, str]) -> None:
    """Greedy spec-decode output must equal the non-speculative output.

    Rejection sampling makes this hold for any draft quality (the spec step
    either accepts the target token or falls back to it), so random weights
    still validate the aux capture and the full draft/verify handshake.
    """

    prompts = [_prompt(salt=3)]
    with VllmRunner(glm5_models["target"], **_engine_args(None)) as runner:
        baseline = _generate(runner.model, prompts)[0].outputs[0].token_ids
    with VllmRunner(glm5_models["target"], **_engine_args(glm5_models["draft"])) as runner:
        speculative = _generate(runner.model, prompts)[0].outputs[0].token_ids

    assert list(speculative) == list(baseline)
