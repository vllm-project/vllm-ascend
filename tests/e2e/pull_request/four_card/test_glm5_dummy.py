# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline non-Flash GLM5.x DSA functional tests on four logical A3 NPUs.

Like test_kimi_k3.py, construct a local config and use load_format="dummy".
No checkpoint, tokenizer download, reduced-model tool or baseline is needed.
Eight layers retain three dense layers, routed MoE, and two full/shared indexer
groups. Production GLM-5.2 tensor widths and top-k are retained; the routed
expert count is reduced to 16. The independent-indexer variant uses the same
widths to exercise the non-sharing path, not every released GLM5.x checkpoint.

These are BF16 execution checks, not checkpoint-loading, quantization, MTP,
answer-accuracy, numerical-parity or performance gates. Flash is out of scope.
"""

import json
import math

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

NUM_LAYERS = 8
DENSE_LAYERS = 3
NUM_EXPERTS = 16
EXPERTS_PER_TOKEN = 8
VOCAB_SIZE = 154880
INDEX_TOPK = 2048
BLOCK_SIZE = 128
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 4
OUTPUT_TOKENS = 4
KV_CACHE_BYTES = 512 * 1024**2
SHARED_INDEXER_TYPES = ("full", "full", "full", "shared", "shared", "shared", "full", "shared")


def _model_config(shared_indexer: bool) -> dict:
    """Build a synthetic DSA config without importing the checkpoint builder."""
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "torch_dtype": "bfloat16",
        "hidden_size": 6144,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": NUM_LAYERS,
        "first_k_dense_replace": DENSE_LAYERS,
        "mlp_layer_types": ["dense"] * DENSE_LAYERS + ["sparse"] * (NUM_LAYERS - DENSE_LAYERS),
        "n_routed_experts": NUM_EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": EXPERTS_PER_TOKEN,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "moe_router_dtype": "float32",
        "moe_layer_freq": 1,
        "hidden_act": "silu",
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 256,
        "v_head_dim": 256,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": INDEX_TOPK,
        "index_topk_freq": 4 if shared_indexer else 1,
        "index_skip_topk_offset": DENSE_LAYERS,
        "indexer_types": list(SHARED_INDEXER_TYPES) if shared_indexer else ["full"] * NUM_LAYERS,
        "indexer_rope_interleave": True,
        "rope_interleave": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 8000000},
        "max_position_embeddings": MAX_MODEL_LEN,
        "num_nextn_predict_layers": 0,
        "vocab_size": VOCAB_SIZE,
        "eos_token_id": [154820, 154827, 154829],
        "pad_token_id": 154820,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-5,
        "initializer_range": 0.02,
        "use_cache": True,
    }


@pytest.fixture(scope="module", params=(False, True), ids=("independent-indexer", "shared-indexer"))
def glm5_dummy_model(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> str:
    """Each layout owns a local config; no other PR's files are consulted."""
    shared_indexer = request.param
    model = tmp_path_factory.mktemp("glm5-shared-dummy" if shared_indexer else "glm5-independent-dummy")
    config = _model_config(shared_indexer)
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(model)


@pytest.fixture(autouse=True)
def glm5_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    # Match K3's process isolation and runner selection. Also make accidental
    # hub lookups fail rather than silently reintroducing a model dependency.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setenv("HCCL_BUFFSIZE", "512")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _engine_args(enforce_eager: bool) -> dict:
    args: dict = {
        "load_format": "dummy",
        "dtype": "bfloat16",
        "skip_tokenizer_init": True,
        "tensor_parallel_size": 4,
        "enable_expert_parallel": True,
        "distributed_executor_backend": "mp",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_MODEL_LEN,
        "block_size": BLOCK_SIZE,
        "kv_cache_memory_bytes": KV_CACHE_BYTES,
        "gpu_memory_utilization": 0.8,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "async_scheduling": False,
        "enforce_eager": enforce_eager,
        "seed": 0,
        "additional_config": {"enable_cpu_binding": False, "enable_fused_mc2": 0},
    }
    if not enforce_eager:
        args["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, MAX_NUM_SEQS],
        }
    return args


def _generate_and_check(runner: VllmRunner, lengths: tuple[int, ...], salt: int) -> None:
    prompts = [
        {"prompt_token_ids": [10 + (index + salt + row * 137) % 1000 for index in range(length)]}
        for row, length in enumerate(lengths)
    ]
    outputs = runner.model.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True, detokenize=False, logprobs=1),
        use_tqdm=False,
    )
    assert len(outputs) == len(prompts)
    for output, prompt in zip(outputs, prompts):
        assert output.finished
        assert output.prompt_token_ids == prompt["prompt_token_ids"]
        assert len(output.outputs) == 1
        completion = output.outputs[0]
        assert len(completion.token_ids) == OUTPUT_TOKENS
        assert all(0 <= token < VOCAB_SIZE for token in completion.token_ids)
        assert completion.logprobs is not None and len(completion.logprobs) == OUTPUT_TOKENS
        for token, token_logprobs in zip(completion.token_ids, completion.logprobs):
            assert token in token_logprobs
            assert all(math.isfinite(entry.logprob) for entry in token_logprobs.values())


def _run_functional_case(model: str, *, enforce_eager: bool) -> None:
    with VllmRunner(model, **_engine_args(enforce_eager)) as runner:
        # Cross KV block boundaries and the real index_topk threshold. Multiple
        # output tokens exercise decode as well as prefill. Subsequent 1/4/3
        # request waves vary graph capacity and include a padded batch of three.
        lengths = (BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1, INDEX_TOPK + 1)
        _generate_and_check(runner, lengths, salt=0)
        _generate_and_check(runner, (2 * BLOCK_SIZE + 1,), salt=421)
        _generate_and_check(runner, lengths, salt=911)
        _generate_and_check(runner, lengths[:3], salt=1231)


@pytest.mark.e2e_model("GLM5.x-DSA-dummy")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_dsa,mixed_lengths,logprobs",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_glm5_dummy_tp4_eager(glm5_dummy_model: str) -> None:
    _run_functional_case(glm5_dummy_model, enforce_eager=True)


@pytest.mark.e2e_model("GLM5.x-DSA-dummy")
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="sfa_dsa,mixed_lengths,logprobs,aclgraph",
    parallel="TP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@wait_until_npu_memory_free()
def test_glm5_dummy_tp4_decode_graph(glm5_dummy_model: str) -> None:
    _run_functional_case(glm5_dummy_model, enforce_eager=False)
