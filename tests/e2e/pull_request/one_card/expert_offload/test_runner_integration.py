# SPDX-License-Identifier: Apache-2.0
"""Real checkpoint loading and MRV2 scheduling; no manually adopted bindings."""

import gc
import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig
from vllm import LLM, SamplingParams

from vllm_ascend.utils import is_950

pytestmark = pytest.mark.skipif(not is_950(), reason="950 MXFP4 checkpoint/runner integration")


@pytest.fixture(autouse=True)
def use_mrv2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Write ordinary HF shards with real packed MXFP weights and E8M0 scales."""
    root = tmp_path_factory.mktemp("mxfp4-qwen3-moe")
    hidden, intermediate, experts, layers, vocab = 256, 256, 8, 2, 256
    config = Qwen3MoeConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=512,
        moe_intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=experts,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
        architectures=["Qwen3MoeForCausalLM"],
        quantization_config={"quant_method": "ascend", "model_quant_type": "W4A8_MXFP"},
    )
    config.save_pretrained(root)
    generator = torch.Generator().manual_seed(17)
    weights = {}

    def floating(name, shape, *, norm=False):
        weights[name] = (
            torch.ones(shape, dtype=torch.bfloat16)
            if norm
            else (torch.randn(shape, generator=generator) * 0.02).to(torch.bfloat16)
        )

    floating("model.embed_tokens.weight", (vocab, hidden))
    floating("model.norm.weight", (hidden,), norm=True)
    floating("lm_head.weight", (vocab, hidden))
    description = {"group_size": 32, "model_quant_type": "W4A8_MXFP"}
    for index in range(layers):
        prefix = f"model.layers.{index}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            floating(f"{prefix}.{name}.weight", (hidden,), norm=True)
        for name, output in (("q_proj", hidden), ("k_proj", 128), ("v_proj", 128), ("o_proj", hidden)):
            floating(f"{prefix}.self_attn.{name}.weight", (output, hidden))
        for name in ("q_norm", "k_norm"):
            floating(f"{prefix}.self_attn.{name}.weight", (64,), norm=True)
        floating(f"{prefix}.mlp.gate.weight", (experts, hidden))
        for expert in range(experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                output, inputs = (hidden, intermediate) if projection == "down_proj" else (intermediate, hidden)
                name = f"{prefix}.mlp.experts.{expert}.{projection}"
                weights[name + ".weight"] = torch.randint(
                    0, 256, (output, inputs // 2), dtype=torch.uint8, generator=generator
                )
                weights[name + ".weight_scale"] = torch.full((output, inputs // 32), 123, dtype=torch.uint8)
                description[name + ".weight"] = "W4A8_MXFP"
    for name in weights:
        if name.endswith(".weight"):
            description.setdefault(name, "FLOAT")
    save_file(weights, root / "model.safetensors", metadata={"format": "pt"})
    (root / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")
    return str(root)


def _stop(llm):
    llm.llm_engine.engine_core.shutdown()
    gc.collect()
    torch.npu.empty_cache()


def _generate_batches(llm, batches):
    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=1)
    result = []
    for batch in batches:
        outputs = llm.generate([{"prompt_token_ids": tokens} for tokens in batch], sampling, use_tqdm=False)
        for output in outputs:
            completion = output.outputs[0]
            logprobs = [entry[token].logprob for token, entry in zip(completion.token_ids, completion.logprobs)]
            assert all(math.isfinite(value) for value in logprobs)
            result.append((list(completion.token_ids), logprobs))
    return result


def test_checkpoint_prefill_graph_decode_capacity_fallback_and_cleanup(tiny_checkpoint):
    """A different model name must pass the same loader/layout contract end to end."""
    prompts = [[2 + (offset + index) % 200 for index in range(40)] for offset in range(3)]
    batches = [[prompts[0]], prompts, [prompts[1]]]
    common = dict(
        model=tiny_checkpoint,
        skip_tokenizer_init=True,
        worker_extension_cls="tests.e2e.pull_request.one_card.expert_offload.runner_utils.ExpertOffloadAuditWorker",
        dtype="bfloat16",
        quantization="ascend",
        async_scheduling=False,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        kv_cache_memory_bytes=32 * 1024 * 1024,
        enable_prefix_caching=False,
        seed=17,
    )
    llm = LLM(
        **common,
        additional_config={"expert_offload_config": {"enabled": True, "num_device_experts": 4}},
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]},
    )
    try:
        before = llm.collective_rpc("expert_offload_test_audit", args=("install",))[0]
        actual = _generate_batches(llm, batches)
        after = llm.collective_rpc("expert_offload_test_audit", args=("snapshot",))[0]
        assert before["loaded"] and len(before["layer_names"]) == 2
        assert set(before["capture_sizes"]) == {1, 2}
        assert after["addresses"] == before["addresses"]
        assert after["state"] == "READY" and after["replays"] > 0
        assert any(step["prefill"] for step in after["steps"])
        assert any(not step["prefill"] and step["mode"] == "FULL" for step in after["steps"])
        assert any(not step["prefill"] and step["mode"] == "NONE" and step["tokens"] > 2 for step in after["steps"])
        closed = llm.collective_rpc("expert_offload_test_audit", args=("close",))[0]
        assert closed["closed_state"] == "CLOSED" and closed["released"]
    finally:
        _stop(llm)
    reference = LLM(**common, enforce_eager=True)
    try:
        expected = _generate_batches(reference, batches)
    finally:
        _stop(reference)
    assert len(actual) == len(expected)
    for (actual_tokens, actual_logprobs), (expected_tokens, expected_logprobs) in zip(actual, expected):
        assert actual_tokens == expected_tokens
        torch.testing.assert_close(torch.tensor(actual_logprobs), torch.tensor(expected_logprobs), rtol=1e-3, atol=1e-3)


def test_checkpoint_native_dspark_target_draft_lifecycle(request):
    """Full-checkpoint acceptance uses the original prepared-pinned storage path."""
    checkpoint = request.config.getoption("--expert-offload-model")
    if checkpoint is None:
        pytest.skip("pass --expert-offload-model with a local DeepSeek V4 checkpoint for DSpark acceptance")
    assert Path(checkpoint, "config.json").is_file()
    llm = LLM(
        model=checkpoint,
        skip_tokenizer_init=True,
        worker_extension_cls="tests.e2e.pull_request.one_card.expert_offload.runner_utils.ExpertOffloadAuditWorker",
        async_scheduling=False,
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        block_size=32,
        kv_cache_dtype="bfloat16",
        kv_cache_memory_bytes=1024 * 1024 * 1024,
        additional_config={"expert_offload_config": {"enabled": True, "num_device_experts": 48}},
        speculative_config={"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": True},
        compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [6]},
    )
    try:
        before = llm.collective_rpc("expert_offload_test_audit", args=("install",))[0]
        assert before["loaded"] and before["target_manager"]
        assert before["target_layers"] > 0 and before["draft_layers"] > 0
        assert before["draft_offload_layers"] == 0 and before["draft_resident_on_npu"]
        _generate_batches(llm, [[list(range(2, 130))], [list(range(3, 67))]])
        after = llm.collective_rpc("expert_offload_test_audit", args=("snapshot",))[0]
        assert after["state"] == "READY" and after["replays"] > 0
        assert after["addresses"] == before["addresses"]
        assert after["draft_addresses"] == before["draft_addresses"]
        assert after["draft_offload_layers"] == 0 and after["draft_resident_on_npu"]
        closed = llm.collective_rpc("expert_offload_test_audit", args=("close",))[0]
        assert closed["closed_state"] == "CLOSED" and closed["released"]
    finally:
        _stop(llm)
