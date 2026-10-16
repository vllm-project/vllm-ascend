# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import os
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = "Eco-Tech/DeepSeek-V4.1-Flash-w8a8"
MAX_MODEL_LEN = 16384
BLOCK_SIZE = 128


def _install_attention_capture(model, target_position: int) -> None:
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl

    assert not hasattr(AscendDSAV41Impl, "_test_replay_original_attention")
    original = AscendDSAV41Impl._forward_attention
    samples: dict[str, torch.Tensor] = {}

    def capture(impl, attn, query, metadata, compressed_indices, *, source_cache=None):
        result = original(impl, attn, query, metadata, compressed_indices, source_cache=source_cache)
        count = metadata.swa.num_actual_tokens
        if count and int(metadata.positions[count - 1].item()) == target_position:
            samples[impl.layer_name] = result[count - 1].detach().cpu().clone()
        return result

    AscendDSAV41Impl._test_replay_original_attention = original
    AscendDSAV41Impl._test_replay_attention_samples = samples
    AscendDSAV41Impl._forward_attention = capture


def _take_attention_capture(model) -> dict[str, torch.Tensor]:
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl

    samples = AscendDSAV41Impl._test_replay_attention_samples
    captured = dict(samples)
    samples.clear()
    return captured


def _remove_attention_capture(model) -> None:
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl

    AscendDSAV41Impl._forward_attention = AscendDSAV41Impl._test_replay_original_attention
    del AscendDSAV41Impl._test_replay_original_attention
    del AscendDSAV41Impl._test_replay_attention_samples


@pytest.mark.e2e_model(MODEL)
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
@wait_until_npu_memory_free()
def test_full_request_window_bounded_replay_with_prefix_hit():
    """Exercise a prefix hit while replaying only the sliding window."""
    with VllmRunner(
        MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=2048,
        tensor_parallel_size=4,
        dtype="auto",
        quantization="ascend",
        tokenizer_mode="deepseek_v41",
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        swa_bounded_replay=True,
        enforce_eager=True,
        attention_config={"indexer_kv_dtype": "int8"},
    ) as runner:
        assert runner.model.llm_engine.vllm_config.cache_config.swa_bounded_replay
        window = runner.model.model_config.get_sliding_window()
        assert window is not None
        target_hit = max(window * 2, runner.model.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens)
        target_hit = (target_hit + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
        assert target_hit + 32 < MAX_MODEL_LEN
        tokenizer = runner.model.get_tokenizer()
        segment = tokenizer.encode(
            "The following is background context for the final question.\n", add_special_tokens=False
        )
        prompt_ids = (segment * (target_hit // len(segment) + 1))[:target_hit]
        prompt_ids += tokenizer.encode(
            "\nQuestion: What is 13 plus 29? Answer with only the number.", add_special_tokens=False
        )
        assert window < len(prompt_ids) < MAX_MODEL_LEN - 16
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        sampling = SamplingParams(max_tokens=16, temperature=0.0, seed=1024)

        runner.model.apply_model(lambda model: _install_attention_capture(model, len(prompt_ids) - 1))
        try:
            cold = runner.model.generate([prompt], sampling, use_tqdm=False)[0]
            cold_attention = runner.model.apply_model(_take_attention_capture)
            warm = runner.model.generate([prompt], sampling, use_tqdm=False)[0]
            warm_attention = runner.model.apply_model(_take_attention_capture)
        finally:
            runner.model.apply_model(_remove_attention_capture)

        assert cold.num_cached_tokens == 0
        assert warm.num_cached_tokens is not None and warm.num_cached_tokens > window
        assert warm.num_cached_tokens % BLOCK_SIZE == 0
        assert cold.prompt_token_ids == warm.prompt_token_ids == prompt_ids
        assert len(cold.outputs) == len(warm.outputs) == 1
        assert cold.outputs[0].token_ids
        assert warm.outputs[0].token_ids
        print(f"cold/warm token equality: {cold.outputs[0].token_ids == warm.outputs[0].token_ids}")
        assert len(cold_attention) == len(warm_attention) == 4
        for rank, (cold_layers, warm_layers) in enumerate(zip(cold_attention, warm_attention)):
            assert cold_layers, f"rank {rank} captured no cold attention"
            assert cold_layers.keys() == warm_layers.keys(), f"rank {rank} attention layers differ"
            for layer_name in cold_layers:
                cold_output = cold_layers[layer_name]
                warm_output = warm_layers[layer_name]
                assert cold_output.shape == warm_output.shape
                assert torch.isfinite(cold_output).all()
                assert torch.isfinite(warm_output).all()
                max_abs_diff = (cold_output.float() - warm_output.float()).abs().max().item()
                print(f"rank {rank} {layer_name}: max attention difference {max_abs_diff}")
