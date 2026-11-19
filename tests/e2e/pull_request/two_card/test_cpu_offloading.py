# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Verify the shared MLA tier's capacity and a real CPU reload under TP2."""

import time

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


def test_mla_tiering_uses_one_copy_and_preserves_greedy_tokens(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    cpu_bytes = 512 << 20
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite-Chat",
        dtype="bfloat16",
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        enforce_eager=True,
        max_model_len=256,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": cpu_bytes,
                "blocks_per_chunk": 1,
            },
        ),
    )
    try:
        vllm_config = llm.llm_engine.vllm_config
        # Ascend MLA selects a supported block size during initialization.
        block_size = vllm_config.cache_config.block_size
        sampling = SamplingParams(max_tokens=8, temperature=0, seed=0)
        prompt = TokensPrompt(prompt_token_ids=[42] * (block_size + 1))
        cold = llm.generate(prompt, sampling, use_tqdm=False)[0]

        hf_config = vllm_config.model_config.hf_text_config
        # This BF16 MLA model has one unquantized latent page per layer.
        bytes_per_block = (
            block_size * hf_config.num_hidden_layers * (hf_config.kv_lora_rank + hf_config.qk_rope_head_dim) * 2
        )
        expected_chunks = cpu_bytes // round_up(bytes_per_block, SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT)
        configs = [
            metric
            for metric in llm.get_metrics()
            if metric.name == "vllm:kv_offload_config_info" and metric.labels.get("tier") == "0:primary"
        ]
        assert len(configs) == 1
        assert int(configs[0].labels["cpu_num_chunks"]) == expected_chunks, (
            block_size,
            bytes_per_block,
            configs[0].labels,
        )

        deadline = time.monotonic() + 60
        while not llm.reset_prefix_cache(reset_connector=False):
            assert time.monotonic() < deadline, "KV stores did not complete"
            llm.generate(
                TokensPrompt(prompt_token_ids=[1, 2]),
                SamplingParams(max_tokens=1, temperature=0),
                use_tqdm=False,
            )

        reloaded = llm.generate(prompt, sampling, use_tqdm=False)[0]
        assert reloaded.num_cached_tokens == block_size
        assert reloaded.outputs[0].token_ids == cold.outputs[0].token_ids
    finally:
        del llm
