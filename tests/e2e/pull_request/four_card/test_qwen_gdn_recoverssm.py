# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.e2e.conftest import VllmRunner


def test_qwen36_recoverssm_matches_mtp_baseline_and_graph(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = ["What is 17 times 23?", "Explain why the sky is blue.", "Write a Python function to reverse a list."]
    baseline = None
    for recovery, eager in [(False, True), (True, True), (True, False)]:
        with VllmRunner(
            "Qwen/Qwen3.6-35B-A3B",
            tensor_parallel_size=4,
            dtype="bfloat16",
            mamba_ssm_cache_dtype="float32",
            mamba_cache_mode="none",
            enable_prefix_caching=False,
            use_replayssm=recovery,
            speculative_config={"method": "mtp", "num_speculative_tokens": 4},
            enforce_eager=eager,
            max_model_len=4096,
            max_num_seqs=8,
            max_num_batched_tokens=1024,
            gpu_memory_utilization=0.9,
        ) as runner:
            result = runner.generate_greedy(prompts, 64)
            # Second call exercises finished-request cache reuse.
            reused = runner.generate_greedy(list(reversed(prompts)), 64)
        tokens = [item[0] for item in result]
        assert all(tokens)
        assert [item[0] for item in reused] == list(reversed(tokens))
        if baseline is None:
            baseline = tokens
        else:
            assert tokens == baseline
