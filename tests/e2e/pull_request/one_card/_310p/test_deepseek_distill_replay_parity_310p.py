# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Real-weight eager/ACL graph parity; run explicitly on a 310P device."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from modelscope import snapshot_download
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.device.device_config import is_310p

MAX_TOKENS = 32
REPLAY_ROUNDS = 3
BATCH_SIZES = (4, 1, 3, 2)
PROMPTS = (
    "Calculate 17 + 25. Explain your answer.",
    "Write a Python function that reverses a string.",
    "请用中文解释为什么天空看起来是蓝色的。",
    "Continue the sequence and explain the rule: 2, 4, 8, 16,",
)


@pytest.mark.skipif(not is_310p(), reason="This acceptance test requires Ascend 310P")
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ],
)
@wait_until_npu_memory_free(0.7)
def test_deepseek_distill_replay_parity(model_name, capfd, record_property, tmp_path):
    # Keep the two engines sequential so their weights/KV caches never coexist.
    # Prefix caching is disabled so every request exercises a fresh prefill.
    common = dict(
        dtype="float16",
        tensor_parallel_size=1,
        block_size=128,
        max_model_len=1024,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        seed=0,
        additional_config={"ascend_compilation_config": {"fuse_norm_quant": False, "enable_npugraph_ex": False}},
    )
    if model_name.endswith("Llama-8B"):
        # Preserve the checkpoint's byte-level pipeline on Transformers 5 too.
        checkpoint = snapshot_download(
            model_name,
            allow_patterns=["config.json", "tokenizer*.json", "special_tokens_map.json", "chat_template.jinja"],
        )
        tokenizer_path = tmp_path / "bytelevel-tokenizer"
        helper = Path(__file__).resolve().parents[5] / "examples" / "prepare_deepseek_distill_tokenizer.py"
        subprocess.run([sys.executable, str(helper), checkpoint, str(tokenizer_path)], check=True)
        common["tokenizer_name"] = str(tokenizer_path)
    # Force several decode steps even if an EOS is predicted at the first step.
    sampling = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True)
    references = {}
    with VllmRunner(model_name, enforce_eager=True, **common) as runner:
        for batch_size in BATCH_SIZES:
            outputs = runner.model.generate(list(PROMPTS[:batch_size]), sampling, use_tqdm=False)
            references[batch_size] = [list(output.outputs[0].token_ids) for output in outputs]
            assert len(outputs) == batch_size
            assert all(len(tokens) == MAX_TOKENS for tokens in references[batch_size])

    # Discard eager logs: only the graph engine may establish replay evidence.
    capfd.readouterr()
    measurements = []
    with VllmRunner(
        model_name,
        enforce_eager=False,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        **common,
    ) as runner:
        for replay_round in range(REPLAY_ROUNDS):
            for batch_size in BATCH_SIZES:
                start = time.perf_counter()
                outputs = runner.model.generate(list(PROMPTS[:batch_size]), sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
                actual = [list(output.outputs[0].token_ids) for output in outputs]
                assert actual == references[batch_size], (
                    f"{model_name}: round={replay_round}, batch={batch_size}, "
                    f"eager={references[batch_size]}, graph={actual}"
                )
                assert all(output.outputs[0].text.strip() for output in outputs)
                measurements.append({"round": replay_round, "batch_size": batch_size, "batch_seconds": elapsed})

    logs = capfd.readouterr()
    assert "ACL graph replay is active" in logs.out + logs.err, "No real graph replay evidence in worker logs"
    # Batch wall time includes prefill/decode/client overhead; this is not TTFT.
    record_property("model", model_name)
    record_property("eager_token_ids", json.dumps(references))
    record_property("graph_batch_timings", json.dumps(measurements))
