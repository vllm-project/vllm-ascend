#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
from math import isfinite

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.environ.get("DEEPSEEK_V2_MODEL", "vllm-ascend/DeepSeek-V2-Lite-W8A8")
PROMPTS = [
    "Hello, what's your name?",
    "The capital of France is",
    "用三句话解释张量并行。",
]
OUTPUT_TOKENS = 8


class FlashCommWorker:
    def check_flashcomm_state(self):
        from vllm_ascend.utils import enable_sp

        enabled = enable_sp(self.vllm_config)
        layers = self.model_runner.model.model.layers
        assert any(layer.use_sequence_parallel_moe for layer in layers) is enabled
        return enabled


def _generate(runner, prompts):
    outputs = runner.model.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            max_tokens=OUTPUT_TOKENS,
            ignore_eos=True,
            temperature=0.0,
            logprobs=20,
            seed=1234,
        ),
        use_tqdm=False,
    )
    assert len(outputs) == len(prompts)
    sequences = [output.outputs[0] for output in outputs]
    for sequence in sequences:
        assert len(sequence.token_ids) == OUTPUT_TOKENS
        assert sequence.logprobs is not None
        assert len(sequence.logprobs) == OUTPUT_TOKENS
        assert all(len(topk) >= 20 for topk in sequence.logprobs)
        assert all(isfinite(value.logprob) for topk in sequence.logprobs for value in topk.values())
    return sequences


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
@wait_until_npu_memory_free(target_free_percentage=0.4)
def test_flashcomm1_additional_config_tp2(monkeypatch, enforce_eager) -> None:
    # The explicit additional-config value must win over the legacy environment
    # switch and reach worker subprocesses without a current-vLLM-config lookup.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("LCCL_DETERMINISTIC", "1")
    monkeypatch.setenv("ATB_MATMUL_SHUFFLE_K_ENABLE", "0")
    monkeypatch.setenv("CLOSE_MATMUL_K_SHIFT", "1")
    results = []
    for additional_config, legacy_value, enabled in (
        ({}, "1", True),
        ({"enable_flashcomm1": True}, "0", True),
        ({"enable_flashcomm1": False}, "1", False),
    ):
        monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", legacy_value)
        options = {}
        if not enforce_eager:
            options["compilation_config"] = {
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4],
            }
        with VllmRunner(
            MODEL,
            max_model_len=1024,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            enforce_eager=enforce_eager,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            worker_extension_cls=f"{__name__}.FlashCommWorker",
            gpu_memory_utilization=0.3,
            kv_cache_memory_bytes=2 * 1024**3,
            additional_config=additional_config,
            **options,
        ) as runner:
            assert runner.model.collective_rpc("check_flashcomm_state") == [enabled] * 2
            # The second call guarantees several single-token decode steps in
            # the same TP2/EP engine, including the ambiguous one-row SP case.
            sequences = _generate(runner, PROMPTS) + _generate(runner, [PROMPTS[-1]])
            if enabled:
                results.append(sequences)

    # Migration must preserve the legacy SP execution path. SP on/off can use
    # different reduction orders, so bitwise greedy parity across those paths
    # is not a configuration-migration invariant.
    for baseline, sequence_parallel in zip(*results):
        assert baseline.token_ids == sequence_parallel.token_ids
        for token, baseline_topk, sp_topk in zip(baseline.token_ids, baseline.logprobs, sequence_parallel.logprobs):
            assert sp_topk[token].logprob == pytest.approx(baseline_topk[token].logprob, abs=1e-2, rel=1e-3)
