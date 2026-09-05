# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free

SFA_CP_MODEL = os.environ.get("SFA_CP_TEST_MODEL", "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning")


@pytest.mark.e2e_model(SFA_CP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_cp",
    parallel="TP,DP,EP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager,full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
@pytest.mark.parametrize("runner_version", ["0", "1"])
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "full_decode"])
def test_sfa_cp_sp_prefill_decode_accuracy(
    runner_version: str, enforce_eager: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner_version)
    prompt_lengths = [1, 3, 4095, 4096, 4097]
    generated_tokens: dict[bool, list[list[int]]] = {}
    for enable_dsa_cp in (False, True):
        mode_tokens = []
        with DPVllmRunner(
            SFA_CP_MODEL,
            max_model_len=8192,
            max_num_seqs=4,
            max_num_batched_tokens=8192,
            dtype="auto",
            data_parallel_size=2,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            gpu_memory_utilization=0.9,
            quantization="ascend",
            block_size=128,
            enforce_eager=enforce_eager,
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4]},
            additional_config={"enable_dsa_cp": enable_dsa_cp, "enable_flashcomm1": True},
        ) as runner:
            for prompt_length in prompt_lengths:
                prompt_tokens = [(index % 1000) + 100 for index in range(prompt_length)]
                outputs = runner.generate(
                    [prompt_tokens], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
                )
                output_ids = outputs[0][0][0]
                assert output_ids[:prompt_length] == prompt_tokens
                assert len(output_ids) == prompt_length + 8
                mode_tokens.append(output_ids[prompt_length:])
        generated_tokens[enable_dsa_cp] = mode_tokens
    assert generated_tokens[True] == generated_tokens[False]
