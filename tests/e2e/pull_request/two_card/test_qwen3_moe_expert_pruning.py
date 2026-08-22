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
# This file is a part of the vllm-ascend project.
#

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import AutoConfig
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from vllm_ascend.utils import vllm_version_is

pytestmark = pytest.mark.skipif(
    not (vllm_version_is("0.22.1") or vllm_version_is("0.23.0")),
    reason="Qwen3 MoE pruning patch is verified with vLLM v0.22.1/v0.23.0.",
)


def _get_model_path() -> str:
    local_model = Path("/root/models/Qwen/Qwen3-30B-A3B-W8A8")
    return str(local_model) if local_model.is_dir() else "vllm-ascend/Qwen3-30B-A3B-W8A8"


def _get_text_config(model: str):
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return getattr(config, "text_config", config)


def _write_ordered_expert_list(model: str, path: Path) -> tuple[float, int]:
    config = _get_text_config(model)
    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    top_k = config.num_experts_per_tok
    prune_ratio = 0.5
    kept_experts = max(top_k, int(num_experts * (1.0 - prune_ratio)))
    lines = [json.dumps({str(layer_idx): list(range(num_experts))}) for layer_idx in range(num_layers)]
    path.write_text("\n".join(lines), encoding="utf-8")
    return prune_ratio, kept_experts


@patch.dict(os.environ, {"OMP_NUM_THREADS": "1"})
@wait_until_npu_memory_free()
def test_qwen3_moe_expert_pruning_generates_with_pruned_expert_ids(tmp_path):
    model = _get_model_path()
    expert_list_file = tmp_path / "qwen3_moe_ordered_expert_list.jsonl"
    prune_ratio, kept_experts = _write_ordered_expert_list(model, expert_list_file)

    prompts = [
        "Briefly explain what deep learning is.",
    ]

    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_return_routed_experts=True,
        distributed_executor_backend="mp",
        quantization="ascend",
        max_model_len=1024,
        additional_config={
            "expert_list_file": str(expert_list_file),
            "expert_prune_ratio": prune_ratio,
        },
        cudagraph_capture_sizes=[1, 2, 4, 8],
    ) as vllm_model:
        sampling_params = SamplingParams(
            max_tokens=8,
            temperature=0.0,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        outputs = vllm_model.model.generate(
            prompts=vllm_model.get_inputs(prompts=prompts),
            sampling_params=sampling_params,
        )

    assert outputs[0].finished
    assert outputs[0].outputs[0].text
    routed_experts = outputs[0].outputs[0].routed_experts
    assert routed_experts.size > 0
    assert int(routed_experts.max()) < kept_experts
