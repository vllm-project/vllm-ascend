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

import pytest
from vllm.utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tools.aisbench import get_TTFT, run_aisbench_cases

MODELS = [
    "vllm-ascend/DeepSeek-R1-0528-W8A8",
]

aisbench_warm_up = [{
    "case_type": "performance",
    "dataset_path": "vllm-ascend/GSM8K-in1024-bs210",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 210,
    "max_out_len": 2,
    "batch_size": 1000,
    "baseline": 0,
    "threshold": 0.97
}]

aisbench_cases0 = [{
    "case_type": "performance",
    "dataset_path": "vllm-ascend/prefix0-in3500-bs210",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 210,
    "max_out_len": 1500,
    "batch_size": 18,
    "baseline": 1,
    "threshold": 0.97
}]

aisbench_cases75 = [{
    "case_type": "performance",
    "dataset_path": "vllm-ascend/prefix75-in3500-bs210",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_str_perf",
    "num_prompts": 210,
    "max_out_len": 1500,
    "batch_size": 18,
    "baseline": 1,
    "threshold": 0.97
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_models(model: str) -> None:
    port = get_open_port()
    env_dict = {
        "OMP_NUM_THREADS": "10",
        "OMP_PROC_BIND": "false",
        "HCCL_BUFFSIZE": "1024",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    additional_config = {
        "torchair_graph_config": {
            "enabled": True,
            "enable_multistream_moe": False,
            "enable_multistream_mla": True,
            "graph_batch_size": [16],
            "use_cached_graph": True
        },
        "chunked_prefill_for_mla": True,
        "enable_weight_nz_layout": True
    }
    speculative_config = {
        "num_speculative_tokens": 1,
        "method": "deepseek_mtp"
    }
    server_args = [
        "--quantization", "ascend", "--data-parallel-size", "2",
        "--tensor-parallel-size", "8", "--enable-expert-parallel", "--port",
        str(port), "--seed", "1024", "--max-model-len", "5200",
        "--max-num-batched-tokens", "4096", "--max-num-seqs", "16",
        "--trust-remote-code", "--gpu-memory-utilization", "0.9",
        "--additional-config",
        json.dumps(additional_config), "--speculative-config",
        json.dumps(speculative_config)
    ]
    with RemoteOpenAIServer(model,
                            server_args,
                            server_port=port,
                            env_dict=env_dict,
                            auto_port=False):
        run_aisbench_cases(model, port, aisbench_warm_up)
        result = run_aisbench_cases(model, port, aisbench_cases0)
        TTFT0 = get_TTFT(result)
    with RemoteOpenAIServer(model,
                            server_args,
                            server_port=port,
                            env_dict=env_dict,
                            auto_port=False):
        run_aisbench_cases(model, port, aisbench_warm_up)
        result = run_aisbench_cases(model, port, aisbench_cases75)
        TTFT75 = get_TTFT(result)
    assert TTFT75 < 0.8 * TTFT0, f"The TTFT for prefix75 {TTFT75} is not less than 0.8*TTFT for prefix0 {TTFT0}."
    print(
        f"The TTFT for prefix75 {TTFT75} is less than 0.8*TTFT for prefix0 {TTFT0}."
    )
