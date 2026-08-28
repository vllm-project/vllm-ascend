#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Model Runner V2 SFA DCP accuracy guard.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_accuracy_v2.py`.
"""

import os
from unittest.mock import patch

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.four_card.context_parallel import test_accuracy as common

MODEL = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"

FULL_FEATURE_MODEL_CASES = common.AccuracyCase(
    name="dsv3_2_sfa_dcp_replicated_indexer_mrv2_tp2_dcp2",
    model=MODEL,
    prompts=common.COMMON_PROMPTS,
    expected_outputs=common.DSV3_2_DCP_GOLDENS,
    max_tokens=5,
    runner_kwargs={
        "max_model_len": 1024,
        "max_num_seqs": common.MAX_NUM_SEQS,
        "max_num_batched_tokens": 1024,
        "tensor_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "enable_expert_parallel": True,
        "gpu_memory_utilization": 0.4,
        "block_size": 128,
        "quantization": "ascend",
        "compilation_config": common.FULL_DECODE_GRAPH,
        "additional_config": {
            "enable_dsa_cp": False,
            "enable_sparse_li_c8": False,
        },
        "speculative_config": {
            "method": "mtp",
            "num_speculative_tokens": 3,
        },
    },
)


@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_dsv3_2_sfa_dcp_tp2_dcp2_model_runner_v2_accuracy() -> None:
    """Guard MRV2 accuracy."""
    common._run_accuracy_case(FULL_FEATURE_MODEL_CASES)
