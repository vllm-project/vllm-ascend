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
#
"""
Compare the outputs of vLLM with and without aclgraph.

Run `pytest tests/compile/test_aclgraph.py`.
"""


import os
from unittest.mock import patch

import pytest
import torch
from vllm import LLM, SamplingParams


MODELS = ["Qwen/Qwen3-8B"]

@pytest.mark.skipif(os.getenv("VLLM_USE_V1") == "0",
                    reason="aclgraph only support on v1")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [4])
def test_acl_graph_auto_enable(
    model: str,
    max_tokens: int,
) -> None:
    from vllm_ascend.platform import NPUPlatform
    original_check = NPUPlatform.check_and_update_config
    captured_config = None

    def mock_check_and_update_config(vllm_config):
        nonlocal captured_config
        captured_config = vllm_config
        return original_check(vllm_config)

    with patch.object(NPUPlatform,
                      "check_and_update_config",
                      side_effect=mock_check_and_update_config):
        prompts = [
            "Hello, my name is", "The president of the United States is",
            "The capital of France is", "The future of AI is"
        ]
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         temperature=0.0)
        vllm_model = LLM(model)
        _ = vllm_model.generate(prompts, sampling_params)
        assert captured_config is not None, "NPUPlatform.check_and_update_config was not called"
        assert captured_config.model_config.enforce_eager is False, \
            f"Expected enforce_eager=False, got {captured_config.model_config.enforce_eager}"
        from vllm.config import CompilationLevel
        from vllm.config.compilation import CUDAGraphMode

        assert captured_config.compilation_config.level == CompilationLevel.PIECEWISE, \
            f"Expected compilation level PIECEWISE, got {captured_config.compilation_config.level}"

        assert captured_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE, \
            f"Expected cudagraph_mode PIECEWISE, got {captured_config.compilation_config.cudagraph_mode}"

        assert captured_config.compilation_config.use_inductor is False, \
            "Expected compilation use_inductor is False"
        del vllm_model
        torch.npu.empty_cache()


