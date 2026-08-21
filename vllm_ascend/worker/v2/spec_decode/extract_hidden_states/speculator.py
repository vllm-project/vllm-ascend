# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend MRV2 thin wrapper around upstream ExtractHiddenStatesSpeculator.

Requires upstream vLLM PR #49811 (``vllm.v1.worker.gpu.spec_decode.extract_hidden_states``).
"""

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
    ExtractHiddenStatesSpeculator,
)


class AscendExtractHiddenStatesSpeculator(ExtractHiddenStatesSpeculator):
    """Reuse upstream extract_hidden_states; keep Ascend update_stream hook."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # NPUModelRunner assigns update_stream for draft graph managers.
        # This method does not capture ACL graphs, but keep the attribute.
        self.update_stream = None
