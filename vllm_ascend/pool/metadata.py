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
# Adapted from vllm-project/vllm/vllm/v1/pool/metadata.py
#
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.pooling_params import PoolingParams


@dataclass
class PoolingMetadata:
    """Tensors for pooling."""

    prompt_lens: torch.Tensor
    prompt_token_ids: Optional[torch.Tensor]
    pooling_params: list[PoolingParams]
