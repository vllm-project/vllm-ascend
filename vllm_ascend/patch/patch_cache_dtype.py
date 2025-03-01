#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
# This file is used to monkey patch int8 cache dtype in vllm to support ascend.
# Remove this file when vllm support int8 cache dtype.

import torch
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

STR_DTYPE_TO_TORCH_DTYPE['int8'] = torch.int8