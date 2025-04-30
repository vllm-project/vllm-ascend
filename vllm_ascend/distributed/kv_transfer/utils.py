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
import torch
import llm_datadist  # type: ignore

TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32,
}

NPU_DTYPE_TO_TORCH_DTYPE = {
    llm_datadist.DataType.DT_FLOAT16: torch.half,
    llm_datadist.DataType.DT_FLOAT16: torch.float16,
    llm_datadist.DataType.DT_BF16: torch.bfloat16,
    llm_datadist.DataType.DT_FLOAT: torch.float,
    llm_datadist.DataType.DT_FLOAT: torch.float32,
    llm_datadist.DataType.DT_INT8: torch.int8,
    llm_datadist.DataType.DT_INT64: torch.int64,
    llm_datadist.DataType.DT_INT32: torch.int32,
}