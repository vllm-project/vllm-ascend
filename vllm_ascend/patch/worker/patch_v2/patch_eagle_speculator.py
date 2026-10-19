# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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

import importlib

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import AutoRegressiveAclGraphManager

# Upstream renamed the AR speculator module for the 0.32 line.
_vllm_speculator_module_name = (
    "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator"
    if vllm_version_is("0.31.0")
    else "vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator"
)
vllm_speculator_module = importlib.import_module(_vllm_speculator_module_name)

vllm_speculator_module.SpeculatorCudaGraphManager = AutoRegressiveAclGraphManager  # type: ignore[attr-defined]
