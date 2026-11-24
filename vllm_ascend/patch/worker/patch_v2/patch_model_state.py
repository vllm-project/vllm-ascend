# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/default.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from vllm.v1.worker.gpu import model_runner

from vllm_ascend.worker.v2.model_states import init_asecnd_model_state

# prepare_attn in AscendModelState is different from vllm,
# we need to override init_model_state.
model_runner.init_model_state = init_asecnd_model_state
