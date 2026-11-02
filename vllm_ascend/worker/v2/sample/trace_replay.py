# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/trace_replay.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from vllm.v1.worker.gpu.sample import trace_replay


def apply_trace_tokens(
    sampled,
    idx_mapping,
    trace_token_ids,
    trace_len,
    total_len,
    prompt_len,
):
    num_reqs = idx_mapping.shape[0]
    if num_reqs == 0:
        return

    trace_replay._trace_replay_kernel[(num_reqs,)](
        sampled,
        idx_mapping,
        trace_token_ids,
        trace_token_ids.stride(0),
        trace_len,
        total_len,
        prompt_len,
    )
