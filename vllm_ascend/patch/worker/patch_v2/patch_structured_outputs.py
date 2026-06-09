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
"""NPU-compatible structured output bitmask for v2 model runner.

Upstream v2 `StructuredOutputsWorker.apply_grammar_bitmask` uses a CUDA Triton
kernel with async CUDA stream copies. On Ascend NPU the patched Triton kernel
may not correctly constrain logits during sampling, producing invalid guided
output. Reuse the xgrammar in-place path from v1, which is already validated on
NPU.
"""

from __future__ import annotations

import numpy as np
import torch
import xgrammar as xgr
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker


def apply_grammar_bitmask(
    self: StructuredOutputsWorker,
    logits: torch.Tensor,
    input_batch: InputBatch,
    grammar_req_ids: list[str],
    grammar_bitmask: np.ndarray,
) -> None:
    if not grammar_req_ids:
        return

    mapping: list[int] = []
    req_ids = input_batch.req_ids
    cu_num_logits = input_batch.cu_num_logits_np.tolist()
    req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
    for grammar_req_id in grammar_req_ids:
        req_idx = req_id_to_idx[grammar_req_id]
        logits_start_idx = cu_num_logits[req_idx]
        logits_end_idx = cu_num_logits[req_idx + 1]
        mapping.extend(range(logits_start_idx, logits_end_idx))

    num_masks = grammar_bitmask.shape[0]
    assert num_masks == len(mapping)

    bitmask_tensor = torch.from_numpy(grammar_bitmask).to(
        logits.device, non_blocking=True
    )
    index_tensor = None
    if len(mapping) != logits.shape[0]:
        index_tensor = torch.tensor(
            mapping, dtype=torch.int32, device=logits.device
        )

    xgr.apply_token_bitmask_inplace(logits, bitmask_tensor, indices=index_tensor)


StructuredOutputsWorker.apply_grammar_bitmask = apply_grammar_bitmask
