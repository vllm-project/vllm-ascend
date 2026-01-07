# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/batch_invariant.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
#
import os

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.triton_utils import HAS_TRITON

logger = init_logger(__name__)

if HAS_TRITON:
    from vllm_ascend.ops.triton.batch_invariant.matmul import (
        addmm_batch_invariant, bmm_batch_invariant, linear_batch_invariant,
        matmul_batch_invariant, mm_batch_invariant)


def override_envs_for_invariance():
    # TODO(Ronald) set attntion backend to deterministic mode

    # enabling NZ mode introduces NZ format input to the triton operator,
    # resulting in accuracy anomalies.
    os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"

    # communication determinism settings
    os.environ["HCCL_DETERMINISTIC"] = "true"
    os.environ["LCCL_DETERMINISTIC"] = "1"


_batch_invariant_LIB = None


def enable_batch_invariant_mode():
    global _batch_invariant_LIB

    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "NPU")
    _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "NPU")


def init_batch_invariance():
    """
    Initialize batch-invariant mode for vLLM on Ascend NPU.

    This function:
    1. Sets environment variables for deterministic computation
    2. Registers batch-invariant implementations for torch operators
    3. Enables batch-invariant flash attention

    Call this function early in your application, or set VLLM_BATCH_INVARIANT=1
    environment variable to enable automatically.
    """
    if vllm_is_batch_invariant():
        if HAS_TRITON:
            logger.info(
                "Enabling batch-invariant mode for vLLM on Ascend NPU.", )
            override_envs_for_invariance()
            enable_batch_invariant_mode()
        else:
            logger.warning(
                "Batch-invariant mode requested but Triton is not available."
                "skipping batch-invariant initialization.", )
