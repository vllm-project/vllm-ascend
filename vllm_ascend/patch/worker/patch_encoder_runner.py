# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/mm/encoder_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from vllm.v1.worker.gpu.mm import encoder_runner

# Large multimodal inputs can exhaust the NPU driver's pinned-host pool.
# Keep encoder H2D staging on pageable host memory without changing the global
# pinned-memory policy used by unrelated buffers.
encoder_runner.PIN_MEMORY = False
