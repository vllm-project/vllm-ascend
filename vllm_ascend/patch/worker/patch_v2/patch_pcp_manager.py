# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from vllm.v1.worker.gpu import pcp_manager

from vllm_ascend.worker.v2.pcp_manager import maybe_build_ascend_pcp_manager

pcp_manager.maybe_build_pcp_manager = maybe_build_ascend_pcp_manager
