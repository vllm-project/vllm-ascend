# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend adaptation of vLLM's CPU encoder-cache connector."""

from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.connector import (
    AscendECCPUConnector,
)

__all__ = ["AscendECCPUConnector"]
