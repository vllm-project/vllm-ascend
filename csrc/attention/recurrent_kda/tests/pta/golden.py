# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Compatibility wrapper for the Kimi recurrent KDA test reference."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_kimi_kda_recurrent_ascendc_npu import (  # noqa: E402
    recurrent_kda_reference as recurrent_kda_golden,
)

__all__ = ["recurrent_kda_golden"]
