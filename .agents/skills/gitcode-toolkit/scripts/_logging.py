#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""Dual-logger pattern for deterministic GitCode toolkit scripts.

OUTPUT_LOGGER → stdout (clean JSON / machine-parseable)
ERROR_LOGGER  → stderr (human-readable diagnostics)
"""

from __future__ import annotations

import logging
import sys


def init_output_logger() -> logging.Logger:
    """Configure root logger to emit info-level messages to stdout."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    return logging.getLogger("output")


def init_error_logger() -> logging.Logger:
    """Configure error logger to emit error-level messages to stderr."""
    logger = logging.getLogger("error")
    logger.setLevel(logging.ERROR)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger