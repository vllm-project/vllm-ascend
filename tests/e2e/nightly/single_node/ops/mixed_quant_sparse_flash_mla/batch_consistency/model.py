#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unified data model for batch consistency testing."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class TokenOrigin:
    """Identifies a single query token's origin in the baseline case."""

    source_batch: int
    source_token: int


@dataclass
class ConsistencyCase:
    """A derived test case produced by a consistency transform.

    ``input_data`` is a deep-enough copy of the baseline ``input_data`` with
    batch-relevant fields transformed.  ``output_origins`` maps each output
    token (in the transformed case's output order) back to its
    :class:`TokenOrigin` in the baseline.

    ``output_coordinates`` narrows comparison to explicit derived coordinates
    when a relation intentionally changes other outputs.
    """

    name: str
    input_data: Dict[str, Any]
    output_origins: List[TokenOrigin]
    transform_meta: Dict[str, Any] = field(default_factory=dict)
    output_coordinates: Optional[List[Tuple[int, int]]] = None


@dataclass
class RunResult:
    """Unified result returned by the consistency runner.

    The two operators use different legacy tuple contracts, so the thin
    operator adapters normalize only the tensors needed by the shared runner.
    """

    output: torch.Tensor
    softmax_lse: Optional[torch.Tensor] = None

    def as_legacy_tuple(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(output, softmax_lse)`` matching the old ``call_npu`` contract."""
        return self.output, self.softmax_lse
