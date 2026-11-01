# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Explicit model matrix and backend-aware pytest parameter selection."""

from typing import Any

from tests.e2e.pull_request.rlhf.weight_transfer.models.base import WeightUpdateModelCase
from tests.e2e.pull_request.rlhf.weight_transfer.models.deepseek import DEEPSEEK_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.glm import GLM_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.minimax import MINIMAX_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.qwen import QWEN_CASES

MODEL_CASES = (*QWEN_CASES, *DEEPSEEK_CASES, *GLM_CASES, *MINIMAX_CASES)


def pytest_model_cases(backend: str, *, model_cases: tuple[WeightUpdateModelCase, ...] = MODEL_CASES) -> list[Any]:
    """Select model lanes and full transport coverage on a representative."""
    import pytest

    if backend not in ("npu_ipc", "hccl"):
        raise ValueError(f"Unsupported weight transfer backend: {backend}")
    return [
        pytest.param(
            case,
            packed,
            id=f"{'packed' if packed else 'unpacked'}-{case.id}",
            marks=pytest.mark.e2e_model(case.model),
        )
        for packed in (False, True)
        for case in model_cases
        if (backend == "npu_ipc" and not packed) or case.full_transfer_matrix
    ]
