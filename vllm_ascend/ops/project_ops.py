# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy resolver for vLLM-Ascend csrc Torch operators."""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch


@lru_cache
def get_project_op(name: str):
    """Resolve an operator after the worker has selected its NPU device."""
    try:
        return getattr(torch.ops._C_ascend, name)
    except AttributeError:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    try:
        return getattr(torch.ops._C_ascend, name)
    except AttributeError as exc:
        raise RuntimeError(f"vLLM-Ascend csrc operator _C_ascend::{name} is unavailable") from exc
