# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ascend Qwen4 exports.

[Role] Resolve model registry exports without loading models during cache planning.
[Invariants] Cache configuration imports must not register model custom operators.
[Timeouts & Locks] Import machinery only; no waits or locks.
[Side-Effect Policy] Model adaptation is imported only when a model export is requested.
"""

from importlib import import_module

__all__ = [
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpMTP",
    "AscendQwen4ExpModel",
    "AscendQwen4ExpMultiTokenPredictor",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".mtp" if name in ("AscendQwen4ExpMTP", "AscendQwen4ExpMultiTokenPredictor") else ".model"
    return getattr(import_module(module, __name__), name)
