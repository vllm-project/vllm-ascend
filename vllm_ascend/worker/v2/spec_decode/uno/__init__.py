# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""UNO shared-model speculative decoding on Ascend.

The package root stays import-light so ``uno.lora`` and ``uno.prepare`` can be
unit-tested without pulling in the speculator's NPU-only imports.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.uno.speculator import AscendUnoSpeculator

__all__ = ["AscendUnoSpeculator"]


def __getattr__(name: str):
    if name == "AscendUnoSpeculator":
        from vllm_ascend.worker.v2.spec_decode.uno.speculator import (
            AscendUnoSpeculator,
        )

        return AscendUnoSpeculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
