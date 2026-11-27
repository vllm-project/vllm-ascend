# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
import importlib
from unittest.mock import patch

warmup = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")


def test_disabled_warmup_never_imports_triton_implementations():
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops.triton"):
            raise AssertionError(f"Disabled warmup imported {name}")
        return original_import(name, *args, **kwargs)

    with (
        patch.object(warmup, "supports_triton", return_value=False),
        patch("builtins.__import__", side_effect=guarded_import),
    ):
        # Capability rejection must precede both configuration reads and imports.
        warmup.kernel_warmup(object())
