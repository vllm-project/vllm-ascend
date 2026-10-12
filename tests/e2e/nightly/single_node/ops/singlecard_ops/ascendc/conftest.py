# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import pytest
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env


@pytest.fixture(scope="module", autouse=True)
def load_local_custom_ops():
    """Load the local binding explicitly, including on A5."""
    torch_npu.npu.config.allow_internal_format = True
    bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")
