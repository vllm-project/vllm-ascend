# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from datetime import datetime

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

init_device_properties_triton()
enable_custom_op()


@pytest.fixture(autouse=True)
def restore_singlecard_default_device():
    device = torch.get_default_device()
    try:
        yield
    finally:
        try:
            if torch.npu.is_initialized():
                torch.npu.synchronize()
        finally:
            # None also removes the default-device dispatch mode when restoring CPU.
            torch.set_default_device(None if device.type == "cpu" else device)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to add timestamp to test reports"""
    start_time = datetime.now().strftime("[%H:%M:%S]")

    outcome = yield

    report = outcome.get_result()

    if report.when == "call":
        print(f"{start_time}")
