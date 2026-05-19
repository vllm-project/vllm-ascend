#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import functools
import inspect
import subprocess
import sys
from enum import Enum
from unittest.mock import MagicMock

try:
    # Note: do not import torch here for cpu env, which will lead to circle import error.
    subprocess.run(["npu-smi", "info"], capture_output=True, check=True)
    _npu_available = True
except (subprocess.CalledProcessError, FileNotFoundError):
    _npu_available = False

if not _npu_available:
    triton_runtime = MagicMock()
    triton_runtime.driver.active.utils.get_device_properties.return_value = {
        "num_aic": 8,
        "num_vectorcore": 8,
    }
    sys.modules["triton.runtime"] = triton_runtime

from vllm_ascend.utils import adapt_patch  # noqa E402
from vllm_ascend.utils import register_ascend_customop  # noqa E402

# Mock torch_npu AFTER vllm_ascend import to avoid circular import in accelerate
if not _npu_available:
    sys.modules["torch_npu"].npu.current_device = MagicMock(return_value=0)
    sys.modules["torch_npu._inductor"] = MagicMock()

adapt_patch()
adapt_patch(True)

# register Ascend CustomOp here because uts will use this
register_ascend_customop()


class RunnerDeviceType(str, Enum):
    """Chip types — values match runner_label.json "chip" field exactly.

    Shared by:
      - tests/ut/conftest.py (npu_test decorator)
      - .github/workflows/scripts/determine_smart_e2e_scope.py (AST parser)
    """

    A2 = "a2"
    A3 = "a3"
    _310P = "310p"
    CPU = "cpu"


_NPU_TEST_REQUIREMENTS_ATTR = "_npu_test_requirements"
_NPU_TEST_WRAPPER_ATTR = "_npu_test_wrapper"


def npu_test(num_npus: int = 1, npu_type: str | RunnerDeviceType = RunnerDeviceType.A2):
    """Decorator that marks a test with NPU resource requirements.

    Can be applied to either a single test function/method or a test class.
    Function/method decorators can be stacked to mark a test as runnable on
    multiple NPU runner types.

    Serves two purposes, depending on the target:

      - **Function/method**: declarative metadata for the AST parser AND
        runtime gating. If the runner does not satisfy the declared
        requirements, the wrapper raises ``RuntimeError`` (fails loudly,
        not ``pytest.skip``) — a mismatch means routing or the runner
        environment is broken.
      - **Class**: declarative metadata only. The class is returned
        unchanged so pytest's standard class-based collection still works.
        CI routing (driven by the AST parser) is the single source of
        truth — runtime gating per method would be redundant.

    The AST parser in ``determine_smart_e2e_scope.py`` reads the decorator
    keyword arguments (num_npus, npu_type) to group tests by runner type.
    The parameter names and decorator name must stay in sync with the parser.

    Args:
        num_npus: Number of NPU devices required (default 1).
        npu_type: The NPU chip type required (default A2).
    """
    if not isinstance(npu_type, RunnerDeviceType):
        npu_type = RunnerDeviceType(npu_type)
    requirement = (num_npus, npu_type)

    def _format_requirements(requirements):
        return ", ".join(f"{req_type.value} x{req_num_npus}" for req_num_npus, req_type in requirements)

    def _wrap_callable(func):
        requirements = list(getattr(func, _NPU_TEST_REQUIREMENTS_ATTR, ()))
        if requirement not in requirements:
            requirements.append(requirement)
        setattr(func, _NPU_TEST_REQUIREMENTS_ATTR, tuple(requirements))

        if getattr(func, _NPU_TEST_WRAPPER_ATTR, False):
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            requirements = getattr(wrapper, _NPU_TEST_REQUIREMENTS_ATTR, (requirement,))
            npu_requirements = [req for req in requirements if req[1] != RunnerDeviceType.CPU]
            if not npu_requirements:
                return func(*args, **kwargs)
            # CI routes this test to a runner matching one of the declared
            # (npu_type, num_npus) requirements.
            # If the requirements are not met at runtime, the routing or the
            # runner environment is broken — fail loudly instead of skipping.
            if not _npu_available:
                raise RuntimeError(
                    f"NPU required but not available on this runner "
                    f"(test needs {_format_requirements(npu_requirements)}). "
                    "Check runner_label.json and the runner's NPU setup."
                )
            import torch  # noqa

            device_count = torch.npu.device_count()
            if not any(device_count >= req_num_npus for req_num_npus, _ in npu_requirements):
                raise RuntimeError(
                    "Insufficient NPUs on this runner: "
                    f"need {_format_requirements(npu_requirements)}, "
                    f"have {device_count}. Check runner_label.json."
                )
            return func(*args, **kwargs)

        setattr(wrapper, _NPU_TEST_REQUIREMENTS_ATTR, tuple(requirements))
        setattr(wrapper, _NPU_TEST_WRAPPER_ATTR, True)
        return wrapper

    def decorator(obj):
        # Class decoration is purely declarative — the AST parser handles
        # routing, and CI routing is the single source of truth. Returning
        # the class unchanged keeps pytest's class-based collection working.
        if inspect.isclass(obj):
            return obj
        return _wrap_callable(obj)

    return decorator
