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
"""Shared UT setup.

NPU vs CPU routing is determined by directory convention, not decorators.
See ``.github/workflows/scripts/select_tests.py`` and
``.github/workflows/scripts/test_config.yaml`` for the routing rules.

Conventions for UT directories:
    tests/ut/<module>/            -> CPU runner (default)
    tests/ut/<module>/a2/         -> A2 NPU x1
    tests/ut/<module>/a2_2/       -> A2 NPU x2
    tests/ut/<module>/a3_2/       -> A3 NPU x2
    tests/ut/<module>/a3_4/       -> A3 NPU x4
    tests/ut/<module>/310p/       -> 310P NPU x1
"""

import importlib.util
import subprocess
import sys
import types
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
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    torch_npu.__path__ = []
    torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
    torch_npu.profiler = MagicMock()  # type: ignore[attr-defined]
    torch_npu.npu_fusion_attention = MagicMock()  # type: ignore[attr-defined]
    torch_npu.npu_format_cast = MagicMock(side_effect=lambda weight, fmt: weight)  # type: ignore[attr-defined]
    torch_npu._C = MagicMock()  # type: ignore[attr-defined]
    torch_npu._C._NPUTaskGroupHandle = MagicMock
    sys.modules["torch_npu"] = torch_npu
    sys.modules["torch_npu._C"] = torch_npu._C
    sys.modules["torch_npu._C._distributed_c10d"] = torch_npu._C._distributed_c10d
    acl_rt = types.ModuleType("acl.rt")
    acl_rt.__spec__ = importlib.util.spec_from_loader("acl.rt", loader=None)
    acl_rt.memcpy = MagicMock()  # type: ignore[attr-defined]
    acl_mod = types.ModuleType("acl")
    acl_mod.__spec__ = importlib.util.spec_from_loader("acl", loader=None)
    acl_mod.rt = acl_rt  # type: ignore[attr-defined]
    sys.modules["acl"] = acl_mod
    sys.modules["acl.rt"] = acl_rt
    mooncake_engine = types.ModuleType("mooncake.engine")
    mooncake_engine.__spec__ = importlib.util.spec_from_loader("mooncake.engine", loader=None)
    mooncake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
    sys.modules["mooncake.engine"] = mooncake_engine
    import torch

    try:  # noqa: SIM105
        torch.utils.rename_privateuse1_backend("npu")
    except RuntimeError:
        pass
    torch.npu = MagicMock()
    torch.npu.Stream = MagicMock
    torch.version.cann = None
    torch.distributed.is_hccl_available = MagicMock(return_value=True)

import pytest

mooncake_engine = types.ModuleType("mooncake.engine")
mooncake_engine.__spec__ = importlib.util.spec_from_loader("mooncake.engine", loader=None)
mooncake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("mooncake.engine", mooncake_engine)

from vllm_ascend.utils import (  # noqa: E402
    adapt_patch,
    clear_enable_sp,
    register_ascend_customop,
)

# Mock torch_npu AFTER vllm_ascend import to avoid circular import in accelerate
if not _npu_available:
    sys.modules["torch_npu"].npu.current_device = MagicMock(return_value=0)
    sys.modules["torch_npu._inductor"] = MagicMock()
    sys.modules["torch_npu"]._npu_flash_attention = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"]._npu_paged_attention_splitfuse = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"]._npu_reshape_and_cache = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_moe_gating_top_k_softmax = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_quant_matmul = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_rms_norm = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_swiglu = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_convert_weight_to_int4pack = MagicMock()  # type: ignore[attr-defined]

adapt_patch()
adapt_patch(True)

# register Ascend CustomOp here because uts will use this
register_ascend_customop()

# Clean up any stale mock modules that may have been installed by
# other test files (e.g., ascend_store/_mock_deps.py) which replace
# real subpackages with MagicMock, breaking later imports.
_stale_modules = [
    k
    for k in sys.modules
    if k.startswith("vllm_ascend.distributed.kv_transfer.") and not isinstance(sys.modules[k], types.ModuleType)
]
for _m in _stale_modules:
    del sys.modules[_m]


@pytest.fixture(autouse=True)
def _clear_enable_sp_before_test():
    clear_enable_sp()
    yield
