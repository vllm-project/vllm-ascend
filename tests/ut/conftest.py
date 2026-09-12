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
    torch_npu.npu_fusion_attention = MagicMock()  # type: ignore[attr-defined]
    torch_npu.npu_format_cast = MagicMock(side_effect=lambda weight, fmt, **kwargs: weight)  # type: ignore[attr-defined]
    torch_npu._C = MagicMock()  # type: ignore[attr-defined]
    torch_npu._C._NPUTaskGroupHandle = MagicMock
    # Note: Assign missing attributes with values from real scenarios
    torch_npu.float4_e2m1fn_x2 = 296  # type: ignore[attr-defined]
    torch_npu.hifloat8 = 290  # type: ignore[attr-defined]
    torch_npu.float8_e8m0fnu = 293  # type: ignore[attr-defined]
    sys.modules["torch_npu"] = torch_npu
    sys.modules["torch_npu._C"] = torch_npu._C
    sys.modules["torch_npu._C._distributed_c10d"] = torch_npu._C._distributed_c10d
    # worker.py imports: from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
    atb_ops = types.ModuleType("torch_npu.op_plugin.atb._atb_ops")
    atb_ops.__spec__ = importlib.util.spec_from_loader("torch_npu.op_plugin.atb._atb_ops", loader=None)
    atb_ops._register_atb_extensions = MagicMock()  # type: ignore[attr-defined]
    atb_mod = types.ModuleType("torch_npu.op_plugin.atb")
    atb_mod.__spec__ = importlib.util.spec_from_loader("torch_npu.op_plugin.atb", loader=None)
    atb_mod.__path__ = []  # type: ignore[attr-defined]
    atb_mod._atb_ops = atb_ops  # type: ignore[attr-defined]
    op_plugin = types.ModuleType("torch_npu.op_plugin")
    op_plugin.__spec__ = importlib.util.spec_from_loader("torch_npu.op_plugin", loader=None)
    op_plugin.__path__ = []  # type: ignore[attr-defined]
    op_plugin.atb = atb_mod  # type: ignore[attr-defined]
    torch_npu.op_plugin = op_plugin  # type: ignore[attr-defined]
    sys.modules["torch_npu.op_plugin"] = op_plugin
    sys.modules["torch_npu.op_plugin.atb"] = atb_mod
    sys.modules["torch_npu.op_plugin.atb._atb_ops"] = atb_ops
    # worker.py: from torch_npu.profiler import dynamic_profile as dp
    # Keep a real ModuleType so `from torch_npu.profiler import ...` works, but
    # pre-seed profiler attrs so @patch(..., create=False) on CPU UTs succeeds.
    profiler_mod = types.ModuleType("torch_npu.profiler")
    profiler_mod.__spec__ = importlib.util.spec_from_loader("torch_npu.profiler", loader=None)
    profiler_mod.dynamic_profile = MagicMock()  # type: ignore[attr-defined]
    for _profiler_attr in (
        "_ExperimentalConfig",
        "profile",
        "tensorboard_trace_handler",
        "ExportType",
        "ProfilerLevel",
        "AiCMetrics",
        "ProfilerActivity",
    ):
        setattr(profiler_mod, _profiler_attr, MagicMock())
    # Seed enum-like attrs so production code can read .Text / .CPU without patches.
    profiler_mod.ExportType.Text = "Text"  # type: ignore[attr-defined]
    profiler_mod.ProfilerLevel.Level1 = "Level1"  # type: ignore[attr-defined]
    profiler_mod.AiCMetrics.PipeUtilization = "PipeUtilization"  # type: ignore[attr-defined]
    profiler_mod.ProfilerActivity.CPU = "CPU"  # type: ignore[attr-defined]
    profiler_mod.ProfilerActivity.NPU = "NPU"  # type: ignore[attr-defined]
    torch_npu.profiler = profiler_mod  # type: ignore[attr-defined]
    sys.modules["torch_npu.profiler"] = profiler_mod
    torch_npu._npu_matmul_add_fp32 = MagicMock()  # type: ignore[attr-defined]
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

    class _NpuStreamStub:
        """Minimal stream stub so record_event / wait_* work in CPU UTs."""

        def __init__(self, *args, **kwargs):
            self.npu_stream = 0

        def record_event(self, *args, **kwargs):
            return MagicMock(name="npu_event")

        def wait_event(self, *args, **kwargs):
            return None

        def wait_stream(self, *args, **kwargs):
            return None

        def synchronize(self):
            return None

    _default_npu_stream = _NpuStreamStub()

    torch.npu = MagicMock()
    torch.npu.is_available = MagicMock(return_value=False)
    torch.npu.Stream = _NpuStreamStub
    torch.npu.Event = MagicMock
    torch.npu.current_stream = MagicMock(return_value=_default_npu_stream)
    torch.npu.set_device = MagicMock()
    torch.npu.set_stream = MagicMock()
    torch.npu.synchronize = MagicMock()
    # Use "cpu" so torch.empty / Tensor.new_empty(device=...) work on CPU runners.
    # Returning an int (e.g. 0) is treated as a CUDA/NPU ordinal and breaks tensor alloc.
    torch.npu.current_device = MagicMock(return_value="cpu")
    torch.npu.device_count = MagicMock(return_value=0)
    torch.npu.empty_cache = MagicMock()
    torch.npu.mem_get_info = MagicMock(return_value=(0, 0))
    torch.npu.memory_stats = MagicMock(return_value={"allocated_bytes.all.peak": 0})
    torch.npu.reset_peak_memory_stats = MagicMock()
    torch.npu.max_memory_allocated = MagicMock(return_value=0)
    torch.npu.get_device_name = MagicMock(return_value="Ascend910B")
    torch.npu.get_device_properties = MagicMock()
    torch.npu.get_device_properties.return_value = MagicMock(uuid="00000000-0000-0000-0000-000000000000")
    torch.npu.graph_task_update_begin = MagicMock()
    torch.npu.graph_task_update_end = MagicMock()
    torch.npu.stream = MagicMock()
    # Some code paths do `import torch.npu`; attribute assignment alone is not enough.
    sys.modules["torch.npu"] = torch.npu
    torch_npu.npu.Stream = _NpuStreamStub  # type: ignore[attr-defined]
    torch_npu.npu.current_stream = MagicMock(return_value=_default_npu_stream)  # type: ignore[attr-defined]
    torch_npu.npu.current_device = MagicMock(return_value="cpu")  # type: ignore[attr-defined]
    torch_npu.npu.set_device = torch.npu.set_device  # type: ignore[attr-defined]
    torch_npu.npu.stream = MagicMock()  # type: ignore[attr-defined]
    torch.version.cann = None
    torch.distributed.is_hccl_available = MagicMock(return_value=True)

import pytest

mooncake_engine = types.ModuleType("mooncake.engine")
mooncake_engine.__spec__ = importlib.util.spec_from_loader("mooncake.engine", loader=None)
mooncake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("mooncake.engine", mooncake_engine)

build_info = types.ModuleType("vllm_ascend._build_info")
build_info.__spec__ = importlib.util.spec_from_loader("vllm_ascend._build_info", loader=None)
setattr(build_info, "__device_type__", "A2")  # noqa: B010
sys.modules.setdefault("vllm_ascend._build_info", build_info)

from vllm_ascend.utils import (  # noqa: E402
    adapt_patch,
    clear_enable_sp,
    enable_custom_op,
    register_ascend_customop,
)

# Mock torch_npu AFTER vllm_ascend import to avoid circular import in accelerate
if not _npu_available:
    sys.modules["torch_npu"].npu.current_device = MagicMock(return_value="cpu")
    sys.modules["torch_npu._inductor"] = MagicMock()
    sys.modules["torch_npu"]._npu_flash_attention = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"]._npu_paged_attention_splitfuse = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"]._npu_reshape_and_cache = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_scatter_pa_kv_cache = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_gather_pa_kv_cache = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_attention_update = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_fused_infer_attention_score = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_fused_infer_attention_score_v2 = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"]._npu_paged_attention = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"]._npu_paged_attention_get_workspace = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"]._npu_fused_infer_attention_score_get_max_workspace = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_moe_gating_top_k_softmax = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_moe_token_permute = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_moe_token_unpermute = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_moe_init_routing_v2 = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_quant_matmul = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_rms_norm = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_interleave_rope = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_dynamic_block_quant = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_dynamic_quant = MagicMock(  # type: ignore[attr-defined]
        return_value=(MagicMock(), MagicMock())
    )
    sys.modules["torch_npu"].npu_dynamic_mx_quant = MagicMock(  # type: ignore[attr-defined]
        return_value=(MagicMock(), MagicMock())
    )
    sys.modules["torch_npu"].npu_mla_prolog_v3 = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_kv_rmsnorm_rope_cache = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_swiglu = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_fast_gelu = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_convert_weight_to_int4pack = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_transpose_batchmatmul = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_scatter_nd_update_ = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_grouped_matmul = MagicMock(return_value=[MagicMock()])  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_clipped_swiglu = MagicMock()  # type: ignore[attr-defined]
    sys.modules["torch_npu"].npu_grouped_matmul_swiglu_quant_v2 = MagicMock(  # type: ignore[attr-defined]
        return_value=(MagicMock(), MagicMock())
    )

adapt_patch()
adapt_patch(True)

# register Ascend CustomOp here because uts will use this
register_ascend_customop()

if not _npu_available:
    import torch

    from tests.ut.helpers.golden_copy_and_expand import npu_copy_and_expand_eagle_inputs_stub

    enable_custom_op()
    if hasattr(torch.ops, "_C_ascend") and not hasattr(torch.ops._C_ascend, "npu_copy_and_expand_eagle_inputs"):
        torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs = npu_copy_and_expand_eagle_inputs_stub
    # Re-sync after enable_custom_op / adapt_patch so @patch("torch.npu.*") hits
    # the same object production code uses via `torch.npu`.
    torch.npu.current_device = MagicMock(return_value="cpu")
    sys.modules["torch.npu"] = torch.npu

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


@pytest.fixture(autouse=True)
def _reset_stream_globals_before_test():
    """Avoid cross-test leakage from utils.current_stream() caching."""
    import vllm_ascend.ops.fused_moe.moe_utils as moe_utils_mod
    import vllm_ascend.utils as utils_mod

    utils_mod._CURRENT_STREAM = None
    utils_mod._GLOBAL_STREAM = None
    if hasattr(utils_mod, "_SHARED_EXPERTS_CALCULATION_STREAM"):
        utils_mod._SHARED_EXPERTS_CALCULATION_STREAM = None
    moe_utils_mod.COMM_STREAM = None
    yield
    utils_mod._CURRENT_STREAM = None
    utils_mod._GLOBAL_STREAM = None
    if hasattr(utils_mod, "_SHARED_EXPERTS_CALCULATION_STREAM"):
        utils_mod._SHARED_EXPERTS_CALCULATION_STREAM = None
    moe_utils_mod.COMM_STREAM = None


@pytest.fixture(autouse=True)
def _mock_ascend_store_deps(request):
    # ascend_store code imports vllm_ascend helpers (AttentionComputeStartGate,
    # get/reset_attention_compute_start_gate, ...) which _mock_deps.py no longer
    # mocks globally (mutating the real modules leaked into other UTs). Mock them
    # per-test, scoped to the ascend_store tests only.
    if "distributed/ascend_store/" not in request.node.nodeid:
        yield
        return
    from unittest.mock import patch

    _pfx = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
    with (
        patch(f"{_pfx}.pool_worker.get_attention_compute_start_gate"),
        patch(f"{_pfx}.pool_worker.reset_attention_compute_start_gate"),
        patch(f"{_pfx}.metadata.AttentionComputeStartGate", type("AttentionComputeStartGate", (), {})),
    ):
        yield
