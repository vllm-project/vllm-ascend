#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import importlib.util
import sys
from types import ModuleType
from typing import Any

_triton_available = importlib.util.find_spec("triton") is not None

try:
    import triton.experimental.gluon  # type: ignore[import-untyped]  # noqa: F401
except (ImportError, ModuleNotFoundError):
    # fallback: create stubs for old triton-ascend
    if "triton.experimental" not in sys.modules:
        _experimental = ModuleType("triton.experimental")
        _experimental.__path__ = []
        sys.modules["triton.experimental"] = _experimental
    for _gluon_stub in (
        "triton.experimental.gluon",
        "triton.experimental.gluon.language",
    ):
        if _gluon_stub not in sys.modules:
            sys.modules[_gluon_stub] = ModuleType(_gluon_stub)

# main2main compat: `_aggregate` was added to triton.language.core in
# vllm main post-0.26.0. Stub it here so vllm.triton_utils can import it
# without breaking on triton-ascend 3.2.1. Skip if triton is not
# installed at all (e.g. 310P or CPU-UT environments).
if _triton_available:
    try:
        import triton.language.core as _tl_core  # type: ignore[import-untyped]
    except Exception:
        pass
    else:
        if not hasattr(_tl_core, "_aggregate"):
            _tl_core._aggregate = lambda *a, **kw: None


def _stub_triton_placeholder_runtime() -> None:
    """Give vLLM's ``TritonPlaceholder`` the attributes import paths read.

    When Triton is installed without an active driver (CPU UT or an
    accelerator-less environment), ``vllm.triton_utils`` exposes a
    ``TritonPlaceholder`` instead of the real module. vLLM main's
    ``_is_autotuned`` reads ``triton.runtime.autotuner.Autotuner`` while
    decorating kernels at import time, which the placeholder does not define.
    vLLM main also evaluates ``tl.constexpr(...)`` at import time (DeepSeek
    V4.1 DSA sparse MQA-logits kernels), while the language placeholder leaves
    ``constexpr`` as ``None``.
    """
    try:
        import vllm.triton_utils as _triton_utils
        from vllm.triton_utils.importing import TritonLanguagePlaceholder, TritonPlaceholder
    except Exception:
        return

    tl = getattr(_triton_utils, "tl", None)
    if isinstance(tl, TritonLanguagePlaceholder) and getattr(tl, "constexpr", None) is None:
        tl.constexpr = lambda *args, **kwargs: args[0] if args else None  # type: ignore[assignment]

    # The placeholder is used only when Triton is disabled; a real Triton
    # module already provides ``runtime``.
    if hasattr(TritonPlaceholder, "runtime"):
        return

    class _Autotuner:
        pass

    runtime: Any = ModuleType("triton.runtime")
    runtime.__path__ = []
    autotuner: Any = ModuleType("triton.runtime.autotuner")
    autotuner.Autotuner = _Autotuner
    runtime.autotuner = autotuner
    TritonPlaceholder.runtime = runtime


def _stub_cpu_gpu_buffer_pinmemory() -> None:
    """Keep ``CpuGpuBuffer.copy_to_gpu`` from re-pinning on Ascend.

    vLLM main calls ``cpu.pin_memory()`` on every ``copy_to_gpu``. Ascend
    reports pinned memory available but ``.pin_memory()`` needs the
    PrivateUse1 hooks that CPU-only environments do not register, so copy
    straight from the buffer's own CPU storage instead (already pinned when
    the buffer was created with pinning enabled).
    """
    try:
        import vllm.v1.utils as _v1_utils
    except Exception:
        return
    buffer_cls = getattr(_v1_utils, "CpuGpuBuffer", None)
    if buffer_cls is None or getattr(buffer_cls.copy_to_gpu, "_vllm_ascend_nopin", False):
        return

    def copy_to_gpu(self, n: int | None = None):
        cpu, gpu = self.cpu, self.gpu
        if n is not None:
            cpu, gpu = cpu[:n], gpu[:n]
        return gpu.copy_(cpu, non_blocking=True)

    copy_to_gpu._vllm_ascend_nopin = True  # type: ignore[attr-defined]
    buffer_cls.copy_to_gpu = copy_to_gpu  # type: ignore[method-assign]


_GLOBAL_PATCH_APPLIED = False


def _ensure_global_patch():
    """Apply process-wide vLLM patches before engine-core initialization.

    vLLM loads general plugins in engine-core subprocesses. E2E test
    conftest hooks do not run there, so global patches that affect scheduler
    and engine code must also be applied through these plugin entry points.
    """
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector
    from vllm_ascend.distributed.weight_transfer import register_engine

    register_connector()
    register_engine()


def register_model_loader():
    _ensure_global_patch()

    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    from .models import register_model

    register_model()


import vllm_ascend.logger  # noqa: E402, F401
