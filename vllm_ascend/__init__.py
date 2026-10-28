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

    # torch_npu's Inductor cache still imports triton_key from the old path.
    # Alias the real function before compilation, including in spawned workers.
    # Remove once supported torch_npu versions import it from runtime.cache.
    try:
        import triton.compiler.compiler as _triton_compiler  # type: ignore[import-untyped]
    except ImportError:
        pass  # Triton may be unavailable in CPU UT environments.
    else:
        if not hasattr(_triton_compiler, "triton_key"):
            from triton.runtime.cache import triton_key as _triton_key  # type: ignore[import-untyped]

            _triton_compiler.triton_key = _triton_key

        # Triton 3.6 moved launch hooks to knobs.runtime. Keep torch_npu's
        # legacy class reads live, including hooks installed after import.
        # Remove once supported torch_npu launchers read knobs.runtime.
        try:
            from triton import knobs as _triton_knobs  # type: ignore[import-untyped]
        except ImportError:
            pass  # Older Triton versions retain the original class attributes.
        else:

            class _TritonLaunchHook:
                def __init__(self, name: str) -> None:
                    self.name = name

                def __get__(self, instance, owner=None):
                    return getattr(_triton_knobs.runtime, self.name)

            for _hook_name in ("launch_enter_hook", "launch_exit_hook"):
                if not hasattr(_triton_compiler.CompiledKernel, _hook_name):
                    setattr(_triton_compiler.CompiledKernel, _hook_name, _TritonLaunchHook(_hook_name))

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

    from vllm_ascend.mrv2_utils import apply_v2_model_runner_config_patch
    from vllm_ascend.utils import adapt_patch

    apply_v2_model_runner_config_patch()
    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the NPU platform."""

    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.ec_transfer import (
        register_connector as register_ec_connector,
    )
    from vllm_ascend.distributed.kv_transfer import (
        register_connector as register_kv_connector,
    )
    from vllm_ascend.distributed.weight_transfer import register_engine

    register_kv_connector()
    register_ec_connector()
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
