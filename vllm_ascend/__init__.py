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
from pathlib import Path
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


# main2main compat: an in-place vllm upgrade can leave the old
# ``vllm/multimodal/cache.py`` module on disk next to the new
# ``vllm/multimodal/cache/`` package (or vice versa). A directory shadows a
# module of the same name, so whichever form is authoritative must win. On
# the 0.29.0 release the module is ``cache.py``; on vLLM main the package is
# authoritative and a stale ``cache.py`` must be ignored.
class _MultimodalCacheModuleFinder:
    """Resolve ``vllm.multimodal.cache`` to the authoritative form.

    The 0.29.0 release uses ``cache.py``; a stale ``cache/`` package from an
    in-place upgrade would shadow it. The version check is deferred to
    ``find_spec`` because importing ``vllm_ascend.utils`` while ``vllm_ascend``
    itself is being imported by vLLM's platform discovery breaks device type
    inference.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm.multimodal.cache":
            return None
        from vllm_ascend.utils import vllm_version_is

        if not vllm_version_is("0.29.0"):
            return None
        try:
            import vllm
        except Exception:
            return None
        if not vllm.__file__:
            return None
        multimodal_dir = Path(vllm.__file__).parent / "multimodal"
        cache_py = multimodal_dir / "cache.py"
        if not cache_py.is_file() or not (multimodal_dir / "cache").is_dir():
            return None
        return importlib.util.spec_from_file_location(fullname, cache_py)


if not any(isinstance(_finder, _MultimodalCacheModuleFinder) for _finder in sys.meta_path):
    sys.meta_path.insert(0, _MultimodalCacheModuleFinder())


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
