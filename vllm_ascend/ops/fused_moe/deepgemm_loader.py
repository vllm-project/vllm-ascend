# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Load the NPU backend built and packaged with this vLLM Ascend installation."""

import importlib
import logging
import os
import sys
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)
_BUNDLED = "vllm_ascend.third_party.deep_gemm"


@cache
def load_deepgemm():
    """Load once per worker; the build pins the native API and JIT headers."""
    import torch_npu  # noqa: F401 -- load NPU shared libraries before the extension
    from vllm import envs as vllm_envs

    # Both forks register global pybind11 types. Do not load two extensions
    # into one process, even if the other package also targets NPU.
    if "deep_gemm" in sys.modules:
        raise RuntimeError("An external deep_gemm is already loaded; MegaMoE requires the bundled Ascend backend")
    os.environ.setdefault("DG_JIT_CACHE_DIR", str(Path(vllm_envs.VLLM_CACHE_ROOT) / "deep_gemm" / "ascend"))
    try:
        module = importlib.import_module(_BUNDLED)
    except ModuleNotFoundError as exc:
        if exc.name is None or not _BUNDLED.startswith(exc.name):
            raise
        raise RuntimeError(
            "NPU MegaMoE was not built. Reinstall vLLM Ascend with its Ascend 950 native backend."
        ) from exc
    logger.info("Loaded bundled NPU DeepGEMM %s from %s", module.__version__, module.__file__)
    return module
