# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker-local loading of NPU DeepGEMM; never fall back to CUDA binaries."""

import importlib
import inspect
import logging
import os
from functools import cache
from pathlib import Path

from vllm_ascend.ascend_config import is_deepgemm_available

logger = logging.getLogger(__name__)
_REQUIRED_APIS = ("get_npu_arch", "fp8_fp4_mega_moe", "transform_weights_for_mega_moe", "SymmBuffer")


def _validate(module):
    missing = [name for name in _REQUIRED_APIS if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"NPU DeepGEMM is missing required APIs: {', '.join(missing)}")
    if "base" not in inspect.signature(module.SymmBuffer).parameters:
        raise RuntimeError("NPU DeepGEMM requires SymmBuffer(base=...) for target/draft storage reuse")


@cache
def load_deepgemm():
    """Load the separately installed Ascend wheel once, in workers only."""
    if not is_deepgemm_available():
        raise RuntimeError("Install the matching deepgemm_ascend wheel to use NPU MegaMoE")

    import torch_npu  # noqa: F401 -- load the NPU shared libraries before the extension
    from vllm import envs as vllm_envs

    os.environ.setdefault("DG_JIT_CACHE_DIR", str(Path(vllm_envs.VLLM_CACHE_ROOT) / "deep_gemm" / "ascend"))
    try:
        module = importlib.import_module("deep_gemm")
        _validate(module)
    except Exception as exc:
        raise RuntimeError(
            "Cannot initialize NPU MegaMoE backend deep_gemm. Check that the deepgemm_ascend wheel "
            "matches Python, Torch, CANN and the Ascend platform. No CUDA DeepGEMM fallback is supported."
        ) from exc
    logger.info("Loaded NPU DeepGEMM %s from %s", getattr(module, "__version__", "unknown"), module.__file__)
    return module
