# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker-local loading of NPU DeepGEMM; never fall back to CUDA binaries."""

import ast
import importlib
import importlib.util
import inspect
import logging
import os
import sys
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)
_BUNDLED = "vllm_ascend.third_party.deep_gemm"
_REQUIRED_APIS = ("get_npu_arch", "fp8_fp4_mega_moe", "transform_weights_for_mega_moe", "SymmBuffer")


def _is_npu_package(spec):
    """Recognize the upstream NPU API before importing a native extension.

    Both distributions use the name deep_gemm. Its upstream NPU initializer
    explicitly re-exports get_npu_arch from ._C; CUDA's initializer does not.
    Unknown layouts are not loaded speculatively into the worker process.
    """
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return False
    try:
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False
    return any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "_C"
        and any(alias.name == "get_npu_arch" for alias in node.names)
        for node in tree.body
    )


def _validate(module):
    missing = [name for name in _REQUIRED_APIS if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"NPU DeepGEMM is missing required APIs: {', '.join(missing)}")
    if "base" not in inspect.signature(module.SymmBuffer).parameters:
        raise RuntimeError("NPU DeepGEMM requires SymmBuffer(base=...) for target/draft storage reuse")


@cache
def load_deepgemm():
    """Resolve once per worker, before weights are transformed or buffers allocated."""
    import torch_npu  # noqa: F401 -- load the NPU shared libraries before the extension
    from vllm import envs as vllm_envs

    # This is the dependency's existing cache setting, with the vLLM default.
    # It is not a new user configuration requirement.
    os.environ.setdefault("DG_JIT_CACHE_DIR", str(Path(vllm_envs.VLLM_CACHE_ROOT) / "deep_gemm" / "ascend"))
    loaded = sys.modules.get("deep_gemm")
    if loaded is not None and not callable(getattr(loaded, "get_npu_arch", None)):
        raise RuntimeError("A non-NPU deep_gemm is already loaded; cannot load a second native backend in this worker")
    external = None if loaded is not None else importlib.util.find_spec("deep_gemm")
    name = "deep_gemm" if loaded is not None or _is_npu_package(external) else _BUNDLED
    if external is not None and name == _BUNDLED:
        logger.warning("Ignoring external deep_gemm without the upstream NPU API; using bundled Ascend backend")
    try:
        module = importlib.import_module(name)
        _validate(module)
    except Exception as exc:
        # Native registration may already have happened. Never attempt a second
        # native implementation after an initialization or API validation error.
        raise RuntimeError(
            f"Cannot initialize NPU MegaMoE backend {name}. Install the matching Ascend 950 "
            "vLLM Ascend distribution and its supported platform dependencies. "
            "No CUDA DeepGEMM fallback is supported."
        ) from exc
    logger.info("Loaded NPU DeepGEMM %s from %s", getattr(module, "__version__", "unknown"), module.__file__)
    return module
