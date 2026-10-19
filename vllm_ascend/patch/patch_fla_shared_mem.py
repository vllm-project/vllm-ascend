# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Avoid NPU context creation while FLA builds import-time autotune configs.

The supported Triton Ascend backend reports a placeholder max_shared_mem=1.
It nevertheless queries NPU core counts to build its property dictionary,
which initializes the caller's NPU context. FLA's CUDA shared-memory capacity
check is always false on this backend, so answer it without querying hardware.

This hook must run before ops/__init__.py imports chunk_o. Patching utils after
an ordinary import is too late: importing the parent package runs that query.
Remove this compatibility patch when FLA/Triton expose a context-free query.
"""

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType

_FLA_UTILS_MODULE = "vllm.third_party.flash_linear_attention.ops.utils"


def _npu_check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    return False


def _patch_fla_utils(module: ModuleType) -> None:
    if getattr(module, "device", None) == "npu":
        module.check_shared_mem = _npu_check_shared_mem


class _FLAUtilsLoader(importlib.abc.Loader):
    def __init__(self, loader, finder):
        self.loader = loader
        self.finder = finder

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch_fla_utils(module)
        # Import locking serializes this module's execution. The hook is only
        # needed once per interpreter and never runs in the inference hot path.
        if self.finder in sys.meta_path:
            sys.meta_path.remove(self.finder)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class _FLAUtilsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _FLA_UTILS_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _FLAUtilsLoader(spec.loader, self)
        return spec


def install_fla_shared_mem_patch() -> None:
    """Install before platform patches, model resolution and CustomOp imports.

    The only process-local mutable state is the one-shot sys.meta_path entry.
    Repeated plugin registration is idempotent; spawn children install their
    own hook by importing vllm_ascend, without importing FLA eagerly.
    """
    module = sys.modules.get(_FLA_UTILS_MODULE)
    if module is not None:
        _patch_fla_utils(module)
    elif not any(isinstance(finder, _FLAUtilsFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FLAUtilsFinder())
