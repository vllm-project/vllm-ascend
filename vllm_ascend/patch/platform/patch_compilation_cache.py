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

from __future__ import annotations

import os
from typing import Any

import regex as re
import torch
import vllm.compilation.backends as vllm_backends
import vllm.compilation.decorators as vllm_decorators

_RANK_DIRECTORY_PATTERN = re.compile(r"^rank_\d+_\d+$")
_DEVICE_DIRECTORY_PATTERN = re.compile(r"^dev(\d+)$")
_COMPILE_CACHE_DIRECTORIES = ("torch_aot_compile", "torch_compile_cache")
_DEVICE_CACHE_PATCH_MARKER = "_vllm_ascend_device_isolated_compile_cache"


def _current_npu_device_index() -> int | None:
    try:
        if not torch.npu.is_available():
            return None
        return int(torch.npu.current_device())
    except (AttributeError, RuntimeError, TypeError):
        # CPU-only unit tests and environments without torch_npu do not have
        # an NPU device. In that case, leave upstream paths intact.
        return None


def _device_isolated_cache_path(path: str | None) -> str | None:
    """Insert the current NPU index into a vLLM per-rank compile cache path.

    A ``rank_{rank}_{dp_rank}`` directory is not sufficient on NPU when the
    same rank is launched with different process/device mappings. For example,
    an externally launched DP worker can expose only two NPUs, while internal
    DP exposes all NPUs; the same rank then maps to a different logical NPU.
    AOT artifacts can retain buffers on the compile-time NPU, so reusing the
    old artifact makes runtime tensors and compiled constants land on
    different devices.
    """

    if path is None:
        return None

    device_index = _current_npu_device_index()
    if device_index is None:
        return path

    parts = path.split(os.sep)
    for index, part in enumerate(parts):
        if _RANK_DIRECTORY_PATTERN.fullmatch(part) is None:
            continue
        if not any(directory in parts[: index + 1] for directory in _COMPILE_CACHE_DIRECTORIES):
            continue
        if index + 1 < len(parts):
            device_match = _DEVICE_DIRECTORY_PATTERN.fullmatch(parts[index + 1])
            if device_match is not None:
                if int(device_match.group(1)) == device_index:
                    return path
                parts[index + 1] = f"dev{device_index}"
                return os.sep.join(parts)

        return os.sep.join(parts[: index + 1] + [f"dev{device_index}", *parts[index + 1 :]])

    return path


_original_try_load_aot_compiled_fn = vllm_decorators._try_load_aot_compiled_fn


def _try_load_aot_compiled_fn(model: Any, aot_compilation_path: str) -> Any | None:
    return _original_try_load_aot_compiled_fn(
        model,
        _device_isolated_cache_path(aot_compilation_path),
    )


vllm_decorators._try_load_aot_compiled_fn = _try_load_aot_compiled_fn


_original_support_torch_compile = vllm_decorators._support_torch_compile


def _support_torch_compile(cls: type, *args: Any, **kwargs: Any) -> type:
    compiled_cls = _original_support_torch_compile(cls, *args, **kwargs)
    if _DEVICE_CACHE_PATCH_MARKER in compiled_cls.__dict__:
        return compiled_cls

    original_save_aot_compiled_function = compiled_cls.save_aot_compiled_function

    def save_aot_compiled_function(self: Any) -> None:
        self._aot_compilation_path = _device_isolated_cache_path(self._aot_compilation_path)
        self._aot_cache_dir = _device_isolated_cache_path(self._aot_cache_dir)
        original_save_aot_compiled_function(self)

    compiled_cls.save_aot_compiled_function = save_aot_compiled_function
    setattr(compiled_cls, _DEVICE_CACHE_PATCH_MARKER, True)
    return compiled_cls


vllm_decorators._support_torch_compile = _support_torch_compile


_original_compiler_manager_initialize_cache = vllm_backends.CompilerManager.initialize_cache


def _compiler_manager_initialize_cache(
    self: vllm_backends.CompilerManager,
    cache_dir: str,
    disable_cache: bool = False,
    prefix: str = "",
) -> None:
    device_cache_dir = _device_isolated_cache_path(cache_dir)
    self.compilation_config.local_cache_dir = device_cache_dir
    _original_compiler_manager_initialize_cache(
        self,
        cache_dir=device_cache_dir,
        disable_cache=disable_cache,
        prefix=prefix,
    )


vllm_backends.CompilerManager.initialize_cache = _compiler_manager_initialize_cache
