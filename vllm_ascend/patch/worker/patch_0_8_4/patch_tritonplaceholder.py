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
# Adapted from vllm/triton_utils/importing.py
#

import importlib
import sys
import types
from importlib.util import find_spec

from vllm.logger import logger

HAS_TRITON = (
    find_spec("triton") is not None
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
)

if not HAS_TRITON:
    logger.info("Triton not installed or not compatible; certain GPU-related"
                " functions will not be available.")

    class TritonPlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton")
            self.jit = self._dummy_decorator("jit")
            self.autotune = self._dummy_decorator("autotune")
            self.heuristics = self._dummy_decorator("heuristics")
            self.language = TritonLanguagePlaceholder()
            self.__spec__ = importlib.machinery.ModuleSpec(
                name="triton", loader=None, origin="placeholder")
            logger.warning_once(
                "Triton is not installed. Using dummy decorators. "
                "Install it via `pip install triton` to enable kernel"
                " compilation.")

        def _dummy_decorator(self, name):

            def decorator(func=None, **kwargs):
                if func is None:
                    return lambda f: f
                return func

            return decorator

    class TritonLanguagePlaceholder(types.ModuleType):

        def __init__(self):
            super().__init__("triton.language")
            self.constexpr = None
            self.dtype = None

    sys.modules['triton'] = TritonPlaceholder()
    sys.modules['triton.language'] = TritonLanguagePlaceholder()

if 'triton' in sys.modules:
    logger.info("Triton module has been replaced with a placeholder.")
