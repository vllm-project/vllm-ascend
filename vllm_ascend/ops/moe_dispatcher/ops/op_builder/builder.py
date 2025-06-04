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

import os
from abc import ABC, abstractmethod
from typing import List, Union
from torch.utils.cpp_extension import load
from torch.library import Library
import torch_npu
import vllm_ascend

ASCEND_HOME_PATH = "ASCEND_HOME_PATH"
AS_LIBRARY = Library("vllm_ascend", "DEF")


class VllmAscendOpBuilder(ABC):
    _cann_path = None
    _torch_npu_path = None
    _cann_version = None
    _loaded_ops = {}

    def __init__(self, name):
        self.name = name
        self._cann_path = self.get_cann_path()
        self._torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))

    def get_cann_path(self):
        if ASCEND_HOME_PATH in os.environ and os.path.exists(os.environ[ASCEND_HOME_PATH]):
            return os.environ[ASCEND_HOME_PATH]
        return None

    def get_absolute_paths(self, paths):
        vllm_ascend_path = os.path.abspath(os.path.join(os.path.dirname(vllm_ascend.__file__), os.pardir))
        return [os.path.join(vllm_ascend_path, path) for path in paths]

    def register_op_proto(self, op_proto: Union[str, List[str]]):
        if isinstance(op_proto, str):
            op_proto = [op_proto]
        for proto in op_proto:
            AS_LIBRARY.define(proto)

    @abstractmethod
    def sources(self):
        ...

    def include_paths(self):
        paths = [
            os.path.join(self._torch_npu_path, 'include'),
            os.path.join(self._torch_npu_path, 'include/third_party/hccl/inc'),
            os.path.join(self._torch_npu_path, 'include/third_party/acl/inc'),
            os.path.join(self._cann_path, 'include'),
        ]
        return paths

    def cxx_args(self):
        args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
                '-s', '-fvisibility=hidden', '-D_FORTIFY_SOURCE=2', '-O2']
        return args

    def extra_ldflags(self):
        flags = [
            '-L' + os.path.join(self._cann_path, 'lib64'), '-lascendcl',
            '-L' + os.path.join(self._torch_npu_path, 'lib'), '-ltorch_npu'
        ]
        return flags

    def load(self, verbose=True):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        op_module = load(name=self.name,
                         sources=self.get_absolute_paths(self.sources()),
                         extra_include_paths=self.get_absolute_paths(self.include_paths()),
                         extra_cflags=self.cxx_args(),
                         extra_ldflags=self.extra_ldflags(),
                         verbose=verbose)
        __class__._loaded_ops[self.name] = op_module

        return op_module
