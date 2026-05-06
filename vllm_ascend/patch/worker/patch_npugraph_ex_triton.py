#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import importlib
import sys

import torch
import torchair
from torch._subclasses.fake_tensor import FakeTensor
from torchair.core._concrete_graph import _is_symlist
from torchair.npu_fx_compiler import _unpack_meta_list


class ValuePack:
    def __init__(self, meta, npu_meta=None) -> None:
        self._meta = meta
        self._npu_meta = meta if npu_meta is None else npu_meta

    @property
    def meta(self):
        return self._meta

    @property
    def npu(self):
        return self._npu_meta

    def __getitem__(self, key):
        if isinstance(self._meta, dict):
            return self._meta.get(key)
        raise ValueError(f"Unsupported meta type for ValuePack __getitem__, key:{key}, type: {type(self._meta)}")

    def __repr__(self) -> str:
        if isinstance(self._meta, FakeTensor):
            meta_str = f"FakeTensor(dtype={self._meta.dtype}, size={list(self._meta.size())}"
        elif isinstance(self._meta, torch.Tensor):
            meta_str = f"torch.Tensor(dtype={self._meta.dtype}, size={list(self._meta.size())}"
        elif isinstance(self._meta, torch.SymInt):
            meta_str = f"torch.SymInt({self._meta})"
        else:
            try:
                meta_str = f"{type(self._meta)}({self._meta})"
            except Exception:
                meta_str = f"{type(self._meta)}"
        return f"Pack(meta:{meta_str} npu:{self._npu_meta})"


def _unpack_meta(args, kwargs):
    unpacked_args = []
    unpacked_kwargs = {}

    def _get_meta_part(arg):
        if isinstance(arg, (list, tuple)) and any(isinstance(v, ValuePack) for v in arg):
            return _unpack_meta_list(arg)
        elif isinstance(arg, dict):
            return {k: v.meta if isinstance(v, ValuePack) else v for k, v in arg.items()}
        elif isinstance(arg, ValuePack):
            return arg.meta
        else:
            return arg

    for arg in args:
        unpacked_args.append(_get_meta_part(arg))

    for key, value in kwargs.items():
        unpacked_kwargs[key] = _get_meta_part(value)

    return list(unpacked_args), unpacked_kwargs


def _unpack_npu(self, args, kwargs):
    unpacked = []
    unpacked_kwargs = {}

    def _get_npu_part(arg):
        if isinstance(arg, (list, tuple)) and len(arg):
            if _is_symlist(arg):
                arg = self._graph.parse_symlist(arg)
            else:
                arg = [(v.npu if isinstance(v, ValuePack) else v) for v in arg]
            return arg
        elif isinstance(arg, dict):
            return {k: v.npu if isinstance(v, ValuePack) else v for k, v in arg.items()}
        elif isinstance(arg, ValuePack):
            return arg.npu
        else:
            return arg

    for arg in args:
        unpacked.append(_get_npu_part(arg))

    for key, value in kwargs.items():
        unpacked_kwargs[key] = _get_npu_part(value)

    return unpacked, unpacked_kwargs


torchair.core._concrete_graph.ValuePack = ValuePack
# The ValuePack class is referenced in these two modules, and after the patch, these two modules need to be reloaded.
importlib.reload(sys.modules["torchair.fx_summary"])
importlib.reload(sys.modules["torchair.npu_fx_compiler"])
torchair.npu_fx_compiler._unpack_meta = _unpack_meta
torchair.npu_fx_compiler._NpuGraphConverter._unpack_npu = _unpack_npu
