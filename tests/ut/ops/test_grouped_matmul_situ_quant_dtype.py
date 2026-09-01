#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""CPU coverage for MX semantic dtype restoration before custom-op dispatch."""

from dataclasses import dataclass, field

import torch

from vllm_ascend.ops.grouped_matmul_situ_quant import _restore_mxfp_semantic_dtype


@dataclass
class _FakeTensor:
    dtype: object
    view_calls: list[object] = field(default_factory=list)

    def view(self, dtype: object) -> "_FakeTensor":
        self.view_calls.append(dtype)
        return _FakeTensor(dtype=dtype)


def test_restore_mxfp_dtype_for_uint8_tensor():
    semantic_dtype = object()
    tensor = _FakeTensor(torch.uint8)

    restored = _restore_mxfp_semantic_dtype(tensor, semantic_dtype)  # type: ignore[arg-type]

    assert tensor.view_calls == [semantic_dtype]
    assert restored.dtype is semantic_dtype


def test_restore_mxfp_dtype_for_tensor_list_and_tuple():
    semantic_dtype = object()
    uint8_tensor = _FakeTensor(torch.uint8)
    typed_tensor = _FakeTensor(torch.float16)

    restored_list = _restore_mxfp_semantic_dtype(  # type: ignore[arg-type]
        [uint8_tensor, typed_tensor], semantic_dtype
    )
    restored_tuple = _restore_mxfp_semantic_dtype(  # type: ignore[arg-type]
        (uint8_tensor, typed_tensor), semantic_dtype
    )

    assert isinstance(restored_list, list)
    assert isinstance(restored_tuple, tuple)
    assert uint8_tensor.view_calls == [semantic_dtype, semantic_dtype]
    assert typed_tensor.view_calls == []
    assert restored_list[1] is typed_tensor
    assert restored_tuple[1] is typed_tensor
