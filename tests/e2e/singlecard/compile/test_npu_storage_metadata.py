#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def _get_npu_storage_shape_op():
    assert enable_custom_op(), "requires vllm_ascend custom ops"
    return torch.ops._C_ascend.get_npu_storage_shape


def test_get_npu_storage_shape_reports_backing_storage_layout():
    get_npu_storage_shape = _get_npu_storage_shape_op()

    base = torch.arange(2 * 3 * 4 * 5, device="npu", dtype=torch.float16).reshape(2, 3, 4, 5)
    sliced = base[:, :, 1:3, :]
    transposed = base.transpose(1, 2)
    contiguous_sliced = sliced.contiguous()

    assert list(get_npu_storage_shape(base)) == [base.storage().size()]
    assert list(get_npu_storage_shape(sliced)) == [base.storage().size()]
    assert list(get_npu_storage_shape(transposed)) == [base.storage().size()]
    assert list(get_npu_storage_shape(contiguous_sliced)) == list(contiguous_sliced.shape)


def test_get_npu_storage_shape_tracks_npu_format_cast_layout():
    get_npu_storage_shape = _get_npu_storage_shape_op()

    base = torch.empty((1, 3, 7, 7), device="npu", dtype=torch.float16)
    nhwc = torch_npu.npu_format_cast(base, 1)

    assert str(torch_npu.get_npu_format(nhwc)) == "NHWC"
    assert list(nhwc.shape) == list(base.shape)
    assert list(get_npu_storage_shape(nhwc)) == list(base.shape)


def test_get_npu_storage_shape_rejects_cpu_tensor():
    get_npu_storage_shape = _get_npu_storage_shape_op()

    with pytest.raises(RuntimeError, match="get_npu_storage_shape only supports NPU tensors"):
        get_npu_storage_shape(torch.empty(2, 3))


def test_get_npu_storage_format_op_is_not_registered():
    assert enable_custom_op(), "requires vllm_ascend custom ops"

    with pytest.raises(AttributeError):
        torch.ops._C_ascend.get_npu_storage_format
