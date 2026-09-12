# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.gdn import prepare_causal_conv1d_weight_for_loading


def test_prepare_causal_conv1d_weight_for_loading() -> None:
    weight = torch.nn.Parameter(torch.empty(6, 1, 4), requires_grad=False)

    def sharded_loader(param: torch.Tensor, loaded_weight: torch.Tensor, offset: int) -> None:
        param.data.copy_(loaded_weight[offset : offset + param.shape[0]])

    weight.weight_loader = sharded_loader
    conv1d = SimpleNamespace(weight=weight)
    loaded_weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 1, 4)

    prepare_causal_conv1d_weight_for_loading(conv1d)
    weight.weight_loader(weight, loaded_weight, 1)

    expected = loaded_weight[1:7].squeeze(1).transpose(0, 1)
    torch.testing.assert_close(weight, expected)
    assert weight.shape == (4, 6)
    assert weight.is_contiguous()


def test_prepare_causal_conv1d_weight_rejects_unexpected_shape() -> None:
    for shape in ((6, 4), (6, 2, 4)):
        weight = torch.nn.Parameter(torch.empty(shape), requires_grad=False)
        weight.weight_loader = lambda param, loaded_weight: None
        with pytest.raises(ValueError, match="Expected causal conv1d weight shape"):
            prepare_causal_conv1d_weight_for_loading(SimpleNamespace(weight=weight))


def test_prepare_causal_conv1d_weight_requires_loader() -> None:
    weight = torch.nn.Parameter(torch.empty(6, 1, 4), requires_grad=False)
    with pytest.raises(AttributeError, match="does not have a weight_loader"):
        prepare_causal_conv1d_weight_for_loading(SimpleNamespace(weight=weight))
