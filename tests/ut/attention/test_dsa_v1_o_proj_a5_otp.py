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
# This file is part of the vllm-ascend project.
#
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
from vllm_ascend.utils import AscendDeviceType


def _make_impl():
    """Bare AscendDSAImpl with just the attributes _wo_a_bmm / _forward_o_proj touch."""
    impl = AscendDSAImpl.__new__(AscendDSAImpl)
    wo_a = nn.Module()
    wo_a.weight = nn.Parameter(torch.randn(8, 16).to(torch.float8_e4m3fn), requires_grad=False)
    wo_a.weight_scale = nn.Parameter(torch.randint(0, 255, (4, 8, 2), dtype=torch.uint8), requires_grad=False)
    impl.wo_a = wo_a
    impl.n_local_groups = 4
    return impl


class TestDSAV1WoABmmDispatch(TestBase):
    """_wo_a_bmm must dispatch to the FP8-quantized matmul on A5 and the BF16
    batch matmul otherwise. This is what lets A5 + oproj_tp compose."""

    def _patch_device(self, is_a5: bool):
        target = "vllm_ascend.attention.dsa_v1.get_ascend_device_type"
        mock = MagicMock(return_value=AscendDeviceType.A5 if is_a5 else AscendDeviceType.A3)
        return target, mock

    def test_a5_uses_quantized_matmul(self):
        impl = _make_impl()
        with (
            patch(*self._patch_device(True)),
            patch("vllm_ascend.attention.dsa_v1.torch_npu") as mock_npu,
        ):
            mock_npu.npu_dynamic_mx_quant.return_value = (
                torch.randn(2, 4, 8),
                torch.randint(0, 255, (2, 4, 8), dtype=torch.uint8),
            )
            mock_npu.npu_transpose_quant_batchmatmul.return_value = torch.randn(2, 4, 8)
            mock_npu.npu_transpose_batchmatmul.return_value = torch.randn(2, 4, 8)

            x = torch.randn(2, 4, 8)
            out = impl._wo_a_bmm(x, num_tokens=2)

            self.assertEqual(out.shape, (2, 32))
            mock_npu.npu_dynamic_mx_quant.assert_called_once()
            mock_npu.npu_transpose_quant_batchmatmul.assert_called_once()
            mock_npu.npu_transpose_batchmatmul.assert_not_called()

    def test_non_a5_uses_bf16_matmul(self):
        impl = _make_impl()
        with (
            patch(*self._patch_device(False)),
            patch("vllm_ascend.attention.dsa_v1.torch_npu") as mock_npu,
        ):
            mock_npu.npu_transpose_batchmatmul.return_value = torch.randn(2, 4, 8)
            mock_npu.npu_dynamic_mx_quant.return_value = (torch.randn(2, 4, 8), torch.zeros(1, dtype=torch.uint8))
            mock_npu.npu_transpose_quant_batchmatmul.return_value = torch.randn(2, 4, 8)

            x = torch.randn(2, 4, 8)
            out = impl._wo_a_bmm(x, num_tokens=2)

            self.assertEqual(out.shape, (2, 32))
            mock_npu.npu_transpose_batchmatmul.assert_called_once()
            mock_npu.npu_dynamic_mx_quant.assert_not_called()
            mock_npu.npu_transpose_quant_batchmatmul.assert_not_called()
