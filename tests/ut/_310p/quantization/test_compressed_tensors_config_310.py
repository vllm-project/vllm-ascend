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
#

from unittest.mock import MagicMock, patch

from vllm.model_executor.layers.linear import RowParallelLinear

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import COMPRESSED_TENSORS_W8A8_CONFIG
from vllm_ascend._310p.quantization.compressed_tensors_config import (
    AscendCompressedTensorsConfig310,
)
from vllm_ascend._310p.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod310,
)
from vllm_ascend.quantization.method_adapters import AscendLinearMethod


class TestAscendCompressedTensorsConfig310(TestBase):
    def setUp(self):
        self.config = AscendCompressedTensorsConfig310.from_config(COMPRESSED_TENSORS_W8A8_CONFIG)

    @patch("vllm_ascend.quantization.method_adapters.AscendLinearMethod.__init__")
    def test_get_linear_quant_method_uses_310p_scheme(self, mock_method):
        mock_method.return_value = None
        layer = MagicMock(spec=RowParallelLinear)

        result = self.config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")

        self.assertTrue(isinstance(result, AscendLinearMethod))
        self.assertTrue(isinstance(layer.scheme, AscendW8A8DynamicLinearMethod310))
