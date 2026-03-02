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

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.ops.vocab_parallel_embedding import (
    AscendParallelLMHead310,
    _AscendParallelLMHead310QuantMethod,
)


class TestAscendParallelLMHead310(TestBase):
    def test_init_wraps_quant_method(self):
        base_method = MagicMock()

        def fake_lm_head_init(
            module,
            num_embeddings: int,
            embedding_dim: int,
            bias: bool = False,
            params_dtype: torch.dtype | None = None,
            org_num_embeddings: int | None = None,
            padding_size: int = 64,
            quant_config=None,
            prefix: str = "",
        ) -> None:
            del num_embeddings, embedding_dim, bias, params_dtype
            del org_num_embeddings, padding_size, quant_config, prefix
            module.quant_method = base_method

        with patch(
            "vllm_ascend._310p.ops.vocab_parallel_embedding.AscendParallelLMHead.__init__",
            new=fake_lm_head_init,
        ):
            lm_head = AscendParallelLMHead310(
                num_embeddings=32,
                embedding_dim=16,
                bias=False,
                params_dtype=torch.float16,
                prefix="lm_head",
            )

        self.assertIsInstance(lm_head.quant_method, _AscendParallelLMHead310QuantMethod)
        self.assertIs(lm_head.quant_method._base_method, base_method)

    def test_apply_delegates_to_base_method(self):
        base_method = MagicMock()
        expected_output = MagicMock()
        base_method.apply.return_value = expected_output
        wrapped_method = _AscendParallelLMHead310QuantMethod(base_method)

        layer = MagicMock()
        hidden_states = torch.randn(2, 4)
        bias = torch.randn(4)

        output = wrapped_method.apply(layer, hidden_states, bias=bias)

        base_method.apply.assert_called_once_with(layer, hidden_states, bias=bias)
        self.assertIs(output, expected_output)

    @patch("vllm_ascend._310p.ops.vocab_parallel_embedding.maybe_trans_nz")
    def test_process_weights_after_loading_casts_weight_to_nz(self, mock_maybe_trans_nz):
        base_method = MagicMock()
        wrapped_method = _AscendParallelLMHead310QuantMethod(base_method)

        original_weight = torch.randn(32, 16, dtype=torch.float16)
        converted_weight = torch.randn(32, 16, dtype=torch.float16)
        mock_maybe_trans_nz.return_value = converted_weight

        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight.data = original_weight

        wrapped_method.process_weights_after_loading(layer)

        base_method.process_weights_after_loading.assert_called_once_with(layer)
        mock_maybe_trans_nz.assert_called_once_with(original_weight)
        self.assertIs(layer.weight.data, converted_weight)
