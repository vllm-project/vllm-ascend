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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from tests.ut.base import TestBase
from vllm_ascend._310p.ops.vocab_parallel_embedding import AscendParallelLMHead310
from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


class TestAscendParallelLMHead310(TestBase):
    def _fake_embedding_init(
        self,
        module: nn.Module,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = 64,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        del num_embeddings, embedding_dim, params_dtype, org_num_embeddings
        del padding_size, quant_config, prefix
        nn.Module.__init__(module)
        module.quant_method = SimpleNamespace()

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_casts_weight_to_nz(self, mock_npu_format_cast):
        mock_npu_format_cast.side_effect = lambda weight, fmt: weight

        with patch.object(
            AscendVocabParallelEmbedding,
            "__init__",
            new=self._fake_embedding_init,
        ):
            lm_head = AscendParallelLMHead310(
                num_embeddings=32,
                embedding_dim=16,
                bias=False,
                params_dtype=torch.float16,
                prefix="lm_head",
            )

        layer = MagicMock()
        layer.weight = MagicMock()
        layer.weight.data = torch.randn(32, 16, dtype=torch.float16)
        original_weight = layer.weight.data

        lm_head.quant_method.process_weights_after_loading(layer)

        mock_npu_format_cast.assert_called_once()
        args, kwargs = mock_npu_format_cast.call_args
        self.assertIs(args[0], original_weight)
        self.assertEqual(args[1], ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(kwargs, {})
