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
from vllm_ascend._310p.attention.attention_v1 import (
    AscendAttentionBackend310,
    AscendAttentionBackendImpl310,
    AscendAttentionMetadataBuilder310,
    AscendAttentionState,
)


class TestAscendAttentionBackend310(TestBase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.utils_patcher = patch("vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_config)
        self.utils_patcher.start()

    def test_get_impl_cls(self):
        self.assertEqual(AscendAttentionBackend310.get_impl_cls(), AscendAttentionBackendImpl310)

    def test_get_builder_cls(self):
        self.assertEqual(AscendAttentionBackend310.get_builder_cls(), AscendAttentionMetadataBuilder310)

    def test_get_kv_cache_shape_not(self):
        result = AscendAttentionBackend310.get_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 75, 20, 16))


class TestAscendAttentionBackendImpl310(TestBase):
    def setUp(self):
        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.attention_type.ENCODER = "encoder"
        self.attn_metadata = MagicMock()
        self.attn_metadata.return_value = "1"
        self.mock_vllm_config = MagicMock()
        self.layer_no_quant = MagicMock(spec=["layer_name", "_k_scale_float", "_v_scale_float"])
        self.layer_no_quant.layer_name = "test_layer"
        self.layer_no_quant._k_scale_float = 1.0
        self.layer_no_quant._v_scale_float = 1.0
        self.config_patcher = patch(
            "vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=self.mock_vllm_config
        )
        self.config_patcher.start()
        self.impl = AscendAttentionBackendImpl310(
            num_heads=8,
            head_size=128,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None,
        )

    @patch("torch_npu._npu_reshape_and_cache")
    @patch("torch_npu._npu_flash_attention")
    @patch("vllm_ascend.attention.attention_v1.get_forward_context")
    def test_forward_prefill_310(
        self, mock_get_forward_context, mock_npu_npu_flash_attention, mock_npu_reshape_and_cache
    ):
        """Test forward pass in PrefillCacheHit state"""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
        output = torch.empty_like(query)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillNoCache
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.actual_seq_lengths_q = [10]
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.num_decode_tokens = 0
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)

        mock_get_forward_context.return_value = MagicMock(capturing=False)
        mock_npu_npu_flash_attention.return_value = torch.ones(10, 8, 64)
        output = self.impl.forward_prefill_310(query, key, value, metadata, output)

        mock_npu_npu_flash_attention.assert_called_once()

    @patch("torch_npu.npu_format_cast", return_value=torch.randn((1, 128, 16, 16), dtype=torch.float16))
    @patch("torch_npu._npu_reshape_and_cache")
    @patch("torch_npu._npu_paged_attention_splitfuse")
    @patch("vllm_ascend.attention.attention_v1.get_forward_context")
    def test_forward_chunked_prefill_310(
        self, mock_get_forward_context, mock_npu_paged_attention_splitfuse, mock_npu_reshape_and_cache, mock_format_cast
    ):
        """Test forward pass in PrefillCacheHit state"""
        query = torch.randn(5, 8, 64)
        output = torch.empty_like(query)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.ChunkedPrefill
        metadata.attn_mask = torch.randn(1, 128, 16, 16)
        metadata.query_lens = torch.tensor([5])
        metadata.seq_lens = torch.tensor([1, 4])
        metadata.query_start_loc = torch.tensor([0, 1, 5])
        metadata.actual_seq_lengths_q = [5]
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.num_decode_tokens = 0
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)

        mock_get_forward_context.return_value = MagicMock(capturing=False)
        mock_npu_paged_attention_splitfuse.return_value = torch.ones(5, 8, 64)
        output = self.impl.forward_chunked_prefill_310(query, metadata, output)

        mock_npu_paged_attention_splitfuse.assert_called_once()

    @patch("vllm_ascend.attention.attention_v1.using_paged_attention")
    @patch("torch_npu._npu_paged_attention")
    @patch("torch_npu._npu_reshape_and_cache")
    @patch("vllm_ascend.attention.attention_v1.get_forward_context")
    def test_forward_paged_attention_310(
        self, mock_get_forward_context, mock_npu_reshape_and_cache, mock_paged_attention, mock_using_paged_attention
    ):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(4, 8 * 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([4])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 4
        metadata.slot_mapping = torch.zeros(4, dtype=torch.long)
        metadata.num_decodes = 4
        metadata.num_prefills = 0
        mock_using_paged_attention.return_value = True

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        output = self.impl.forward_paged_attention(query, metadata, output)

        mock_paged_attention.assert_called_once()
