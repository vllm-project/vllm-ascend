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
from vllm.config import CUDAGraphMode

from tests.ut.base import TestBase
from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310


class TestAttentionMaskBuilder310(TestBase):
    def setUp(self):
        self.max_seqlen = 4096
        self.attention_mask_builder = AttentionMaskBuilder310(torch.device("cpu"), self.max_seqlen)

    @patch("torch_npu.npu_format_cast")
    def test_get_attention_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        self.attention_mask_builder.support_compressed_mask = False
        model_config = MagicMock()
        attn_mask = self.attention_mask_builder.get_attention_mask(causal=True, model_config=model_config)
        self.assertEqual(attn_mask.shape, (1, self.max_seqlen // 16, self.max_seqlen, 16))
        self.assertEqual(attn_mask[0][-1][0][-1], torch.tensor(float("-inf"), dtype=torch.float16))

    @patch("torch_npu.npu_format_cast")
    def test_get_splitfuse_attn_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        attn_metadata = MagicMock()
        attn_metadata.query_start_loc = torch.tensor([0, 1, 5])
        attn_metadata.seq_lens = torch.tensor([7, 4])
        attn_mask = self.attention_mask_builder.get_splitfuse_mask(attn_metadata, torch.device("cpu"))
        self.assertEqual(attn_mask.shape, (1, self.max_seqlen // 16, 16, 16))

    @patch("torch_npu.npu_format_cast")
    def test_get_non_causal_splitfuse_mask_310(self, mock_format_cast):
        mock_format_cast.side_effect = lambda x, y: x
        attn_metadata = MagicMock()
        attn_metadata.query_start_loc = torch.tensor([0, 2, 5])
        attn_metadata.seq_lens = torch.tensor([10, 8])
        attn_mask = AttentionMaskBuilder310.get_non_causal_splitfuse_mask(attn_metadata, torch.device("cpu"))
        self.assertEqual(attn_mask.shape, (1, self.max_seqlen // 16, 16, 16))

    def test_graph_safe_query_positions_support_mixed_request_lengths(self):
        attn_metadata = MagicMock()
        attn_metadata.num_actual_tokens = 5
        attn_metadata.query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
        attn_metadata.seq_lens = torch.tensor([7, 10], dtype=torch.int32)

        causal = AttentionMaskBuilder310._get_graph_safe_query_positions(
            attn_metadata,
            torch.device("cpu"),
            causal=True,
            zero_descriptor_padding=True,
        )
        non_causal = AttentionMaskBuilder310._get_graph_safe_query_positions(
            attn_metadata,
            torch.device("cpu"),
            causal=False,
            zero_descriptor_padding=True,
        )

        torch.testing.assert_close(
            causal,
            torch.tensor([5, 6, 7, 8, 9], dtype=torch.int32),
        )
        torch.testing.assert_close(
            non_causal,
            torch.tensor([6, 6, 9, 9, 9], dtype=torch.int32),
        )

    def test_graph_safe_query_positions_preserve_piecewise_padding(self):
        attn_metadata = MagicMock()
        attn_metadata.num_actual_tokens = 32
        attn_metadata.query_start_loc = torch.tensor([0, 15, 30], dtype=torch.int32)
        attn_metadata.seq_lens = torch.tensor([20, 25], dtype=torch.int32)

        causal = AttentionMaskBuilder310._get_graph_safe_query_positions(
            attn_metadata,
            torch.device("cpu"),
            causal=True,
            zero_descriptor_padding=False,
        )
        non_causal = AttentionMaskBuilder310._get_graph_safe_query_positions(
            attn_metadata,
            torch.device("cpu"),
            causal=False,
            zero_descriptor_padding=False,
        )

        assert causal[-2:].tolist() == [25, 26]
        assert non_causal[-2:].tolist() == [24, 24]

    def test_graph_safe_query_positions_zero_descriptor_padding(self):
        attn_metadata = MagicMock()
        attn_metadata.num_actual_tokens = 32
        attn_metadata.query_start_loc = torch.tensor([0, 15, 30], dtype=torch.int32)
        attn_metadata.seq_lens = torch.tensor([20, 25], dtype=torch.int32)

        causal = AttentionMaskBuilder310._get_graph_safe_query_positions(
            attn_metadata,
            torch.device("cpu"),
            causal=True,
            zero_descriptor_padding=True,
        )
        non_causal = AttentionMaskBuilder310._get_graph_safe_query_positions(
            attn_metadata,
            torch.device("cpu"),
            causal=False,
            zero_descriptor_padding=True,
        )

        torch.testing.assert_close(
            causal,
            torch.cat(
                (
                    torch.arange(5, 20, dtype=torch.int32),
                    torch.arange(10, 25, dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                )
            ),
        )
        torch.testing.assert_close(
            non_causal,
            torch.cat(
                (
                    torch.full((15,), 19, dtype=torch.int32),
                    torch.full((15,), 24, dtype=torch.int32),
                    torch.zeros(2, dtype=torch.int32),
                )
            ),
        )

    def test_exact_full_decode_only_routes_query_positions_to_device_math(self):
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(method="dflash"),
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            ),
        )
        forward_context = SimpleNamespace(
            vllm_config=config,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        )
        attn_metadata = MagicMock()
        expected = torch.tensor([0], dtype=torch.int32)

        with (
            patch(
                "vllm_ascend._310p.attention.attention_mask.get_forward_context",
                return_value=forward_context,
            ),
            patch(
                "vllm_ascend._310p.dflash_full_decode_only.is_310p",
                return_value=True,
            ),
            patch.object(
                AttentionMaskBuilder310,
                "_get_graph_safe_query_positions",
                return_value=expected,
            ) as graph_safe_positions,
        ):
            actual = AttentionMaskBuilder310._get_query_positions(
                attn_metadata,
                torch.device("cpu"),
                causal=True,
            )

        self.assertIs(actual, expected)
        graph_safe_positions.assert_called_once_with(
            attn_metadata,
            torch.device("cpu"),
            causal=True,
            zero_descriptor_padding=True,
        )

    @patch("torch_npu.npu_format_cast", side_effect=lambda tensor, _: tensor)
    @patch.object(
        AttentionMaskBuilder310,
        "_requires_graph_safe_query_positions",
        return_value=True,
    )
    def test_exact_piecewise_routes_splitfuse_positions_to_device_math(
        self,
        _,
        __,
    ):
        attn_metadata = MagicMock()
        attn_metadata.num_actual_tokens = 5
        attn_metadata.query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
        attn_metadata.seq_lens = torch.tensor([7, 10], dtype=torch.int32)

        with patch.object(
            AttentionMaskBuilder310,
            "_get_graph_safe_query_positions",
            wraps=AttentionMaskBuilder310._get_graph_safe_query_positions,
        ) as graph_safe_positions:
            AttentionMaskBuilder310.get_splitfuse_mask(
                attn_metadata,
                torch.device("cpu"),
            )

        graph_safe_positions.assert_called_once_with(
            attn_metadata,
            torch.device("cpu"),
            causal=True,
            zero_descriptor_padding=False,
        )

    def test_hybrid_graph_safe_positions_follow_effective_runtime_mode(self):
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(method="dflash"),
            compilation_config=SimpleNamespace(
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            ),
            additional_config={
                "ascend_compilation_config": {
                    "dflash_full_and_piecewise_capture_config": {
                        "piecewise_capture_size": 64,
                        "full_capture_size": 160,
                    },
                },
            },
        )

        with patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ):
            with patch(
                "vllm_ascend._310p.attention.attention_mask.get_forward_context",
                return_value=SimpleNamespace(
                    vllm_config=config,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                ),
            ):
                self.assertTrue(AttentionMaskBuilder310._requires_graph_safe_query_positions())

            with patch(
                "vllm_ascend._310p.attention.attention_mask.get_forward_context",
                return_value=SimpleNamespace(
                    vllm_config=config,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                ),
            ):
                self.assertTrue(AttentionMaskBuilder310._requires_graph_safe_query_positions())

    def test_get_compressed_non_causal_splitfuse_mask_310(self):
        from vllm_ascend._310p.attention.attention_mask import COMPRESSED_MASK_SEQ_LEN

        mask = AttentionMaskBuilder310.get_compressed_non_causal_splitfuse_mask(torch.device("cpu"))
        self.assertEqual(mask.shape, (COMPRESSED_MASK_SEQ_LEN, COMPRESSED_MASK_SEQ_LEN))
        self.assertTrue(torch.all(mask == 0))
