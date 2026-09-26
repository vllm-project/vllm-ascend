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
"""Regression tests for the windowed encoder-only attention path.

``npu_fusion_attention`` reads an absent ``atten_mask`` at the default
``sparse_mode=0`` as "no masking", so the encoder-only branch used to run with
unbounded attention. Every ``sliding_attention`` layer of a bidirectional
encoder therefore ignored its window: on a Laya (ModernBERT) encoder that is a
max |dlogit| of 12.7 against the reference.
"""

from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.attention_v1 as attn_module
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    FIA_FULL_MASK_SPARSE_MODE,
    AscendAttentionBackendImpl,
)

SLIDING_WINDOW = 4
NUM_HEADS = 2
HEAD_SIZE = 8


def _make_impl(sliding_window=SLIDING_WINDOW) -> AscendAttentionBackendImpl:
    vllm_config = MagicMock()
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.kv_transfer_config = None
    vllm_config.quant_config = None
    vllm_config.cache_config.cache_dtype = "float16"
    with (
        patch("vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=vllm_config),
        patch.object(attn_module, "needs_layer_aware_fia_graph_replay", return_value=False),
    ):
        return AscendAttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=1.0,
            num_kv_heads=NUM_HEADS,
            alibi_slopes=None,
            sliding_window=sliding_window,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type="encoder_only",
            kv_sharing_target_layer_name=None,
        )


def _make_metadata(seq_lens: list[int]) -> MagicMock:
    metadata = MagicMock()
    metadata.actual_seq_lengths_q = list(seq_lens)
    metadata.num_actual_tokens = sum(seq_lens)
    return metadata


def _make_qkv(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (num_tokens, NUM_HEADS, HEAD_SIZE)
    return (
        torch.zeros(shape, dtype=torch.float16),
        torch.zeros(shape, dtype=torch.float16),
        torch.zeros(shape, dtype=torch.float16),
    )


class TestEncoderBandMask(TestBase):
    def setUp(self):
        self.impl = _make_impl()
        self.device = torch.device("cpu")

    def test_blocks_everything_outside_the_inclusive_window(self):
        mask = self.impl._encoder_band_mask(6, SLIDING_WINDOW, self.device)

        self.assertEqual(mask.shape, (6, 6))
        self.assertEqual(mask.dtype, torch.bool)
        for query in range(6):
            for key in range(6):
                expected = abs(query - key) >= SLIDING_WINDOW
                self.assertEqual(bool(mask[query, key]), expected, f"q={query} k={key}")

    def test_mask_is_reused_for_the_same_token_count(self):
        first = self.impl._encoder_band_mask(6, SLIDING_WINDOW, self.device)
        second = self.impl._encoder_band_mask(6, SLIDING_WINDOW, self.device)

        self.assertIs(first, second)

    def test_only_the_current_step_mask_is_kept(self):
        first = self.impl._encoder_band_mask(6, SLIDING_WINDOW, self.device)
        other = self.impl._encoder_band_mask(7, SLIDING_WINDOW, self.device)

        self.assertIsNot(first, other)
        # Going back to an earlier token count rebuilds instead of restoring a
        # dropped entry: the memo only ever holds the mask of the last call.
        rebuilt = self.impl._encoder_band_mask(6, SLIDING_WINDOW, self.device)

        self.assertIsNot(rebuilt, first)
        self.assertIsNot(rebuilt, other)
        self.assertTrue(torch.equal(rebuilt, first))


class TestEncoderAttentionWindow(TestBase):
    def setUp(self):
        self.impl = _make_impl()
        self.query, self.key, self.value = _make_qkv(8)
        self.output = torch.zeros(8, NUM_HEADS, HEAD_SIZE, dtype=torch.float16)
        self.attention = patch.object(
            attn_module.torch_npu,
            "npu_fusion_attention",
            return_value=(torch.zeros(8, NUM_HEADS, HEAD_SIZE, dtype=torch.float16),),
        )
        self.mocked_attention = self.attention.start()
        self.addCleanup(self.attention.stop)

    def _call(self, metadata: MagicMock) -> None:
        self.impl._forward_encoder_attention(self.query, self.key, self.value, metadata, self.output)

    def test_long_sequence_gets_a_full_pairwise_band_mask(self):
        # 3 + 5 tokens: the 5-token sequence is longer than the 4-token window.
        self._call(_make_metadata([3, 8]))

        kwargs = self.mocked_attention.call_args.kwargs
        self.assertEqual(kwargs["actual_seq_qlen"], [3, 8])
        self.assertEqual(kwargs["query"].shape[0], 8)
        self.assertEqual(kwargs["atten_mask"].shape, (8, 8))
        self.assertEqual(kwargs["sparse_mode"], FIA_FULL_MASK_SPARSE_MODE)

    def test_pad_rows_are_trimmed_and_the_trailing_zero_is_dropped(self):
        # 10 rows are passed down but only 8 tokens are real, as in TND layout
        # with padding; a mask larger than the query is rejected by the op.
        query, key, value = _make_qkv(10)
        self.impl._forward_encoder_attention(query, key, value, _make_metadata([3, 8]), self.output)

        kwargs = self.mocked_attention.call_args.kwargs
        self.assertEqual(kwargs["actual_seq_qlen"], [3, 8])
        self.assertEqual(kwargs["query"].shape[0], 8)
        self.assertEqual(kwargs["atten_mask"].shape, (8, 8))

    def test_short_sequence_keeps_the_maskless_call(self):
        # Every sequence fits inside the inclusive window, so nothing is masked.
        self._call(_make_metadata([2, 5]))

        kwargs = self.mocked_attention.call_args.kwargs
        self.assertNotIn("atten_mask", kwargs)
        self.assertNotIn("sparse_mode", kwargs)
        self.assertEqual(kwargs["actual_seq_qlen"], [2, 5])

    def test_without_a_window_the_original_call_is_preserved(self):
        impl = _make_impl(sliding_window=None)
        query, key, value = _make_qkv(10)

        impl._forward_encoder_attention(query, key, value, _make_metadata([2, 8]), self.output)

        kwargs = self.mocked_attention.call_args.kwargs
        self.assertEqual(kwargs["actual_seq_qlen"], [2, 8, 0])
        self.assertEqual(kwargs["query"].shape[0], 10)


class TestEncoderAttentionOutputCopy(TestBase):
    def setUp(self):
        self.impl = _make_impl()

    def test_matching_shapes_are_copied_in_place(self):
        output = torch.zeros(4, 2, 8, dtype=torch.float16)
        attn_output = torch.ones(4, 2, 8, dtype=torch.float16)

        returned = self.impl._write_encoder_attention_output(output, attn_output, 4)

        self.assertIs(returned, output)
        self.assertTrue(torch.equal(output, attn_output))

    def test_padded_shapes_copy_the_valid_prefix_only(self):
        output = torch.zeros(6, 2, 8, dtype=torch.float16)
        attn_output = torch.ones(4, 2, 8, dtype=torch.float16)

        returned = self.impl._write_encoder_attention_output(output, attn_output, 4)

        self.assertIs(returned, output)
        self.assertTrue(torch.equal(output[:4], attn_output))
        self.assertTrue(torch.equal(output[4:], torch.zeros(2, 2, 8, dtype=torch.float16)))

    def test_a_padded_token_count_copies_the_operator_overlap(self):
        # The model runner pads the step up to the next cudagraph capture size,
        # so num_tokens may exceed the fused operator's row count.
        output = torch.zeros(6, 2, 8, dtype=torch.float16)
        attn_output = torch.ones(4, 2, 8, dtype=torch.float16)

        returned = self.impl._write_encoder_attention_output(output, attn_output, 6)

        self.assertIs(returned, output)
        self.assertTrue(torch.equal(output[:4], attn_output))
        self.assertTrue(torch.equal(output[4:], torch.zeros(2, 2, 8, dtype=torch.float16)))

    def test_the_operator_buffer_is_returned_as_is(self):
        output = torch.zeros(4, 2, 8, dtype=torch.float16)

        self.assertIs(self.impl._write_encoder_attention_output(output, output, 4), output)
