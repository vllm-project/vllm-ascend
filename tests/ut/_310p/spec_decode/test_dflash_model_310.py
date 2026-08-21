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
from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
from vllm_ascend._310p.spec_decode.dflash_model_310 import precompute_and_store_context_kv_310


class TestPrecomputeContextKv310(TestBase):
    def setUp(self):
        AscendRotaryEmbedding310.set_rope_position_flag_310p(False)

    def tearDown(self):
        AscendRotaryEmbedding310.set_rope_position_flag_310p(False)

    def test_precompute_per_layer_rope_and_kv_update(self):
        model = MagicMock()
        model._num_attn_layers = 2
        model._kv_size = 256
        model._head_dim = 64
        model._num_kv_heads = 4
        model.hidden_norm = MagicMock(side_effect=lambda x: x)

        num_ctx = 3
        nkv = 4
        hd = 64
        kv = nkv * hd
        L = 2

        context_states = torch.randn(num_ctx, 128)
        context_positions = torch.tensor([10, 11, 12], dtype=torch.int32)
        slot_mapping = torch.tensor([1, 2, 3], dtype=torch.int32)

        # fused_kv output: [num_ctx, L, 2, nkv, hd]
        fused_out = torch.randn(num_ctx, L * 2 * kv)
        model._fused_kv_weight = torch.randn(L * 2 * kv, 128)
        model._fused_kv_bias = None

        layers = []
        attn_layers = []
        for _ in range(L):
            layer = MagicMock()
            layer.self_attn.k_norm = MagicMock(side_effect=lambda x: x)
            # 310P RoPE returns new (query, key) tensors; precompute must use them.
            layer.self_attn.rotary_emb = MagicMock(side_effect=lambda pos, q, k: (q, k))
            layers.append(layer)

            attn = MagicMock()
            attn.kv_cache = (torch.randn(1), torch.randn(1))
            attn.impl.do_kv_cache_update = MagicMock()
            attn_layers.append(attn)

        model.layers = layers
        model._attn_layers = attn_layers

        with patch("torch.nn.functional.linear", return_value=fused_out):
            precompute_and_store_context_kv_310(model, context_states, context_positions, slot_mapping)

        self.assertFalse(AscendRotaryEmbedding310._is_drafting_update_enabled)
        self.assertEqual(layers[0].self_attn.rotary_emb.call_count, 1)
        self.assertEqual(layers[1].self_attn.rotary_emb.call_count, 1)
        for attn in attn_layers:
            attn.impl.do_kv_cache_update.assert_called_once()
            # K passed to the cache update must be the RoPE'd output.
            k_arg = attn.impl.do_kv_cache_update.call_args.args[1]
            self.assertEqual(tuple(k_arg.shape), (num_ctx, nkv, hd))

    def test_precompute_restores_prior_drafting_flag(self):
        """Flag must be restored to its prior value (True inside _run_merged_draft)."""
        model = MagicMock()
        model._num_attn_layers = 1
        model._kv_size = 256
        model._head_dim = 64
        model._num_kv_heads = 4
        model.hidden_norm = MagicMock(side_effect=lambda x: x)

        num_ctx = 2
        kv = 256
        fused_out = torch.randn(num_ctx, 2 * kv)
        model._fused_kv_weight = torch.randn(2 * kv, 128)
        model._fused_kv_bias = None

        captured = {}

        def _rope(pos, q, k):
            captured["flag_during"] = AscendRotaryEmbedding310._is_drafting_update_enabled
            return q, k

        layer = MagicMock()
        layer.self_attn.k_norm = MagicMock(side_effect=lambda x: x)
        layer.self_attn.rotary_emb = MagicMock(side_effect=_rope)
        attn = MagicMock()
        attn.kv_cache = (torch.randn(1), torch.randn(1))
        attn.impl.do_kv_cache_update = MagicMock()
        model.layers = [layer]
        model._attn_layers = [attn]

        # Simulate being called inside _run_merged_draft (flag already True).
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        with patch("torch.nn.functional.linear", return_value=fused_out):
            precompute_and_store_context_kv_310(
                model,
                torch.randn(num_ctx, 128),
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32),
            )
        # Flag stays True during RoPE and is restored to True (not forced False).
        self.assertTrue(captured["flag_during"])
        self.assertTrue(AscendRotaryEmbedding310._is_drafting_update_enabled)

    def test_precompute_skips_kv_when_no_slot_mapping(self):
        model = MagicMock()
        model._num_attn_layers = 1
        model._kv_size = 256
        model._head_dim = 64
        model._num_kv_heads = 4
        model.hidden_norm = MagicMock(side_effect=lambda x: x)

        num_ctx = 2
        kv = 256
        fused_out = torch.randn(num_ctx, 2 * kv)
        model._fused_kv_weight = torch.randn(2 * kv, 128)
        model._fused_kv_bias = None

        layer = MagicMock()
        layer.self_attn.k_norm = MagicMock(side_effect=lambda x: x)
        layer.self_attn.rotary_emb = MagicMock(side_effect=lambda pos, q, k: (q, k))
        model.layers = [layer]
        model._attn_layers = []

        with patch("torch.nn.functional.linear", return_value=fused_out):
            precompute_and_store_context_kv_310(
                model, torch.randn(num_ctx, 128), torch.tensor([0, 1], dtype=torch.int32), None
            )

        layer.self_attn.rotary_emb.assert_called_once()

    def test_precompute_reports_context_probe_intermediates(self):
        model = MagicMock()
        model._num_attn_layers = 1
        model._kv_size = 2
        model._head_dim = 1
        model._num_kv_heads = 2
        model.hidden_norm = MagicMock(side_effect=lambda value: value + 10)

        context_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
        context_positions = torch.tensor([7, 8], dtype=torch.int32)
        slot_mapping = torch.tensor([11, 12], dtype=torch.int32)
        fused_out = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=torch.float16,
        )
        model._fused_kv_weight = torch.randn(4, 2)
        model._fused_kv_bias = None

        layer = MagicMock()
        layer.self_attn.k_norm = MagicMock(side_effect=lambda value: value + 20)
        layer.self_attn.rotary_emb = MagicMock(side_effect=lambda _pos, query, key: (query + 100, key + 200))
        attn = MagicMock()
        attn.kv_cache = (torch.randn(1), torch.randn(1))
        attn.impl.do_kv_cache_update = MagicMock()
        model.layers = [layer]
        model._attn_layers = [attn]
        context_probe = MagicMock()
        model._fdo_context_probe = context_probe

        with patch("torch.nn.functional.linear", return_value=fused_out):
            precompute_and_store_context_kv_310(
                model,
                context_states,
                context_positions,
                slot_mapping,
            )

        context_probe.capture_context_inputs.assert_called_once()
        input_call = context_probe.capture_context_inputs.call_args.kwargs
        torch.testing.assert_close(input_call["context_states"], context_states)
        torch.testing.assert_close(input_call["context_positions"], context_positions)
        torch.testing.assert_close(input_call["normed_context_states"], context_states + 10)
        torch.testing.assert_close(input_call["slot_mapping"], slot_mapping)

        context_probe.capture_context_k_norm.assert_called_once()
        norm_call = context_probe.capture_context_k_norm.call_args.kwargs
        assert norm_call["layer_index"] == 0
        expected_k = torch.tensor([[1.0, 2.0], [5.0, 6.0]], dtype=torch.float16)
        expected_v = torch.tensor([[3.0, 4.0], [7.0, 8.0]], dtype=torch.float16)
        torch.testing.assert_close(norm_call["k_norm_input"].reshape(2, 2), expected_k)
        torch.testing.assert_close(norm_call["k_norm_output"], expected_k + 20)
        context_probe.capture_context_rope.assert_called_once()
        rope_call = context_probe.capture_context_rope.call_args.kwargs
        assert rope_call["layer_index"] == 0
        torch.testing.assert_close(rope_call["k_rope"], expected_k + 120)
        torch.testing.assert_close(rope_call["value"].reshape(2, 2), expected_v)
