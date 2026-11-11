#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Routing and contract tests for the C8 multi-token-query decode segment.

Covers the MTP / speculative decoding path: `AscendC8AttentionBackendImpl`
must serve decode segments whose requests carry several query tokens through
FIA TND + quant_mode=5 pseudo-quant (with a gather+dequant fallback), while
single-token decode keeps the BNSD fast path.
"""

from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.attention_v1 as attn_module
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    C8_FIA_PSEUDO_QUANT_MODE,
    AscendC8AttentionBackendImpl,
    FIAParamProvider,
)

NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_SIZE = 64
NUM_BLOCKS = 16
BLOCK_SIZE = 128


def _fake_fia_recorder(calls):
    """Build a torch_npu FIA stand-in that records kwargs and returns
    view-compatible outputs for both the TND and BNSD call shapes."""

    def fake_fia(*args, **kwargs):
        calls.append(kwargs)
        query = kwargs.get("query")
        if query is None:
            query = args[0]
        tokens = query.shape[0]
        if kwargs.get("input_layout") == "BNSD":
            out = torch.zeros(tokens, NUM_HEADS, 1, HEAD_SIZE)
        else:
            out = torch.zeros(tokens, NUM_HEADS, HEAD_SIZE)
        return out, None

    return fake_fia


class TestC8Quant5MultiTokenDecode(TestBase):
    """Standalone fixtures: deliberately NOT derived from the shared
    TestAscendAttentionBackendImpl so its suite is not re-executed against
    the C8 impl built here."""

    def setUp(self):
        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.mock_vllm_config.cache_config.cache_dtype = "float16"
        self.config_patcher = patch(
            "vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=self.mock_vllm_config
        )
        self.utils_config_patcher = patch(
            "vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_vllm_config
        )
        self.config_patcher.start()
        self.utils_config_patcher.start()
        self.addCleanup(self.config_patcher.stop)
        self.addCleanup(self.utils_config_patcher.stop)

        self.impl = AscendC8AttentionBackendImpl(
            num_heads=NUM_HEADS,
            head_size=HEAD_SIZE,
            scale=1.0,
            num_kv_heads=NUM_KV_HEADS,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None,
        )
        self.impl.key_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.int8)
        self.impl.value_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.int8)
        # float kv_cache_dtype forces enable_c8_quant off in __init__; the
        # capture-path tests exercise the C8 branch of full_graph_fia, so
        # enable the INT8 KV recipe explicitly.
        self.impl.enable_c8_quant = True

        self.layer = MagicMock()
        self.layer._c8_k_aq_scale_nz_bnsd = torch.ones(1, NUM_KV_HEADS, HEAD_SIZE)
        self.layer._c8_v_aq_scale_nz_bnsd = torch.ones(1, NUM_KV_HEADS, HEAD_SIZE)
        self.layer._c8_k_scale = torch.ones(1, NUM_KV_HEADS, HEAD_SIZE)
        self.layer._c8_v_scale = torch.ones(1, NUM_KV_HEADS, HEAD_SIZE)
        self.layer._c8_k_offset = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE)
        self.layer._c8_v_offset = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE)

        # Two MTP decode requests with 3 query tokens each, plus one prefill
        # request with 3 tokens: cumulative q lengths [3, 6, 9].
        self.metadata = MagicMock()
        self.metadata.num_decodes = 2
        self.metadata.num_decode_tokens = 6
        self.metadata.num_prefills = 0
        self.metadata.num_actual_tokens = 9
        self.metadata.max_query_len = 3
        self.metadata.actual_seq_lengths_q = [3, 6, 9]
        self.metadata.seq_lens_list = [10, 12, 3]
        self.metadata.block_tables = torch.arange(12, dtype=torch.int64).reshape(3, 4)
        self.metadata.attn_mask = None
        self.metadata.causal = True

    def _quant5_call(self, calls):
        return next((kw for kw in calls if kw.get("quant_mode") == C8_FIA_PSEUDO_QUANT_MODE), None)

    @patch("torch_npu.npu_fused_infer_attention_score", create=True)
    def test_multi_token_decode_routes_to_tnd_quant5(self, mock_fia):
        calls: list[dict] = []
        mock_fia.side_effect = _fake_fia_recorder(calls)
        query = torch.randn(6, NUM_HEADS, HEAD_SIZE)
        output = torch.empty_like(query)
        self.metadata.attn_state = attn_module.AscendAttentionState.DecodeOnly

        returned = self.impl._forward_c8_decode(query, self.metadata, output, self.layer)

        self.assertIs(returned, output)
        self.assertEqual(len(calls), 1)
        kw = self._quant5_call(calls)
        self.assertIsNotNone(kw)
        self.assertEqual(kw["input_layout"], "TND")
        self.assertEqual(kw["quant_mode"], C8_FIA_PSEUDO_QUANT_MODE)
        self.assertEqual(kw["sparse_mode"], 3)
        self.assertTrue(kw["inner_precise"])
        self.assertEqual(kw["actual_seq_lengths"], [3, 6])
        self.assertEqual(kw["actual_seq_lengths_kv"], [10, 12])
        # Decode segment slices: query rows and KV batch rows.
        self.assertEqual(kw["query"].shape[0], 6)
        self.assertEqual(kw["block_table"].shape[0], 2)
        # Paged INT8 KV is passed through as the NZ view (no dequant copy):
        # a strided view keeps the cache dtype.
        self.assertEqual(kw["key"].dtype, torch.int8)
        self.assertEqual(kw["value"].dtype, torch.int8)
        self.assertTrue(torch.equal(output[:6], torch.zeros(6, NUM_HEADS, HEAD_SIZE)))

    @patch("torch_npu.npu_fused_infer_attention_score", create=True)
    def test_single_token_decode_keeps_bnsd(self, mock_fia):
        calls: list[dict] = []
        mock_fia.side_effect = _fake_fia_recorder(calls)
        query = torch.randn(2, NUM_HEADS, HEAD_SIZE)
        output = torch.empty_like(query)
        self.metadata.attn_state = attn_module.AscendAttentionState.DecodeOnly
        self.metadata.num_decodes = 2
        self.metadata.num_decode_tokens = 2
        self.metadata.seq_lens_list = [10, 12]

        self.impl._forward_c8_decode(query, self.metadata, output, self.layer)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["input_layout"], "BNSD")
        self.assertNotIn("quant_mode", calls[0])

    @patch("torch_npu.npu_fused_infer_attention_score", create=True)
    def test_chunked_prefill_mixed_batch_runs_decode_segment_and_prefill(self, mock_fia):
        calls: list[dict] = []
        mock_fia.side_effect = _fake_fia_recorder(calls)
        query = torch.randn(9, NUM_HEADS, HEAD_SIZE)
        output = torch.empty_like(query)
        float_key = torch.randn(9, NUM_KV_HEADS, HEAD_SIZE)
        float_value = torch.randn(9, NUM_KV_HEADS, HEAD_SIZE)
        self.metadata.attn_state = attn_module.AscendAttentionState.ChunkedPrefill
        self.metadata.num_prefills = 1

        returned = self.impl._forward_c8_chunked_prefill(
            query, float_key, float_value, self.metadata, output, self.layer
        )

        self.assertIs(returned, output)
        self.assertEqual(len(calls), 2)
        kw = self._quant5_call(calls)
        self.assertIsNotNone(kw)
        self.assertEqual(kw["input_layout"], "TND")
        self.assertEqual(kw["query"].shape[0], 6)
        # Prefill keeps its own float-KV TND call without antiquant args.
        prefill_kw = calls[-1]
        self.assertNotIn("quant_mode", prefill_kw)
        self.assertNotIn("key_antiquant_scale", prefill_kw)
        self.assertIs(prefill_kw["block_table"], None)
        self.assertEqual(prefill_kw["key"].dtype, torch.float32)

    @patch("torch_npu.npu_fused_infer_attention_score", create=True)
    def test_quant5_rejection_falls_back_and_persists(self, mock_fia):
        calls: list[dict] = []

        def rejecting_fia(*args, **kwargs):
            calls.append(kwargs)
            if kwargs.get("quant_mode") == C8_FIA_PSEUDO_QUANT_MODE:
                raise RuntimeError("quant_mode not supported")
            query = kwargs.get("query")
            if query is None:
                query = args[0]
            return torch.zeros(query.shape[0], NUM_HEADS, HEAD_SIZE), None

        mock_fia.side_effect = rejecting_fia
        query = torch.randn(6, NUM_HEADS, HEAD_SIZE)
        output = torch.empty_like(query)
        self.metadata.attn_state = attn_module.AscendAttentionState.DecodeOnly

        self.impl._forward_c8_decode(query, self.metadata, output, self.layer)
        self.assertFalse(self.impl._c8_fia_quant5_supported)
        self.assertEqual(len(calls), 2)
        fallback_kw = calls[-1]
        self.assertNotIn("quant_mode", fallback_kw)
        # Gather+dequant fallback hands dense float KV without a block table.
        self.assertIs(fallback_kw["block_table"], None)
        self.assertEqual(fallback_kw["key"].dtype, torch.float32)
        self.assertEqual(fallback_kw["input_layout"], "TND")
        self.assertEqual(fallback_kw["sparse_mode"], 3)

        # The next step skips the rejected quant_mode=5 call entirely.
        self.impl._forward_c8_decode(query, self.metadata, output, self.layer)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(kw.get("quant_mode") != C8_FIA_PSEUDO_QUANT_MODE for kw in calls[2:]))

    def _capture_ready_layer(self):
        layer = MagicMock()
        layer.layer_name = "test_layer"
        layer.k_cache_scale = torch.nn.Parameter(torch.ones(1))
        layer.k_cache_offset = torch.nn.Parameter(torch.zeros(1))
        layer.v_cache_scale = torch.nn.Parameter(torch.ones(1))
        layer.v_cache_offset = torch.nn.Parameter(torch.zeros(1))
        layer._c8_k_aq_scale_nz_bnsd = torch.ones(1, NUM_KV_HEADS, HEAD_SIZE)
        layer._c8_v_aq_scale_nz_bnsd = torch.ones(1, NUM_KV_HEADS, HEAD_SIZE)
        # MagicMock auto-attributes make hasattr(layer, "_c8_scales_prepared")
        # always True, so _prepare_c8_scales early-returns: precompute the
        # full set of real scales it would have derived and mark it prepared.
        k_scale = torch.ones(1, NUM_KV_HEADS, HEAD_SIZE)
        v_scale = torch.ones(1, NUM_KV_HEADS, HEAD_SIZE)
        layer._c8_k_scale = k_scale
        layer._c8_v_scale = v_scale
        layer._c8_k_offset = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE)
        layer._c8_v_offset = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE)
        layer._c8_k_inv_scale = 1.0 / k_scale
        layer._c8_v_inv_scale = 1.0 / v_scale
        layer._c8_scales_prepared = True
        return layer

    @patch(
        "vllm_ascend.ascend_forward_context.get_forward_context",
        return_value=MagicMock(capturing=True),
    )
    def test_graph_capture_multi_token_decode_is_rejected(self, _mock_ctx):
        # Through the public forward(): under graph capture the C8 branch of
        # full_graph_fia would use the one-query-row BNSD call, so MTP-shaped
        # capture must be rejected with an actionable error instead.
        self.mock_vllm_config.kv_transfer_config = None
        layer = self._capture_ready_layer()
        query = torch.randn(9, NUM_HEADS, HEAD_SIZE)
        key = torch.randn(9, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(9, NUM_KV_HEADS, HEAD_SIZE)
        output = torch.empty_like(query)
        self.metadata.num_actual_tokens = 9
        self.metadata.num_decodes = 2
        self.metadata.num_decode_tokens = 6
        self.metadata.model_runner_type = "generate"
        self.metadata.attn_state = attn_module.AscendAttentionState.PrefillNoCache

        with self.assertRaises(NotImplementedError) as ctx:
            self.impl.forward(layer, query, key, value, (), self.metadata, output)
        self.assertIn("ACL graph", str(ctx.exception))

    @patch(
        "vllm_ascend.ascend_forward_context.get_forward_context",
        return_value=MagicMock(capturing=True),
    )
    @patch("torch_npu.npu_fused_infer_attention_score.out", create=True)
    def test_graph_capture_single_token_decode_keeps_c8_bnsd(self, mock_out, _mock_ctx):
        # Non-MTP capture keeps the historical BNSD call unchanged (no
        # quant_mode kwarg, output unsqueezed to 4D).
        self.mock_vllm_config.kv_transfer_config = None
        layer = self._capture_ready_layer()
        self.impl._use_max_workspace_for_fia_graph = False
        query = torch.randn(2, NUM_HEADS, HEAD_SIZE)
        key = torch.randn(2, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(2, NUM_KV_HEADS, HEAD_SIZE)
        output = torch.empty_like(query)
        self.metadata.num_actual_tokens = 2
        self.metadata.num_decodes = 2
        self.metadata.num_decode_tokens = 2
        self.metadata.actual_seq_lengths_q = [1, 2]
        self.metadata.attn_mask = None
        self.metadata.causal = True
        self.metadata.model_runner_type = "generate"
        self.metadata.attn_state = attn_module.AscendAttentionState.PrefillNoCache

        returned = self.impl.forward(layer, query, key, value, (), self.metadata, output)

        self.assertIs(returned, output)
        mock_out.assert_called_once()
        kw = mock_out.call_args.kwargs
        self.assertEqual(kw["input_layout"], "BNSD")
        self.assertNotIn("quant_mode", kw)
        self.assertIsNone(kw["atten_mask"])
        self.assertEqual(kw["out"][0].dim(), 4)

    def test_fia_param_provider_rejects_multi_token_replay_for_c8_bnsd(self):
        provider = FIAParamProvider(layer_name="test_layer", sliding_window=None, is_c8_bnsd=True)
        multi_token = MagicMock()
        multi_token.num_decode_tokens = 6
        multi_token.num_decodes = 2
        with self.assertRaises(NotImplementedError) as ctx:
            provider.resolve({"test_layer": multi_token})
        self.assertIn("ACL graph", str(ctx.exception))

        # Single-token replay resolves normally.
        single_token = MagicMock()
        single_token.num_decode_tokens = 2
        single_token.num_decodes = 2
        single_token.actual_seq_lengths_q = [1, 2]
        single_token.seq_lens_list = [10, 12]
        single_token.block_tables = torch.zeros(2, 4)
        resolved = provider.resolve({"test_layer": single_token})
        self.assertEqual(resolved["actual_seq_lengths"], [1, 2])

        # The base TND tasks (is_c8_bnsd=False) are unaffected by MTP replays.
        base_provider = FIAParamProvider(layer_name="test_layer", sliding_window=None)
        resolved = base_provider.resolve({"test_layer": multi_token})
        self.assertIn("block_table", resolved)
