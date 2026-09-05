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
"""Unit tests for the FIA v2 sink integration in attention_v1 (phase A).

Phase A only wires the routing and guards behind two default-off switches;
these tests exercise exactly that wiring on CPU mocks. No operator execution,
no model run, no NPU side effects — runtime correctness stays unproven until
the phase B experiments (docs/13 §4 S4/S5).

Guard philosophy mirrors the DSpark integration's delegation pattern: static
layer facts are resolved once in __init__ (_fia_v2_sink_layer_ok) and input
validity beyond those (head topology, dtype, layout) is delegated to the
operators, which receive them as inputs.
"""

from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.attention_v1 as attn_module
from tests.ut.base import TestBase
from vllm.v1.attention.backend import AttentionType
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState
from vllm_ascend.attention.utils import needs_layer_aware_fia_graph_replay


def _make_impl(**kwargs):
    """Build an impl on CPU mocks.

    Supported extra kwargs: quant=True emulates a c8-quant config;
    sink_enabled/graph_enabled patch the module-level switches (read at impl
    __init__) so the routing gate becomes reachable.
    """
    quant = kwargs.pop("quant", False)
    sink_enabled = kwargs.pop("sink_enabled", False)
    graph_enabled = kwargs.pop("graph_enabled", False)
    mock_config = MagicMock()
    mock_config.quant_config = MagicMock() if quant else None
    mock_config.kv_transfer_config = None
    with (
        patch("vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=mock_config),
        patch("vllm_ascend.attention.utils.get_current_vllm_config", return_value=mock_config),
        patch.object(attn_module, "_FIA_V2_SINK_ENABLED", sink_enabled),
        patch.object(attn_module, "_FIA_V2_SINK_GRAPH_ENABLED", graph_enabled),
    ):
        needs_layer_aware_fia_graph_replay.cache_clear()
        defaults = dict(
            num_heads=8,
            head_size=128,
            scale=0.08838834764831845,
            num_kv_heads=4,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="bfloat16",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
        )
        defaults.update(kwargs)
        return AscendAttentionBackendImpl(**defaults)


def _make_metadata(num_reqs=3, state=AscendAttentionState.DecodeOnly):
    meta = MagicMock()
    meta.attn_state = state
    meta.actual_seq_lengths_q = [2, 5, 9][:num_reqs]
    meta.seq_lens = torch.arange(10, 10 + num_reqs)
    meta.query_start_loc = torch.tensor([0, 2, 5, 9][: num_reqs + 1])
    meta.block_tables = torch.zeros((num_reqs, 8), dtype=torch.int32)
    return meta


class TestFiaV2SinkSwitchDefaults(TestBase):
    def test_switches_default_off(self):
        self.assertFalse(attn_module._FIA_V2_SINK_ENABLED)
        self.assertFalse(attn_module._FIA_V2_SINK_GRAPH_ENABLED)

    def test_required_op_names(self):
        self.assertEqual(
            attn_module._FIA_V2_SINK_REQUIRED_OPS,
            (
                "_npu_fused_infer_attention_score_v2_sink_metadata",
                "npu_fused_infer_attention_score_v2_sink",
            ),
        )

    def test_init_is_inert_when_switches_off(self):
        impl = _make_impl()
        self.assertFalse(impl._fia_v2_sink_requested)
        self.assertFalse(impl._fia_v2_sink_layer_ok)


class TestFiaV2SinkLayerGate(TestBase):
    """Static per-layer facts are resolved once at __init__."""

    def test_clean_layer_passes(self):
        impl = _make_impl(sink_enabled=True)
        self.assertTrue(impl._fia_v2_sink_requested)
        self.assertTrue(impl._fia_v2_sink_layer_ok)

    def test_sliding_window_layer(self):
        impl = _make_impl(sliding_window=4096)
        self.assertFalse(impl._fia_v2_sink_layer_ok)
        self.assertFalse(impl._use_fia_v2_sink(_make_metadata()))

    def test_sinks_layer(self):
        # sinks reach the impl through the ctor (`sinks=`), before the gate.
        impl = _make_impl(sinks=torch.randn(1))
        self.assertFalse(impl._fia_v2_sink_layer_ok)
        self.assertFalse(impl._use_fia_v2_sink(_make_metadata()))

    def test_c8_quant_variant(self):
        impl = _make_impl(quant=True)
        self.assertTrue(impl.enable_c8_quant)
        self.assertFalse(impl._fia_v2_sink_layer_ok)

    def test_encoder_decoder(self):
        impl = _make_impl(attn_type=AttentionType.ENCODER_DECODER)
        self.assertFalse(impl._fia_v2_sink_layer_ok)


class TestUseFiaV2SinkRouting(TestBase):
    """Runtime/batch conditions; any mismatch falls back to the official path."""

    def setUp(self):
        self.impl = _make_impl(sink_enabled=True)

    def test_none_metadata(self):
        self.assertFalse(self.impl._use_fia_v2_sink(None))

    def test_prefill_no_cache_state(self):
        self.assertFalse(self.impl._use_fia_v2_sink(_make_metadata(state=AscendAttentionState.PrefillNoCache)))

    def test_seq_lens_shorter_than_batch(self):
        meta = _make_metadata()
        meta.seq_lens = torch.arange(10, dtype=torch.int64)[:1]
        self.assertFalse(self.impl._use_fia_v2_sink(meta))

    def test_query_start_loc_shorter_than_batch(self):
        meta = _make_metadata()
        meta.query_start_loc = torch.tensor([0, 2])
        self.assertFalse(self.impl._use_fia_v2_sink(meta))

    def test_block_table_shorter_than_batch(self):
        meta = _make_metadata()
        meta.block_tables = torch.zeros((1, 8), dtype=torch.int32)
        self.assertFalse(self.impl._use_fia_v2_sink(meta))

    def test_valid_decode_batch(self):
        self.assertTrue(self.impl._use_fia_v2_sink(_make_metadata()))


class TestBuildFiaV2SinkSeqTensors(TestBase):
    def test_uniform_fallback_without_npu_query_start_loc(self):
        # query_start_loc not on NPU -> DSpark uniform arange derivation.
        seq_lens = torch.tensor([904, 1180, 1024, 1290])
        q_loc = torch.tensor([0, 4, 8, 12, 16])
        qlen, kvlen = attn_module._build_fia_v2_sink_seq_tensors(16, 4, seq_lens, q_loc)
        self.assertEqual(qlen.tolist(), [4, 8, 12, 16])
        self.assertEqual(kvlen.tolist(), [904, 1180, 1024, 1290])
        self.assertEqual(qlen.dtype, torch.int64)
        self.assertEqual(kvlen.dtype, torch.int64)

    def test_kv_clamp_keeps_lengths_strictly_positive(self):
        seq_lens = torch.tensor([7, 0, 0])  # padded dummy requests carry 0
        qlen, kvlen = attn_module._build_fia_v2_sink_seq_tensors(9, 3, seq_lens, None)
        self.assertEqual(qlen.tolist(), [3, 6, 9])
        self.assertEqual(kvlen.tolist(), [7, 1, 1])
        self.assertEqual(kvlen.dtype, torch.int64)

    def test_uniform_fallback_rejects_uneven_batch(self):
        seq_lens = torch.tensor([900, 1100])
        with self.assertRaises(RuntimeError):
            attn_module._build_fia_v2_sink_seq_tensors(7, 2, seq_lens, None)


class TestGetOrComputeCache(TestBase):
    def test_computes_once_per_signature(self):
        # A plain attribute holder mimicking ForwardContext (MagicMock would
        # auto-create a non-container attribute and defeat the dict cache).
        holder = type("ForwardContextStub", (), {})()

        def fake_get_forward_context():
            return holder

        calls = []

        def compute():
            # Distinct object per invocation (avoid constant-folded tuples).
            calls.append(1)
            marker = object()
            return marker, "kvlen", "meta"

        with patch.object(attn_module, "get_forward_context", fake_get_forward_context):
            first = attn_module._get_or_compute_fia_v2_sink_inputs(("k", 1), compute)
            second = attn_module._get_or_compute_fia_v2_sink_inputs(("k", 1), compute)
            other = attn_module._get_or_compute_fia_v2_sink_inputs(("k", 2), compute)

        self.assertEqual(len(calls), 2)
        self.assertIs(first, second)
        self.assertIsNot(first[0], other[0])
