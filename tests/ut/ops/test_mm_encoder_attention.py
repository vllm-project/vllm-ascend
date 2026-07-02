from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F
from vllm.config import CompilationConfig, VllmConfig
from vllm.config.vllm import get_cached_compilation_config

from tests.ut.base import TestBase
from vllm_ascend.ops.mm_encoder_attention import (
    FIA_BLOCK_SIZE,
    MAX_PAD_SIZE,
    AscendMMEncoderAttention,
)
from vllm_ascend.worker import encoder_acl_graph
from vllm_ascend.worker.encoder_acl_graph import (
    get_encoder_forward_context,
    get_encoder_graph_params,
    set_encoder_graph_params,
)


class FIAMockMixin(TestBase):
    captured: dict[str, Any]

    def _install_vllm_config_mock(self):
        mock_vllm_config = MagicMock(spec=VllmConfig)
        mock_vllm_config.compilation_config = CompilationConfig()
        patcher = patch(
            "vllm.config.vllm.get_current_vllm_config",
            return_value=mock_vllm_config,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        get_cached_compilation_config.cache_clear()
        self.addCleanup(get_cached_compilation_config.cache_clear)

    def _make_layer(self, num_heads=4, num_kv_heads=4, head_size=72, scale=None):
        return AscendMMEncoderAttention(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
        )

    def _compute_fia_output(self, query, key, value, actual_seq_lengths, scale):
        output = torch.zeros_like(query)
        starts = [0] + list(actual_seq_lengths[:-1])
        for i, end in enumerate(actual_seq_lengths):
            beg = starts[i]
            if end <= beg:
                continue
            q_seg = query[beg:end].permute(1, 0, 2).to(torch.float32)
            k_seg = key[beg:end].permute(1, 0, 2).to(torch.float32)
            v_seg = value[beg:end].permute(1, 0, 2).to(torch.float32)
            attn = F.scaled_dot_product_attention(
                q_seg.unsqueeze(0),
                k_seg.unsqueeze(0),
                v_seg.unsqueeze(0),
                scale=scale,
                dropout_p=0.0,
                is_causal=False,
            ).squeeze(0)
            output[beg:end] = attn.permute(1, 0, 2).to(query.dtype)
        return output

    def _record_fia_call(self, **kwargs):
        self.captured["mode"] = "functional"
        self.captured["q_shape"] = kwargs["query"].shape
        self.captured["k_shape"] = kwargs["key"].shape
        self.captured["v_shape"] = kwargs["value"].shape
        self.captured["input_layout"] = kwargs["input_layout"]
        self.captured["block_size"] = kwargs["block_size"]
        self.captured["actual_seq_lengths"] = kwargs["actual_seq_lengths"]
        self.captured["actual_seq_lengths_kv"] = kwargs["actual_seq_lengths_kv"]
        self.captured["num_heads"] = kwargs["num_heads"]
        self.captured["num_key_value_heads"] = kwargs["num_key_value_heads"]
        self.captured["scale"] = kwargs["scale"]
        self.captured["sparse_mode"] = kwargs["sparse_mode"]
        self.captured["block_table"] = kwargs["block_table"]
        self.captured["atten_mask"] = kwargs["atten_mask"]
        self.captured["pre_tokens"] = kwargs.get("pre_tokens")
        self.captured["next_tokens"] = kwargs.get("next_tokens")

    def _fake_fia(self, **kwargs):
        self._record_fia_call(**kwargs)
        return (
            self._compute_fia_output(
                kwargs["query"],
                kwargs["key"],
                kwargs["value"],
                kwargs["actual_seq_lengths"],
                kwargs["scale"],
            ),
            None,
        )

    def _fake_fia_out(self, *, workspace, out, **kwargs):
        self.captured["mode"] = "out"
        self.captured["workspace"] = workspace
        self.captured["softmax_lse"] = out[1]
        self.captured["q_shape"] = kwargs["query"].shape
        self.captured["actual_seq_lengths"] = kwargs["actual_seq_lengths"]
        self.captured["block_size"] = kwargs["block_size"]
        self.captured["sparse_mode"] = kwargs["sparse_mode"]
        self.captured["pre_tokens"] = kwargs.get("pre_tokens")
        self.captured["next_tokens"] = kwargs.get("next_tokens")
        self.captured["scale"] = kwargs["scale"]
        out[0][...] = self._compute_fia_output(
            kwargs["query"],
            kwargs["key"],
            kwargs["value"],
            kwargs["actual_seq_lengths"],
            kwargs["scale"],
        )

    def _install_fia_mocks(self, *, capture: bool):
        self.captured = {}
        mock_fia = MagicMock(side_effect=self._fake_fia)
        mock_fia.out = self._fake_fia_out

        patch_targets: list[tuple[str, Any]] = [
            (
                "vllm_ascend.ops.mm_encoder_attention.torch_npu.npu_fused_infer_attention_score",
                mock_fia,
            ),
            (
                "vllm_ascend.ops.mm_encoder_attention.torch_npu._npu_fused_infer_attention_score_get_max_workspace",
                MagicMock(return_value=torch.zeros(1)),
            ),
        ]
        if capture:
            self.mock_graph_begin = MagicMock()
            self.mock_graph_end = MagicMock(return_value=42)
            mock_event = MagicMock()
            patch_targets.extend(
                [
                    (
                        "vllm_ascend.ops.mm_encoder_attention.weak_ref_tensors",
                        lambda tensors: tensors,
                    ),
                    (
                        "vllm_ascend.ops.mm_encoder_attention.torch_npu.npu.current_stream",
                        MagicMock(return_value=MagicMock()),
                    ),
                    (
                        "vllm_ascend.ops.mm_encoder_attention.torch.npu.ExternalEvent",
                        MagicMock(return_value=mock_event),
                    ),
                    (
                        "vllm_ascend.ops.mm_encoder_attention.torch.npu.graph_task_group_begin",
                        self.mock_graph_begin,
                    ),
                    (
                        "vllm_ascend.ops.mm_encoder_attention.torch.npu.graph_task_group_end",
                        self.mock_graph_end,
                    ),
                ]
            )

        for target, replacement in patch_targets:
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)


class TestAscendMMEncoderAttentionEager(FIAMockMixin):
    def setUp(self):
        self._install_vllm_config_mock()
        self._install_fia_mocks(capture=False)

    def test_shape_basic(self):
        layer = self._make_layer(num_heads=4, num_kv_heads=4, head_size=128)
        bsz, q_len = 2, 4
        query = torch.randn(bsz, q_len, layer.num_heads * layer.head_size)
        key = query.clone()
        value = query.clone()
        cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

        out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

        self.assertEqual(out.shape, (bsz, q_len, layer.num_heads * layer.head_size))
        self.assertEqual(self.captured["mode"], "functional")
        self.assertEqual(self.captured["input_layout"], "TND")
        self.assertEqual(self.captured["sparse_mode"], 0)
        self.assertEqual(self.captured["block_size"], FIA_BLOCK_SIZE)
        self.assertEqual(self.captured["scale"], layer.scale)

    def test_maybe_compute_actual_seq_lengths_from_sequence_lengths(self):
        layer = self._make_layer()
        actual_q, actual_kv = layer._maybe_compute_actual_seq_lengths(
            num_query_tokens=8,
            bsz=2,
            q_len=4,
            cu_seqlens=None,
            sequence_lengths=torch.tensor([4, 4], dtype=torch.int64),
        )
        self.assertEqual(actual_q, [4, 8])
        self.assertEqual(actual_kv, [4, 8])

    def test_maybe_compute_actual_seq_lengths_aligns_padded_cu_seqlens(self):
        layer = self._make_layer()
        cu_seqlens = torch.tensor([0, 4, 8, 0, 0], dtype=torch.int32)
        actual_q, actual_kv = layer._maybe_compute_actual_seq_lengths(
            num_query_tokens=8,
            bsz=2,
            q_len=4,
            cu_seqlens=cu_seqlens,
            sequence_lengths=None,
        )
        self.assertEqual(actual_q, [4, 8])
        self.assertEqual(actual_kv, [4, 8])

    def test_variable_seqlens(self):
        layer = self._make_layer(num_heads=4, num_kv_heads=4, head_size=72)
        seq_lens = [3, 7, 2]
        cu_seqlens = torch.tensor([0, 3, 10, 12], dtype=torch.int32, device="cpu")
        max_q_len = max(seq_lens)
        query = torch.randn(len(seq_lens), max_q_len, layer.num_heads, 72, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

        self.assertEqual(out.shape, query.shape)
        self.assertEqual(self.captured["actual_seq_lengths"], [3, 10, 12, 21])
        self.assertEqual(self.captured["q_shape"], (len(seq_lens) * max_q_len, 4, MAX_PAD_SIZE))

    def test_custom_scale_used_in_fia(self):
        custom_scale = 0.25
        layer = self._make_layer(num_heads=4, num_kv_heads=4, head_size=128, scale=custom_scale)
        bsz, q_len = 1, 4
        query = torch.randn(bsz, q_len, layer.num_heads * layer.head_size)
        key = query.clone()
        value = query.clone()
        cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

        layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

        self.assertEqual(self.captured["scale"], custom_scale)


class TestAscendMMEncoderAttentionCapture(FIAMockMixin):
    def setUp(self):
        self._install_vllm_config_mock()
        set_encoder_graph_params([2048])
        self._install_fia_mocks(capture=True)

    def tearDown(self):
        encoder_acl_graph._encoder_graph_params = None
        encoder_acl_graph._reset_encoder_forward_context()

    def test_capture_appends_attn_params(self):
        layer = self._make_layer(num_heads=4, num_kv_heads=4, head_size=72)
        ctx = get_encoder_forward_context()
        ctx.capturing = True
        ctx.token_budget = 2048
        ctx.capture_layer_cursor = 0

        bsz, q_len = 2, 4
        query = torch.randn(bsz, q_len, layer.num_heads, 72, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

        layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

        params = get_encoder_graph_params()
        self.assertIsNotNone(params)
        self.assertEqual(len(params.attn_params[2048]), 1)
        self.assertEqual(len(params.handles[2048]), 1)
        self.assertFalse(params.attn_params[2048][0][6])
        self.assertEqual(params.attn_params[2048][0][10], layer.scale)
        self.assertEqual(self.captured["mode"], "out")
        self.assertEqual(self.captured["softmax_lse"].numel(), 1)
        self.mock_graph_begin.assert_called_once()
        self.mock_graph_end.assert_called_once()

    def test_capture_uses_sequence_lengths_host(self):
        layer = self._make_layer(num_heads=4, num_kv_heads=4, head_size=72)
        ctx = get_encoder_forward_context()
        ctx.capturing = True
        ctx.token_budget = 2048

        seq_lens = [3, 5]
        sequence_lengths = torch.tensor(seq_lens, dtype=torch.int64)
        query = torch.randn(2, 5, layer.num_heads, 72, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        layer.forward_oot(query, key, value, sequence_lengths=sequence_lengths)

        params = get_encoder_graph_params()
        self.assertIsNotNone(params)
        self.assertTrue(params.attn_params[2048][0][6])
        self.assertEqual(self.captured["actual_seq_lengths"], [3, 8, 10])
