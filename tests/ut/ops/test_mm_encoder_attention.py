from typing import Any
from unittest.mock import MagicMock, patch

import torch
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

    def _fake_fia(self, **kwargs):
        self.captured = {
            "mode": "functional",
            "q_shape": kwargs["query"].shape,
            "input_layout": kwargs["input_layout"],
            "block_size": kwargs["block_size"],
            "actual_seq_lengths": kwargs["actual_seq_lengths"],
            "scale": kwargs["scale"],
            "sparse_mode": kwargs["sparse_mode"],
        }
        return torch.zeros_like(kwargs["query"]), None

    def _fake_fia_out(self, *, workspace, out, **kwargs):
        self.captured = {"mode": "out", "softmax_lse": out[1]}
        out[0].zero_()

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
        self.assertEqual(params.attn_params[2048][0][8], layer.scale)
        self.assertEqual(self.captured["mode"], "out")
        self.assertEqual(self.captured["softmax_lse"].numel(), 1)
        self.mock_graph_begin.assert_called_once()
        self.mock_graph_end.assert_called_once()
