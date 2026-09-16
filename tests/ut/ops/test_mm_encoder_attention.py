from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CompilationConfig, VllmConfig
from vllm.config.vllm import get_cached_compilation_config
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tests.ut.base import TestBase
from vllm_ascend.ops import mm_encoder_attention as mm_encoder_attention_module
from vllm_ascend.ops.mm_encoder_attention import (
    MAX_PAD_SIZE,
    AscendMMEncoderAttention,
    _get_or_convert_cu_seqlens_host_lengths,
    peek_cu_seqlens_host_lengths,
    prime_cu_seqlens_host_lengths,
    reset_vit_fusion_stats,
)
from vllm_ascend.worker import encoder_acl_graph
from vllm_ascend.worker.encoder_acl_graph import (
    get_encoder_graph_params,
    set_encoder_forward_context,
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
            "actual_seq_lengths": kwargs["actual_seq_lengths"],
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

    def test_forward_oot_basic(self):
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

    def test_forward_oot_seqlens(self):
        layer = self._make_layer(num_heads=4, num_kv_heads=4, head_size=72)
        seq_lens = [3, 7, 2]
        cu_seqlens = torch.tensor([0, 3, 10, 12], dtype=torch.int32, device="cpu")
        max_q_len = max(seq_lens)
        query = torch.randn(len(seq_lens), max_q_len, layer.num_heads, 72, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        out = layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

        self.assertEqual(out.shape, query.shape)
        self.assertEqual(self.captured["actual_seq_lengths"], [3, 10, 12])
        self.assertEqual(self.captured["q_shape"], (len(seq_lens) * max_q_len, 4, MAX_PAD_SIZE))

    @patch(
        "vllm_ascend.ops.mm_encoder_attention.get_encoder_forward_context",
        return_value=SimpleNamespace(capturing=False),
    )
    @patch("vllm_ascend.ops.mm_encoder_attention.HAS_TRITON", True)
    def test_fused_qkv_rope_pad_guard_accepts_supported_contract(
        self,
        _mock_forward_context,
    ):
        layer = self._make_layer(num_heads=8, num_kv_heads=8, head_size=72)
        device = SimpleNamespace(type="npu")

        qkv = MagicMock(
            ndim=3,
            shape=(41, 1, 1728),
            dtype=torch.bfloat16,
            device=device,
        )
        qkv.is_contiguous.return_value = True
        cos = MagicMock(
            shape=(41, 36),
            dtype=torch.bfloat16,
            device=device,
        )
        cos.is_contiguous.return_value = True
        sin = MagicMock(
            shape=(41, 36),
            dtype=torch.bfloat16,
            device=device,
        )
        sin.is_contiguous.return_value = True

        self.assertTrue(
            layer._can_use_fused_qkv_rope_pad_fia(
                qkv,
                cos,
                sin,
                torch.tensor([0, 41], dtype=torch.int32),
            )
        )

        # Packed multi-sequence inputs (multi-image, multi-frame, window
        # attention) stay on the fused path; FIA separates the sequences via
        # the host lengths derived from cu_seqlens.
        self.assertTrue(
            layer._can_use_fused_qkv_rope_pad_fia(
                qkv,
                cos,
                sin,
                torch.tensor([0, 20, 41], dtype=torch.int32),
            )
        )

        self.assertFalse(
            layer._can_use_fused_qkv_rope_pad_fia(
                qkv,
                cos,
                sin,
                torch.tensor([[0, 41]], dtype=torch.int32),
            )
        )

        qkv.dtype = torch.float16
        self.assertFalse(
            layer._can_use_fused_qkv_rope_pad_fia(
                qkv,
                cos,
                sin,
                torch.tensor([0, 41], dtype=torch.int32),
            )
        )


class TestFusedQkvRopePadHostLengths(TestBase):
    def setUp(self):
        mm_encoder_attention_module._CU_SEQLENS_HOST_CACHE.clear()

    def tearDown(self):
        mm_encoder_attention_module._CU_SEQLENS_HOST_CACHE.clear()

    def test_convert_multi_sequence_lengths(self):
        cu_seqlens = torch.tensor([0, 100, 240, 400], dtype=torch.int32)
        lengths = _get_or_convert_cu_seqlens_host_lengths(cu_seqlens, 400)
        self.assertEqual(lengths, (100, 240, 400))

    def test_cache_reuses_one_conversion(self):
        cu_seqlens = torch.tensor([0, 100, 240, 400], dtype=torch.int32)
        conversions = []

        def fake_cpu(*args, **kwargs):
            conversions.append(1)
            return args[0] if args else cu_seqlens

        with patch.object(torch.Tensor, "cpu", autospec=True, side_effect=fake_cpu):
            first = _get_or_convert_cu_seqlens_host_lengths(cu_seqlens, 400)
            second = _get_or_convert_cu_seqlens_host_lengths(cu_seqlens, 400)

        self.assertEqual(first, (100, 240, 400))
        self.assertEqual(second, (100, 240, 400))
        self.assertEqual(len(conversions), 1)

    def test_prime_skips_conversion(self):
        cu_seqlens = torch.tensor([0, 100, 240, 400], dtype=torch.int32)
        prime_cu_seqlens_host_lengths(cu_seqlens, [100, 240, 400])

        conversions = []

        def fake_cpu(*args, **kwargs):
            conversions.append(1)
            return args[0] if args else cu_seqlens

        with patch.object(torch.Tensor, "cpu", autospec=True, side_effect=fake_cpu):
            lengths = _get_or_convert_cu_seqlens_host_lengths(cu_seqlens, 400)

        self.assertEqual(lengths, (100, 240, 400))
        self.assertEqual(len(conversions), 0)

    def test_invalid_values_return_none(self):
        truncated = torch.tensor([0, 100, 240], dtype=torch.int32)
        self.assertIsNone(_get_or_convert_cu_seqlens_host_lengths(truncated, 400))

        non_monotonic = torch.tensor([0, 240, 100, 400], dtype=torch.int32)
        self.assertIsNone(_get_or_convert_cu_seqlens_host_lengths(non_monotonic, 400))

    def test_maybe_recompute_cu_seqlens_primes_cache(self):
        boundaries = np.array([0, 100, 240, 400], dtype=np.int32)
        result = AscendMMEncoderAttention.maybe_recompute_cu_seqlens(
            AttentionBackendEnum.TORCH_SDPA,
            boundaries,
            1152,
            1,
            torch.device("cpu"),
        )

        self.assertEqual(peek_cu_seqlens_host_lengths(result), (100, 240, 400))
        # The primed tensor needs no conversion when the fused path reads it.
        self.assertEqual(
            _get_or_convert_cu_seqlens_host_lengths(result, 400),
            (100, 240, 400),
        )

    def test_maybe_recompute_cu_seqlens_skips_flashinfer(self):
        boundaries = np.array([0, 100, 240, 400], dtype=np.int32)
        result = AscendMMEncoderAttention.maybe_recompute_cu_seqlens(
            AttentionBackendEnum.FLASHINFER,
            boundaries,
            1152,
            1,
            torch.device("cpu"),
        )

        self.assertIsNone(peek_cu_seqlens_host_lengths(result))


class TestFusedQkvRopePadForward(TestBase):
    def setUp(self):
        mm_encoder_attention_module._CU_SEQLENS_HOST_CACHE.clear()
        reset_vit_fusion_stats()

    def tearDown(self):
        mm_encoder_attention_module._CU_SEQLENS_HOST_CACHE.clear()
        reset_vit_fusion_stats()

    def _make_layer(self):
        self._install_vllm_config_mock()
        return AscendMMEncoderAttention(num_heads=8, num_kv_heads=8, head_size=72)

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

    def _make_forward_mocks(self, token_count: int, boundaries: list[int]):
        device = SimpleNamespace(type="npu")
        qkv = MagicMock(
            ndim=3,
            shape=(token_count, 1, 1728),
            dtype=torch.bfloat16,
            device=device,
        )
        qkv.is_contiguous.return_value = True
        cos = MagicMock(shape=(token_count, 36), dtype=torch.bfloat16, device=device)
        cos.is_contiguous.return_value = True
        sin = MagicMock(shape=(token_count, 36), dtype=torch.bfloat16, device=device)
        sin.is_contiguous.return_value = True
        cu_seqlens = MagicMock(ndim=1)
        cu_seqlens.numel.return_value = len(boundaries)
        cu_seqlens.detach.return_value.cpu.return_value.view.return_value.tolist.return_value = boundaries
        return qkv, cos, sin, cu_seqlens

    @patch(
        "vllm_ascend.ops.mm_encoder_attention.get_encoder_forward_context",
        return_value=SimpleNamespace(capturing=False),
    )
    @patch("vllm_ascend.ops.mm_encoder_attention.HAS_TRITON", True)
    def test_forward_multi_sequence_uses_real_lengths(
        self,
        _mock_forward_context,
    ):
        layer = self._make_layer()
        qkv, cos, sin, cu_seqlens = self._make_forward_mocks(400, [0, 100, 240, 400])

        captured: dict[str, Any] = {}

        def fake_run_fia(query, key, value, lengths_q, lengths_kv):
            captured["lengths_q"] = lengths_q
            captured["lengths_kv"] = lengths_kv
            return MagicMock()

        with (
            patch(
                "vllm_ascend.ops.triton.vision_qkv_rope_pad.vision_qkv_rope_pad",
                return_value=(MagicMock(), MagicMock(), MagicMock()),
            ) as mock_kernel,
            patch.object(layer, "_run_vit_fia", side_effect=fake_run_fia),
        ):
            context = layer.forward_qkv_rope_pad_fia(
                qkv,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                sequence_lengths=None,
            )

        self.assertIsNotNone(context)
        mock_kernel.assert_called_once()
        self.assertEqual(captured["lengths_q"], [100, 240, 400])
        self.assertEqual(captured["lengths_kv"], [100, 240, 400])
        stats = reset_vit_fusion_stats()
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["fused"], 1)

    @patch(
        "vllm_ascend.ops.mm_encoder_attention.get_encoder_forward_context",
        return_value=SimpleNamespace(capturing=False),
    )
    @patch("vllm_ascend.ops.mm_encoder_attention.HAS_TRITON", True)
    def test_forward_single_sequence_avoids_conversion(
        self,
        _mock_forward_context,
    ):
        layer = self._make_layer()
        qkv, cos, sin, cu_seqlens = self._make_forward_mocks(257, [0, 257])

        with (
            patch(
                "vllm_ascend.ops.triton.vision_qkv_rope_pad.vision_qkv_rope_pad",
                return_value=(MagicMock(), MagicMock(), MagicMock()),
            ),
            patch.object(
                layer,
                "_run_vit_fia",
                return_value=MagicMock(),
            ) as mock_fia,
        ):
            context = layer.forward_qkv_rope_pad_fia(
                qkv,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                sequence_lengths=None,
            )

        self.assertIsNotNone(context)
        self.assertEqual(mock_fia.call_args[0][3], [257])
        cu_seqlens.detach.assert_not_called()

    @patch(
        "vllm_ascend.ops.mm_encoder_attention.get_encoder_forward_context",
        return_value=SimpleNamespace(capturing=False),
    )
    @patch("vllm_ascend.ops.mm_encoder_attention.HAS_TRITON", True)
    def test_forward_invalid_boundaries_fall_back(
        self,
        _mock_forward_context,
    ):
        layer = self._make_layer()
        qkv, cos, sin, cu_seqlens = self._make_forward_mocks(400, [0, 100, 240])

        with patch(
            "vllm_ascend.ops.triton.vision_qkv_rope_pad.vision_qkv_rope_pad",
        ) as mock_kernel:
            context = layer.forward_qkv_rope_pad_fia(
                qkv,
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                sequence_lengths=None,
            )

        self.assertIsNone(context)
        mock_kernel.assert_not_called()
        stats = reset_vit_fusion_stats()
        self.assertEqual(stats["fallback:cu_seqlens_values"], 1)

    @patch(
        "vllm_ascend.ops.mm_encoder_attention.get_encoder_forward_context",
        return_value=SimpleNamespace(capturing=False),
    )
    @patch("vllm_ascend.ops.mm_encoder_attention.HAS_TRITON", True)
    def test_forward_records_fallback_reason(
        self,
        _mock_forward_context,
    ):
        layer = self._make_layer()
        device = SimpleNamespace(type="npu")
        qkv = MagicMock(
            ndim=3,
            shape=(41, 1, 1728),
            dtype=torch.bfloat16,
            device=device,
        )
        qkv.is_contiguous.return_value = True

        context = layer.forward_qkv_rope_pad_fia(
            qkv,
            None,
            None,
            cu_seqlens=torch.tensor([0, 41], dtype=torch.int32),
            sequence_lengths=None,
        )

        self.assertIsNone(context)
        stats = reset_vit_fusion_stats()
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["fallback:missing_inputs"], 1)


class TestAscendMMEncoderAttentionCapture(FIAMockMixin):
    def setUp(self):
        self._install_vllm_config_mock()
        set_encoder_graph_params([2048])
        self._install_fia_mocks(capture=True)

    def tearDown(self):
        encoder_acl_graph._encoder_graph_params = None
        encoder_acl_graph._reset_encoder_forward_context()

    def test_forward_oot_basic(self):
        layer = self._make_layer(num_heads=4, num_kv_heads=4, head_size=72)
        bsz, q_len = 2, 4
        query = torch.randn(bsz, q_len, layer.num_heads, 72, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        cu_seqlens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32)

        with set_encoder_forward_context(2048, True):
            layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

        params = get_encoder_graph_params()
        self.assertIsNotNone(params)
        self.assertEqual(len(params.attn_params[2048]), 1)
        self.assertEqual(len(params.handles[2048]), 1)
        self.assertEqual(self.captured["mode"], "out")
        self.mock_graph_begin.assert_called_once()
        self.mock_graph_end.assert_called_once()

    def test_forward_oot_seqlens(self):
        layer = self._make_layer(num_heads=4, num_kv_heads=4, head_size=72)
        seq_lens = [3, 7, 2]
        cu_seqlens = torch.tensor([0, 3, 10, 12], dtype=torch.int32, device="cpu")
        max_q_len = max(seq_lens)
        query = torch.randn(len(seq_lens), max_q_len, layer.num_heads, 72, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        captured_lengths: list[Any] = []

        def capture_workspace(**kwargs):
            captured_lengths.append(kwargs.get("actual_seq_lengths"))
            return torch.zeros(1)

        with (
            patch(
                "vllm_ascend.ops.mm_encoder_attention.torch_npu._npu_fused_infer_attention_score_get_max_workspace",
                side_effect=capture_workspace,
            ),
            set_encoder_forward_context(2048, True),
        ):
            layer.forward_oot(query, key, value, cu_seqlens=cu_seqlens)

        self.assertEqual(captured_lengths[-1], [7, 14, 21])
