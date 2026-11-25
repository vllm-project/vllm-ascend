import sys
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

SFA_MODULE = "vllm_ascend.attention.sfa_v1"


class TestIndexerMultistreamOverlap(TestBase):
    """Unit tests for ``multistream_sfa_indexer_overlap``.

    The optimization splits the indexer pre-processing into two independent
    strands and runs the indexer q projection (``quantize`` + ``quant matmul``)
    on the attention calculation stream:

        main stream: wk_weights_proj(cube) -> k vector part -> c8 quant
        aux stream : q_c quantize(vector) -> q quant matmul(cube)

    These tests pin the stream choreography, the fallback decision and the
    ``q_li`` hand-off between the pre and post processing, without requiring an
    NPU (every device op is mocked).
    """

    @staticmethod
    def _make_impl():
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.has_indexer = True
        impl.layer_name = "model.layers.0.self_attn.attn"
        impl.head_dim = 128
        impl.n_head = 64
        impl.qk_rope_head_dim = impl.head_dim
        impl.is_rope_neox_style = True
        impl.enable_sparse_li_c8 = False
        impl.use_torch_npu_lightning_indexer = False
        impl.enable_indexer_multistream = True
        impl.k_norm = MagicMock(side_effect=lambda t: t)
        return impl

    def test_overlapped_pre_process_stream_choreography(self):
        """q projection is issued on the aux stream, interleaved with the main
        stream work, and the returned q_li is the aux stream result."""
        impl = self._make_impl()
        calls = []

        def rec(name, ret):
            def _side_effect(*args, **kwargs):
                calls.append(name)
                return ret

            return MagicMock(side_effect=_side_effect)

        aux_stream = MagicMock()
        aux_stream.wait_stream.side_effect = lambda stream: calls.append("aux.wait(main)")
        main_stream = MagicMock()
        main_stream.wait_stream.side_effect = lambda stream: calls.append("main.wait(aux)")

        switch_cm = MagicMock()
        switch_cm.__enter__.side_effect = lambda *a: calls.append("enter(aux)")
        switch_cm.__exit__.side_effect = lambda *a: calls.append("exit(aux)")
        npu_stream_switch = MagicMock(return_value=switch_cm)

        impl._indexer_q_quantize = rec("q_quantize", "q_c_quant")
        impl.wk_weights_proj = rec("wk_weights_proj", ("indexer_kw", None))
        impl._indexer_q_matmul = rec("q_matmul", "q_li")
        impl._indexer_pre_vector_part = rec("k_vector_part", "k_li")
        impl._indexer_pre_c8_quant = rec("k_c8_quant", ("k_li_c8", None))

        with (
            patch(f"{SFA_MODULE}.attention_calculation_stream", return_value=aux_stream),
            patch(f"{SFA_MODULE}.npu_stream_switch", npu_stream_switch),
            patch(f"{SFA_MODULE}.torch.npu.current_stream", return_value=main_stream),
        ):
            indexer_kw, k_li, k_li_scale, q_li = impl._indexer_pre_process_overlapped(
                "hidden_states", "q_c", "cos", "sin"
            )

        self.assertEqual(indexer_kw, "indexer_kw")
        self.assertEqual(k_li, "k_li_c8")
        self.assertIsNone(k_li_scale)
        self.assertEqual(q_li, "q_li")

        # The quantization (vector) overlaps with the wk projection (cube), the
        # q matmul (cube) overlaps with the k vector part, and each cube phase
        # is gated by a wait on the other stream.
        self.assertEqual(
            calls,
            [
                "aux.wait(main)",
                "enter(aux)",
                "q_quantize",
                "exit(aux)",
                "wk_weights_proj",
                "aux.wait(main)",
                "enter(aux)",
                "q_matmul",
                "exit(aux)",
                "k_vector_part",
                "main.wait(aux)",
                "k_c8_quant",
            ],
        )
        # Both q-projection kernels ran on the auxiliary stream.
        self.assertEqual(switch_cm.__enter__.call_count, 2)
        self.assertEqual(npu_stream_switch.call_args_list[0].kwargs, {"enabled": True})
        self.assertEqual(aux_stream.wait_stream.call_count, 2)
        self.assertEqual(main_stream.wait_stream.call_count, 1)

    def test_can_overlap_requires_split_table_w8a8(self):
        impl = self._make_impl()
        impl.wq_b = torch.nn.Module()
        self.assertFalse(impl._can_overlap_indexer_q_proj(torch.empty(2, 8)))

        # Static W8A8 exposes the ACLNN quantization parameters used by the split.
        impl.wq_b.aclnn_input_scale = torch.ones(1)
        self.assertTrue(impl._can_overlap_indexer_q_proj(torch.empty(2, 8)))

        # Already quantized / compressed-tensors inputs need no split at all.
        self.assertTrue(impl._can_overlap_indexer_q_proj(torch.empty(2, 8, dtype=torch.int8)))
        self.assertTrue(impl._can_overlap_indexer_q_proj((torch.empty(2, 8), torch.empty(2))))

    @patch("torch.ops.vllm.quantize")
    def test_q_projection_split_matches_static_w8a8(self, mock_quantize):
        impl = self._make_impl()
        wq_b = MagicMock()
        wq_b.params_dtype = torch.bfloat16
        wq_b.ascend_quant_method = ""
        impl.wq_b = wq_b

        q_c = torch.randn(4, 8)
        q_c_quant = torch.empty(4, 8, dtype=torch.int8)
        mock_quantize.return_value = q_c_quant
        indexer_kw = torch.randn(4, 4, dtype=torch.bfloat16)

        with patch(f"{SFA_MODULE}.torch_npu.npu_quant_matmul") as mock_matmul:
            quantized = impl._indexer_q_quantize(q_c)
            impl._indexer_q_matmul(quantized, indexer_kw)

        mock_quantize.assert_called_once_with(
            q_c,
            wq_b.aclnn_input_scale,
            wq_b.aclnn_input_scale_reciprocal,
            wq_b.aclnn_input_offset,
        )
        mock_matmul.assert_called_once()
        self.assertIs(mock_matmul.call_args.args[0], q_c_quant)
        self.assertIs(mock_matmul.call_args.args[1], wq_b.weight)
        self.assertIs(mock_matmul.call_args.args[2], wq_b.deq_scale)
        self.assertIs(mock_matmul.call_args.kwargs["bias"], wq_b.quant_bias)
        self.assertEqual(mock_matmul.call_args.kwargs["output_dtype"], torch.bfloat16)

    @patch("torch.ops.vllm.quantize")
    def test_q_projection_split_passes_through_quantized_input(self, mock_quantize):
        impl = self._make_impl()
        impl.wq_b = MagicMock()
        q_c_int8 = torch.empty(4, 8, dtype=torch.int8)

        self.assertIs(impl._indexer_q_quantize(q_c_int8), q_c_int8)

        q_c_tuple = (torch.empty(4, 8, dtype=torch.int8), torch.empty(4, 1))
        self.assertIs(impl._indexer_q_quantize(q_c_tuple), q_c_tuple)
        mock_quantize.assert_not_called()

    @patch(f"{SFA_MODULE}.HAS_TRITON", False)
    @patch(f"{SFA_MODULE}.torch_npu.npu_rotary_mul", create=True)
    @patch(f"{SFA_MODULE}.DeviceOperator.indexer_select_post_process")
    def test_post_process_consumes_precomputed_q_li(self, mock_device_op, mock_rotary):
        """When q_li is handed over from the overlapped pre-processing, the post
        processing must not project q again."""
        mock_rotary.side_effect = lambda q_li_pe, cos, sin: q_li_pe
        mock_device_op.return_value = torch.zeros(4, 1, 4, dtype=torch.int32)
        impl = self._make_impl()
        impl.wq_b = MagicMock()

        num_tokens = 4
        kw = torch.randn(num_tokens, impl.head_dim + impl.n_head, dtype=torch.bfloat16)
        q_li = torch.randn(num_tokens, impl.n_head, impl.head_dim, dtype=torch.bfloat16)
        cos = torch.randn(num_tokens, impl.qk_rope_head_dim)
        sin = torch.randn(num_tokens, impl.qk_rope_head_dim)

        impl.indexer_select_post_process(
            kw=kw,
            q_c=torch.randn(num_tokens, impl.head_dim),
            kv_cache=(torch.empty(1),),
            attn_metadata=MagicMock(),
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=torch.tensor([num_tokens], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([num_tokens], dtype=torch.int32),
            q_li=q_li,
        )

        impl.wq_b.assert_not_called()
        # DeviceOperator.indexer_select_post_process positional args:
        # (self, q_li, q_li_scale, q_li_shape_ori, weights, kv_cache, ...)
        self.assertTrue(torch.equal(mock_device_op.call_args.args[1], q_li))
        self.assertTrue(torch.equal(mock_device_op.call_args.args[4], kw[:, impl.head_dim :]))
