import sys
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


class TestIndexerSelectKwReuse(TestBase):
    """Regression tests for the ``wk_weights_proj`` de-duplication optimization.

    ``wk_weights_proj`` is a fused GEMM that produces both the indexer key
    ``k_li`` (``kw[:, :head_dim]``) and the top-k scoring ``weights``
    (``kw[:, head_dim:]``) in a single call.  Previously these two helpers each
    invoked ``self.wk_weights_proj(x)`` on the same, unmodified ``hidden_states``,
    so the fused GEMM ran twice and only the two complementary halves were used.

    After the optimization the fused GEMM is hoisted into ``forward``, computed
    once, and the resulting ``kw`` tensor is passed to both helpers.  These tests
    pin that contract:

    1. neither helper calls ``self.wk_weights_proj`` anymore;
    2. ``indexer_select_pre_process`` reads the first half ``kw[:, :head_dim]``;
    3. ``indexer_select_post_process`` reads the second half ``kw[:, head_dim:]``.

    The tests are hardware-agnostic (all NPU ops are mocked), so they run on the
    CPU runner as well as any NPU machine (A2/A3).
    """

    @staticmethod
    def _make_impl():
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.has_indexer = True
        impl.layer_name = "model.layers.0.self_attn.attn"
        impl.head_dim = 128
        impl.n_head = 64
        # qk_rope_head_dim == head_dim makes the rope split produce an empty
        # "nope" part, so the output stays directly comparable to the input slice.
        impl.qk_rope_head_dim = impl.head_dim
        impl.is_rope_neox_style = True
        impl.enable_sparse_li_c8 = False
        impl.use_torch_npu_lightning_indexer = False
        # Identity norm keeps k_li == kw[:, :head_dim] through pre_process.
        impl.k_norm = MagicMock(side_effect=lambda t: t)
        # Must stay uncalled: the redundant projection has been hoisted out.
        impl.wk_weights_proj = MagicMock()
        return impl

    @patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False)
    @patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_rotary_mul", create=True)
    def test_pre_process_slices_first_half_and_skips_wk_weights_proj(self, mock_rotary):
        mock_rotary.side_effect = lambda k_li_pe, cos, sin: k_li_pe
        impl = self._make_impl()

        num_tokens = 4
        # Fused output width == head_dim + n_head (k | weights).
        kw = torch.randn(num_tokens, impl.head_dim + impl.n_head, dtype=torch.bfloat16)
        cos = torch.randn(num_tokens, impl.qk_rope_head_dim)
        sin = torch.randn(num_tokens, impl.qk_rope_head_dim)

        k_li, k_li_scale = impl.indexer_select_pre_process(kw=kw, cos=cos, sin=sin)

        self.assertEqual(k_li.shape, (num_tokens, 1, impl.head_dim))
        self.assertTrue(torch.equal(k_li[:, 0, :], kw[:, : impl.head_dim]))
        self.assertIsNone(k_li_scale)
        impl.wk_weights_proj.assert_not_called()

    @patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False)
    @patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_rotary_mul", create=True)
    @patch("vllm_ascend.attention.sfa_v1.DeviceOperator.indexer_select_post_process")
    def test_post_process_slices_second_half_and_skips_wk_weights_proj(self, mock_device_op, mock_rotary):
        mock_rotary.side_effect = lambda q_li_pe, cos, sin: q_li_pe
        mock_device_op.return_value = torch.zeros(4, 1, 4, dtype=torch.int32)
        impl = self._make_impl()

        num_tokens = 4
        kw = torch.randn(num_tokens, impl.head_dim + impl.n_head, dtype=torch.bfloat16)
        # Plain (non-tuple) q_c exercises the self.wq_b(q_c) branch.
        impl.wq_b = MagicMock(return_value=(torch.randn(num_tokens, impl.n_head, impl.head_dim), None))
        q_c = torch.randn(num_tokens, impl.head_dim)
        cos = torch.randn(num_tokens, impl.qk_rope_head_dim)
        sin = torch.randn(num_tokens, impl.qk_rope_head_dim)
        kv_cache = (torch.empty(1),)
        attn_metadata = MagicMock()
        actual_seq_lengths_query = torch.tensor([num_tokens], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([num_tokens], dtype=torch.int32)

        impl.indexer_select_post_process(
            kw=kw,
            q_c=q_c,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
        )

        impl.wk_weights_proj.assert_not_called()
        # DeviceOperator.indexer_select_post_process positional args:
        # (self, q_li, q_li_scale, q_li_shape_ori, weights, kv_cache, ...)
        weights_arg = mock_device_op.call_args.args[4]
        self.assertTrue(torch.equal(weights_arg, kw[:, impl.head_dim :]))
