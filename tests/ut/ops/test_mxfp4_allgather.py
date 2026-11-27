# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

import vllm_ascend.ops.fused_moe.prepare_finalize as prepare_module
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.quantization.quant_type import QuantType


def _quantize_rows(x, **kwargs):
    # A CPU wire-format stub, not a numerical model of MXFP4. Distinct row
    # markers detect a scale/payload permutation; zero rows have nonzero scales.
    packed = x[:, ::2].contiguous().to(torch.uint8)
    scale = (x[:, 0].to(torch.uint8) + 127).reshape(-1, 1, 1).expand(-1, x.shape[1] // 64, 2).clone()
    return packed, scale


class TestMXFP4AllGather(TestCase):
    def _prepare(self, dp, pcp, dtype, rows):
        layer = object.__new__(PrepareAndFinalizeWithAllGather)
        layer.moe_config = SimpleNamespace(is_sequence_parallel=False, dp_size=dp, pcp_size=pcp)
        x = torch.full((rows, 128), 3, dtype=dtype)
        logits = torch.full((rows, 8), 5.0)
        dp_rows = 5 if dp > 1 else rows
        pcp_rows = dp_rows * dp + (2 if pcp > 1 else 0)
        padded = torch.nn.functional.pad(x, (0, 0, 0, dp_rows - rows))
        padded_logits = torch.nn.functional.pad(logits, (0, 0, 0, dp_rows - rows))
        payload, scales = _quantize_rows(padded)
        local = [payload, padded_logits, scales]
        after_dp = [torch.cat((t, t + 1)) for t in local] if dp > 1 else local
        events = []

        def gather_dp(tensor, dim=0):
            index = len(events)
            torch.testing.assert_close(tensor, local[index], rtol=0, atol=0)
            events.append("dp")
            # A different peer contribution makes a row-order mistake visible.
            return after_dp[index]

        layer.moe_config.dp_group = SimpleNamespace(all_gather=gather_dp)
        expected = after_dp
        pcp_calls = []
        if pcp > 1:
            zero_x, zero_scale = _quantize_rows(torch.zeros(2, 128, dtype=dtype))
            expected = [
                torch.cat((after_dp[0], zero_x)),
                torch.nn.functional.pad(after_dp[1], (0, 0, 0, 2)),
                torch.cat((after_dp[2], zero_scale)),
            ]

        def gather_pcp(tensor, dim=0):
            index = len(pcp_calls)
            torch.testing.assert_close(tensor, expected[index], rtol=0, atol=0)
            pcp_calls.append("pcp")
            return torch.cat((tensor, tensor + 2), dim=dim)

        context = SimpleNamespace(max_tokens_across_dp=dp_rows, max_tokens_across_pcp=pcp_rows)
        with (
            patch.object(prepare_module, "_EXTRA_CTX", context),
            patch.object(prepare_module, "get_pcp_group", return_value=SimpleNamespace(all_gather=gather_pcp)),
            patch.object(prepare_module.torch_npu, "npu_dynamic_mx_quant", side_effect=_quantize_rows, create=True),
        ):
            actual = layer.prepare(x, logits, quant_type=QuantType.W4A4MXFP)
        for result, reference in zip((actual.hidden_states, actual.router_logits, actual.pertoken_scale), expected):
            if pcp > 1:
                reference = torch.cat((reference, reference + 2))
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        if dp > 1:
            self.assertEqual(layer.num_tokens, rows)
        if pcp > 1:
            self.assertEqual(layer.num_tokens_pcp, dp_rows * dp)
        self.assertEqual(len(events), 3 if dp > 1 else 0)
        self.assertEqual(len(pcp_calls), 3 if pcp > 1 else 0)
        torch.testing.assert_close(x, torch.full_like(x, 3))

    def test_payload_scale_and_logits_alignment(self):
        for dp, pcp in ((2, 1), (1, 2), (2, 2)):
            for dtype in (torch.bfloat16, torch.float16):
                for rows in (0, 3, 5):
                    with self.subTest(dp=dp, pcp=pcp, dtype=dtype, rows=rows):
                        self._prepare(dp, pcp, dtype, rows)

    def test_empty_quantization_uses_nonempty_input_and_typed_empty_views(self):
        quant = Mock(side_effect=_quantize_rows)
        with patch.object(prepare_module.torch_npu, "npu_dynamic_mx_quant", quant, create=True):
            payload, scales = PrepareAndFinalizeWithAllGather._quantize_mxfp4_for_allgather(
                torch.empty(0, 128, dtype=torch.bfloat16)
            )
        self.assertEqual(quant.call_args.args[0].shape, (1, 128))
        self.assertEqual(payload.shape, (0, 64))
        self.assertEqual(scales.shape, (0, 2, 2))
        self.assertEqual(payload.dtype, torch.uint8)

    def test_noncontiguous_input_and_original_rounding(self):
        quant = Mock(side_effect=_quantize_rows)
        with patch.object(prepare_module.torch_npu, "npu_dynamic_mx_quant", quant, create=True):
            PrepareAndFinalizeWithAllGather._quantize_mxfp4_for_allgather(torch.randn(128, 3, dtype=torch.float16).T)
        self.assertTrue(quant.call_args.args[0].is_contiguous())
        self.assertNotIn("round_mode", quant.call_args.kwargs)

    def test_invalid_activations_fail_before_quantization(self):
        with patch.object(prepare_module.torch_npu, "npu_dynamic_mx_quant", create=True) as quant:
            for x in (torch.zeros(3, 128), torch.zeros(3, 2, 64, dtype=torch.bfloat16)):
                with self.assertRaisesRegex(ValueError, "BF16/FP16"):
                    PrepareAndFinalizeWithAllGather._quantize_mxfp4_for_allgather(x)
            quant.assert_not_called()

    def test_sequence_parallel_keeps_existing_path(self):
        layer = object.__new__(PrepareAndFinalizeWithAllGather)
        layer.moe_config = SimpleNamespace(is_sequence_parallel=True)
        with patch.object(layer, "_prepare_with_ep_group", return_value="ep") as prepare:
            x, logits = torch.empty(3, 128), torch.empty(3, 8)
            self.assertEqual(layer.prepare(x, logits, quant_type=QuantType.W4A4MXFP), "ep")
            prepare.assert_called_once_with(x, logits, QuantType.W4A4MXFP)

    def test_other_quantization_keeps_floating_protocol(self):
        layer = object.__new__(PrepareAndFinalizeWithAllGather)
        layer.moe_config = SimpleNamespace(is_sequence_parallel=False, dp_size=1, pcp_size=1)
        x, logits = torch.randn(3, 128), torch.randn(3, 8)
        with patch.object(prepare_module.torch_npu, "npu_dynamic_mx_quant", create=True) as quant:
            for quant_type in (QuantType.NONE, QuantType.W8A8, QuantType.W8A8MXFP):
                actual = layer.prepare(x, logits, quant_type=quant_type)
                self.assertIs(actual.hidden_states, x)
                self.assertIsNone(actual.pertoken_scale)
            quant.assert_not_called()
