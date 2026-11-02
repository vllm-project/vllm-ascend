# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU regressions for the chunked prefill loop and FP32 merge state."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention import mla_v1
from vllm_ascend.attention import utils as attention_utils
from vllm_ascend.attention.mla_v1 import AscendMLAImpl

GROUP_COUNTS = (1, 2, 15, 16, 17, 30, 31, 32, 33, 46, 64, 65)
TOLERANCES = {torch.float32: (2e-5, 1e-4), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}


def _reference(outputs, lses):
    # Independent FP64 reference over all original pairs, never staged.
    values = torch.stack(outputs).double()
    scores = torch.stack(lses).double()
    # CANN masks +/-inf LSEs, propagates NaN LSEs, and multiplies outputs
    # by zero weights without masking NaN/Inf outputs.
    weights = torch.softmax(scores.masked_fill(torch.isinf(scores), -torch.inf), dim=0)
    return (values * weights[..., None]).sum(dim=0)


def _cpu_update(lses, outputs, update_type):
    # Model the native infinity masking and FP32 intermediate LSE rounding.
    scores = torch.stack(lses)
    scores = scores.masked_fill(torch.isinf(scores), -torch.inf)
    maximum = scores.amax(dim=0)
    weights = (scores - maximum).exp()
    total = weights.sum(dim=0)
    output = torch.zeros_like(outputs[0])
    for value, weight in zip(outputs, weights):
        output += value * weight[:, None]
    output /= total[:, None]
    lse = (maximum + total.log()) if update_type else None
    return output, lse


def _prepare_update(lses, previous_anchor):
    # CPU mock of the fused device operation, not the numerical oracle.
    candidates = list(lses)
    if previous_anchor is not None:
        candidates[0] = previous_anchor
    scores = torch.stack(candidates)
    anchor = scores.masked_fill(~torch.isfinite(scores), -torch.inf).amax(0)
    relative = [lse - anchor for lse in lses]
    if previous_anchor is not None:
        relative[0] = lses[0] + (previous_anchor - anchor)
    return tuple(relative), anchor


class TestMLAPrefillMerge(unittest.TestCase):
    def _run(self, outputs, lses, heads, *, layout="HT", skip_empty=False, no_context=False):
        tokens, _, width = outputs[0].shape
        impl = AscendMLAImpl.__new__(AscendMLAImpl)
        impl.num_heads = heads
        impl.v_head_dim = width
        impl.qk_nope_head_dim = 4
        impl.qk_rope_head_dim = 2
        impl.head_padding = int(heads == 3)
        impl.scale = 1.0
        impl.fa_quant_layer = False
        impl.get_context_block_table = lambda meta: torch.zeros(1, 1, dtype=torch.int32)
        impl.get_context_seq_len_npu = lambda index, meta: torch.tensor([2], dtype=torch.int32)
        impl.kv_b_proj = lambda x: (torch.zeros(x.shape[0], heads * (4 + width)),)
        # Include both zero-sized gathers and chunks emptied by reorganization.
        sizes = [2] * (len(outputs) - 1)
        if skip_empty:
            sizes = [0, 2, *sizes, 0]
        impl._reorg_kvcache = lambda key, value, **kw: (
            (key[:0], value[:0]) if skip_empty and kw["chunk_idx"] == 1 else (key, value)
        )
        chunked = SimpleNamespace(
            seq_tot=sizes,
            starts=[torch.zeros(1, dtype=torch.int32)] * len(sizes),
            chunk_actual_seq_lengths_kv_list=[[2]] * len(sizes),
        )
        metadata = SimpleNamespace(
            prefill=SimpleNamespace(chunked_context=None if no_context else chunked, actual_seq_lengths_q=[tokens])
        )
        cache = (torch.zeros(2, 2, 1, 8), torch.zeros(2, 2, 1, 2))

        def lse_layout(lse):
            return lse.T if layout == "HT" else lse[..., None]

        pairs = iter(zip(outputs[1:], lses[1:]))

        def attention(*args, **kwargs):
            value, lse = next(pairs)
            return value, lse_layout(lse)

        calls = []

        def update(lses, values, mode):
            self.assertLessEqual(len(values), 16)
            self.assertEqual(len(lses), len(values))
            self.assertTrue(all(x.dtype == torch.float32 for x in (*lses, *values)))
            calls.append((len(values), mode))
            return _cpu_update(lses, values, mode)

        prefix_lse = lse_layout(lses[0])
        with (
            patch.object(mla_v1, "torch_npu", SimpleNamespace(npu_fused_infer_attention_score=attention)),
            patch.object(attention_utils, "torch_npu", SimpleNamespace(npu_attention_update=update)),
            patch.object(attention_utils, "prepare_attention_update", _prepare_update),
            patch.object(mla_v1.DeviceOperator, "kv_cache_load"),
        ):
            result, result_lse = impl._compute_prefill_context(
                torch.zeros(tokens, heads, 4), torch.zeros(tokens, heads, 2), cache, 2, metadata, outputs[0], prefix_lse
            )
        if len(outputs) == 1 or no_context:
            self.assertIs(result, outputs[0])
            self.assertIs(result_lse, prefix_lse)
            self.assertEqual(calls, [])
        else:
            self.assertIsNone(result_lse)
            self.assertEqual(result.shape, outputs[0].shape)
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.device, outputs[0].device)
            self.assertEqual(sum(mode == 1 for _, mode in calls), (len(outputs) - 1) // 15)
            remaining = (len(outputs) - 1) % 15 + 1
            self.assertEqual(calls[-1], (16, 1) if remaining == 1 else (remaining, 0))
        return result

    def test_group_dtype_layout_matrix(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        for dtype in TOLERANCES:
            for heads, layout in ((1, "HT"), (3, "TH1")):
                for offset in (0, 1000):
                    for count in GROUP_COUNTS:
                        with self.subTest(dtype=dtype, heads=heads, offset=offset, groups=count):
                            generator = torch.Generator().manual_seed(42 + count)
                            outputs = [torch.randn(5, heads, 8, generator=generator).to(dtype) for _ in range(count)]
                            lses = [torch.randn(5, heads, generator=generator) * 5 + offset for _ in range(count)]
                            result = self._run(outputs, lses, heads, layout=layout, skip_empty=True)
                            atol, rtol = TOLERANCES[dtype]
                            torch.testing.assert_close(result.double(), _reference(outputs, lses), atol=atol, rtol=rtol)

    def test_extreme_and_history_sentinel_rows(self):
        for case in ("equal", "dominant", "late_anchor", "history_sentinels", "nan_history"):
            for count in (16, 17, 31, 46, 65):
                with self.subTest(case=case, groups=count):
                    generator = torch.Generator().manual_seed(count)
                    outputs = [torch.randn(5, 3, 8, generator=generator) for _ in range(count)]
                    lses = [torch.full((5, 3), 1000.0) for _ in range(count)]
                    if case == "dominant":
                        for lse in lses[1:]:
                            lse.fill_(-1000)
                    elif case == "late_anchor":
                        for index, lse in enumerate(lses):
                            lse.fill_(-10000 + index * 1000)
                    elif case in ("history_sentinels", "nan_history"):
                        # Production always starts with finite current-chunk LSE.
                        for index, (output, lse) in enumerate(zip(outputs[1:], lses[1:]), 1):
                            lse[0] = -torch.inf
                            lse[1] = torch.inf
                            output[:2] = 0
                            lse[2] = -torch.inf if index < 16 else 1000.25
                            output[2] = 0 if index < 16 else output[2]
                            if case == "nan_history":
                                lse[3] = torch.nan
                                output[4] = torch.nan
                    result = self._run(outputs, lses, 3)
                    torch.testing.assert_close(
                        result.double(), _reference(outputs, lses), atol=2e-5, rtol=1e-4, equal_nan=True
                    )

    def test_signed_infinity_and_nan_masks(self):
        for count in (2, 15, 16, 17, 31, 46, 65):
            for layout in ("HT", "TH1"):
                with self.subTest(groups=count, layout=layout):
                    outputs = [torch.ones(5, 3, 8) for _ in range(count)]
                    lses = [torch.full((5, 3), 1000.0) for _ in range(count)]
                    outputs[0][0] = torch.inf
                    outputs[0][1] = -torch.inf
                    outputs[0][2] = torch.inf
                    outputs[1][2] = -torch.inf
                    outputs[1][3] = torch.inf
                    lses[1][3] = -torch.inf
                    lses[1][4] = torch.nan
                    result = self._run(outputs, lses, 3, layout=layout)
                    expected = _reference(outputs, lses)
                    self.assertTrue(torch.isposinf(expected[0]).all())
                    self.assertTrue(torch.isneginf(expected[1]).all())
                    torch.testing.assert_close(result.double(), expected, atol=2e-5, rtol=1e-4, equal_nan=True)

    def test_empty_and_no_context_preserve_objects(self):
        outputs = [torch.ones(5, 1, 8, dtype=torch.bfloat16)]
        lses = [torch.zeros(5, 1)]
        self._run(outputs, lses, 1)
        self._run(outputs, lses, 1, skip_empty=True)
        self._run(outputs, lses, 1, no_context=True)

    def test_zero_query_tokens(self):
        for count in (2, 16, 17, 31):
            self._run([torch.empty(0, 1, 8)] * count, [torch.empty(0, 1)] * count, 1)

    def test_naive_absolute_intermediate_lse_loses_precision(self):
        generator = torch.Generator().manual_seed(91)
        outputs = [torch.randn(15, 8, generator=generator) * 10 for _ in range(65)]
        lses = [1000 + torch.randn(15, generator=generator) for _ in outputs]
        pending_outputs, pending_lses = [], []
        for output, lse in zip(outputs, lses):
            pending_outputs.append(output)
            pending_lses.append(lse)
            if len(pending_outputs) == 16:
                output, lse = _cpu_update(pending_lses, pending_outputs, 1)
                pending_outputs, pending_lses = [output], [lse]
        naive, _ = _cpu_update(pending_lses, pending_outputs, 0)
        expected = _reference(outputs, lses)
        self.assertFalse(torch.allclose(naive.double(), expected, atol=2e-5, rtol=1e-4))
        actual = self._run([x.reshape(5, 3, 8) for x in outputs], [x.reshape(5, 3) for x in lses], 3)
        torch.testing.assert_close(actual.double().reshape(15, 8), expected, atol=2e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
