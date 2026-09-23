# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


class TestFLAPrefill(unittest.TestCase):
    def test_packed_varlen_layout_and_state_round_trip(self):
        # Unequal state dimensions and non-symmetric data catch missing transposes.
        q = torch.randn(1, 7, 2, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 7, 4, 256, dtype=torch.bfloat16)
        state = torch.randn(2, 4, 256, 128)
        g = -torch.rand(1, 7, 4)
        beta = torch.rand(1, 7, 4)
        output = torch.randn_like(v)
        final = torch.randn(2, 4, 128, 256, dtype=torch.bfloat16)
        op = Mock(return_value=(output, final, None, None, None, None, None, None, None, None))
        module = ModuleType("fla_npu.ops.ascendc")
        module.chunk_gated_delta_rule_fwd = op
        with (
            patch.dict(sys.modules, {"fla_npu.ops.ascendc": module}),
            patch("vllm_ascend.ops.gdn.l2norm_fwd", side_effect=lambda x: torch.nn.functional.normalize(x, dim=-1)),
        ):
            result, result_state = AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                q, q, v, g, beta, state, (0, 3, 7), (0, 0, 1, 0), 128**-0.5
            )
        args, kwargs = op.call_args
        torch.testing.assert_close(args[0], torch.nn.functional.normalize(q, dim=-1))
        self.assertEqual(args[0].shape, q.shape)
        self.assertEqual(args[4].dtype, v.dtype)
        torch.testing.assert_close(kwargs["initial_state"], state.transpose(-1, -2).to(torch.bfloat16))
        self.assertTrue(kwargs["initial_state"].is_contiguous())
        self.assertEqual(kwargs["cu_seqlens"], (0, 3, 7))
        self.assertEqual(kwargs["chunk_indices"], (0, 0, 1, 0))
        self.assertEqual(kwargs["layout"], "TND")
        self.assertEqual(kwargs["chunk_size"], 64)
        self.assertTrue(kwargs["output_final_state"])
        self.assertTrue(kwargs["disable_recompute"])
        self.assertFalse(kwargs["return_intermediate_states"])
        self.assertIs(result, output)
        torch.testing.assert_close(result_state, final.transpose(-1, -2))
        self.assertTrue(result_state.is_contiguous())

    def test_missing_fla_is_cached_as_unavailable(self):
        with (
            patch.object(AscendGatedDeltaNetAttention, "_fused_chunk_available", None),
            patch.dict(sys.modules, {"fla_npu.ops.ascendc": None}),
        ):
            self.assertFalse(AscendGatedDeltaNetAttention._probe_fused_chunk())
            self.assertIs(AscendGatedDeltaNetAttention._fused_chunk_available, False)
            self.assertFalse(AscendGatedDeltaNetAttention._probe_fused_chunk())
