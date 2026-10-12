# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops import gdn
from vllm_ascend.ops.gdn_attn_builder import _compact_empty_segments


@pytest.mark.parametrize("lengths", [(129, 3), (0, 129, 0, 3, 0), (0, 0)])
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_fla_compacts_states_and_preserves_empty_rows(lengths, chunk_size):
    cu = (0, *torch.tensor(lengths).cumsum(0).tolist())
    compact_cu, _, keep = _compact_empty_segments(cu, None)
    total = sum(lengths)
    q = torch.zeros(1, total, 1, 128, dtype=torch.bfloat16)
    state = torch.arange(len(lengths) * 128 * 128, dtype=torch.float32).reshape(len(lengths), 1, 128, 128)
    original = state.clone()
    expected_indices = tuple(
        item
        for seq, length in enumerate(length for length in lengths if length)
        for chunk in range((length + chunk_size - 1) // chunk_size)
        for item in (seq, chunk)
    )

    def fake_fla(q, k, v, g, beta, **kwargs):
        assert tuple(kwargs["cu_seqlens"]) == tuple(compact_cu)
        assert kwargs["chunk_indices"] == expected_indices
        expected_state = state if keep is None else state[keep]
        torch.testing.assert_close(kwargs["initial_state"], expected_state.transpose(-1, -2).to(torch.bfloat16))
        return v.clone(), kwargs["initial_state"] + 16

    op = Mock(side_effect=fake_fla)
    ascendc = ModuleType("fla_npu.ops.ascendc")
    ascendc.chunk_gated_delta_rule_fwd = op
    gdn._fla_chunk_indices_host.cache_clear()
    try:
        with (
            patch.dict(sys.modules, {"fla_npu.ops.ascendc": ascendc}),
            patch.object(gdn, "_FLA_CHUNK_SIZE", chunk_size),
            patch.object(gdn, "l2norm_fwd", side_effect=lambda x: x),
        ):
            out, final = gdn.AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                q,
                q,
                q,
                torch.zeros(1, total, 1),
                torch.ones(1, total, 1),
                state,
                tuple(compact_cu),
                128**-0.5,
                keep_meta=keep,
            )
        assert out.shape == q.shape
        torch.testing.assert_close(state, original)
        if total == 0:
            op.assert_not_called()
            torch.testing.assert_close(final, state)
        else:
            op.assert_called_once()
            expected = (state.to(torch.bfloat16) + 16).to(final.dtype)
            if keep is not None:
                expected[~keep] = original[~keep]
            torch.testing.assert_close(final, expected)
    finally:
        gdn._fla_chunk_indices_host.cache_clear()
