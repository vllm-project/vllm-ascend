# SPDX-License-Identifier: Apache-2.0
"""Validate packed rows and exact checkpoint handoff against a dense oracle."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[3] / "vllm_ascend/ops" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLAN = _load_module("fused_checkpoint_plan_under_test", "gdn_ascendc_checkpoint.py")


def _dense(q, k, v, g, beta, initial_state, cu_seqlens):
    output = torch.empty_like(v)
    final = initial_state.clone()
    for row, (begin, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        for token in range(begin, end):
            update = v[0, token, :, :, None] * k[0, token, :, None, :]
            final[row] = final[row] * g[0, token].exp()[:, None, None] + update * beta[0, token, :, None, None]
            output[0, token] = torch.einsum("hvk,hk->hv", final[row], q[0, token])
    return output, final


@pytest.mark.parametrize(
    "lengths,offsets", [([1464], [1408]), ([1336], [1280]), ([315], [256]), ([145, 0, 65, 209], [128, 0, 0, 192])]
)
def test_fused_segments_match_dense_recurrence_and_prefix_snapshot(lengths, offsets):
    torch.manual_seed(37)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    q, k, v = [torch.randn(1, cu[-1], 2, 2) * 0.1 for _ in range(3)]
    g = -torch.rand(1, cu[-1], 2) * 0.01
    beta = torch.rand_like(g)
    initial = torch.randn(len(lengths), 2, 2, 2) * 0.01
    snapshots = torch.full_like(initial, float("nan"))
    plan = _PLAN.build_ascendc_checkpoint_plan(tuple(cu), offsets, torch.device("cpu"))

    def fake_fused(q, k, v, g, beta, initial_state, scale, prebuilt_meta, fused_fwd):
        return _dense(q, k, v, g, beta, initial_state, prebuilt_meta.cu_seqlens_host)

    device = SimpleNamespace(DeviceOperator=SimpleNamespace(fla_gdn_prefill=fake_fused))
    with patch.dict(sys.modules, {"vllm_ascend.device.device_op": device}):
        module = _load_module("fused_checkpoint_under_test", "gdn_fused_checkpoint.py")
    output, final = module.fused_gdn_checkpoint_prefill(q, k, v, g, beta, initial, snapshots, plan, object())
    expected_output, expected_final = _dense(q, k, v, g, beta, initial, cu)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(final, expected_final, rtol=0, atol=0)
    for row, offset in enumerate(offsets):
        if not offset:
            assert snapshots[row].isnan().all()
            continue
        stop = cu[row] + offset
        inputs = [tensor[:, cu[row] : stop] for tensor in (q, k, v, g, beta)]
        _, expected = _dense(*inputs, initial[row : row + 1], [0, offset])
        torch.testing.assert_close(snapshots[row], expected[0], rtol=0, atol=0)
