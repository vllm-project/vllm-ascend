# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU accuracy tests for DSpark expert-pool routing (DeepSeek verify batches)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.worker.v2.dspark_expert_pool import (
    CONTRIBUTOR,
    RESTRICTED,
    TOP_K,
    DSparkExpertPool,
)

DEVICE = "npu"
SUPPORTED_DEVICES = (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5)
EXPERTS, SUFFIX_EXPERTS, SCALING = 256, 2, 2.5
RTOL, ATOL = 2**-16, 2**-10  # documented weight tolerance vs the fp32 torch reference


def _report(name, actual, expected, *, exact):
    """Print max_abs/max_rel error (or mismatch counts) before the assertion."""
    actual_cpu, expected_cpu = actual.detach().cpu(), expected.detach().cpu()
    if exact:
        mismatches = torch.count_nonzero(actual_cpu != expected_cpu).item()
        print(f"[precision] {name}: mismatches={mismatches}, total={expected_cpu.numel()}")
        return
    actual_cpu, expected_cpu = actual_cpu.float(), expected_cpu.float()
    abs_error = (actual_cpu - expected_cpu).abs()
    rel_error = abs_error / expected_cpu.abs().clamp_min(torch.finfo(torch.float32).eps)
    print(
        f"[precision] {name}: max_abs_error={abs_error.max().item():.8e}, "
        f"max_rel_error={rel_error.max().item():.8e}, rtol={RTOL:.8e}, atol={ATOL:.8e}"
    )


def _reference(logits, roles, bias, suffix_experts, scaling):
    """Pure-torch twin of _build_pool/_finish_route: identical ties and normalization.

    Selection ranks raw + bias (raw = sqrt(log(exp(x) + 1))) and breaks ties on the
    lowest expert id. Pool votes: contributors add their top-K, restricted rows their
    top-`suffix_experts`, role-0 rows nothing. Final routing ranks the pooled experts
    for restricted rows (for other rows ranking all experts is equivalent, since a
    contributor's own votes always sit inside the pool). Weights renormalize the RAW
    (unbiased) scores: weight[i] = raw(id[i]) / max(sum_j raw(id[j]), 1e-20) * scaling.
    """
    raw = torch.sqrt(torch.log(torch.exp(logits.float()) + 1.0))
    order = torch.sort(raw + bias.float(), dim=1, descending=True, stable=True).indices
    vote_limits = {CONTRIBUTOR: TOP_K, RESTRICTED: suffix_experts}
    pool = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
    for row, role in enumerate(roles.tolist()):
        if vote_limits.get(role):
            pool[order[row, : vote_limits[role]]] = True
    ids = torch.empty((logits.shape[0], TOP_K), dtype=torch.int64, device=logits.device)
    weights = torch.empty_like(ids, dtype=torch.float32)
    for row, role in enumerate(roles.tolist()):
        ranked = order[row][pool[order[row]]] if role == RESTRICTED else order[row]
        ids[row] = ranked[:TOP_K]
        top = raw[row, ranked[:TOP_K]]
        weights[row] = top / top.sum().clamp_min(1e-20) * scaling
    return weights, ids.to(torch.int32)


def _make_pool(padded_rows):
    return DSparkExpertPool(
        torch.device(DEVICE),
        capacity=padded_rows,
        protected_positions=2,
        suffix_experts=SUFFIX_EXPERTS,
        num_speculative_tokens=5,
        num_device_experts=84,
    )


def _make_batch(boundaries, drafts, padded_rows):
    return SimpleNamespace(
        num_reqs=len(boundaries) - 1,
        num_tokens=boundaries[-1],
        num_tokens_after_padding=padded_rows,
        query_start_loc_np=np.array(boundaries),
        num_draft_tokens_per_req=drafts,
        is_prefilling_np=[False] * (len(boundaries) - 1),
        has_structured_output_reqs=False,
    )


def _router(bias):
    return SimpleNamespace(num_logical_experts=EXPERTS, e_score_correction_bias=bias, routed_scaling_factor=SCALING)


@pytest.mark.skipif(
    get_ascend_device_type() not in SUPPORTED_DEVICES,
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("reqs", [2, 3])
def test_dspark_pool_routing_matches_reference_and_handles_strides(reqs):
    torch.manual_seed(1)
    rows = 6 * reqs  # 12 rows reuse the cached workspace; 18 rows take the fresh-buffer path.
    pool = _make_pool(rows)
    strided = torch.randn(rows, EXPERTS * 2, dtype=torch.float32, device=DEVICE)[:, ::2]
    strided[0, 0] = strided[0, 1] = 8.0  # exact tie: the lowest expert id must come first
    bias = torch.randn(EXPERTS, dtype=torch.float32, device=DEVICE) * 0.1
    bias[0] = bias[1] = 0.0
    pool.prepare(_make_batch(list(range(0, rows + 1, 6)), [5] * reqs, rows))
    ref_weights, ref_ids = _reference(strided, pool.roles[:rows], bias, SUFFIX_EXPERTS, SCALING)
    outs = [pool.route(_router(bias), x) for x in (strided, strided.contiguous())]
    for name, (weights, ids) in zip(("strided", "contiguous"), outs):
        _report(f"dspark_expert_ids[{name}]", ids, ref_ids, exact=True)
        _report(f"dspark_weights[{name}]", weights, ref_weights, exact=False)
        torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, ref_weights, rtol=RTOL, atol=ATOL)
    # Strided logits are honoured directly: the kernels index via row/expert strides.
    assert outs[0][0].equal(outs[1][0]) and outs[0][1].equal(outs[1][1])


@pytest.mark.skipif(
    get_ascend_device_type() not in SUPPORTED_DEVICES,
    reason="Requires Ascend A2/A3/A5",
)
def test_dspark_padding_rows_neither_extend_the_pool_nor_get_routed():
    torch.manual_seed(7)
    real, pad = 12, 4
    pool = _make_pool(real + pad)
    logits = torch.randn(real + pad, EXPERTS, dtype=torch.float32, device=DEVICE)
    # Real rows are kept off experts 198-206 so only padding rows favour them; if padding
    # could vote, expert 200 would pool-enter and reroute restricted row 2, which ranks it
    # just below its own two suffix votes (210, 211).
    logits[:real, 198:207] = -6.0
    logits[real:, 200:206] = 20.0
    logits[2, 200], logits[2, 210], logits[2, 211] = 9.0, 12.0, 11.5
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=DEVICE)
    pool.prepare(_make_batch([0, 6, 12], [5, 5], real + pad))
    expected_roles = [CONTRIBUTOR] * 2 + [RESTRICTED] * 4
    assert pool.roles[: real + pad].tolist() == expected_roles * 2 + [0] * pad
    fused = pool.route(_router(bias), logits)
    ref_weights, ref_ids = _reference(logits, pool.roles[: real + pad], bias, SUFFIX_EXPERTS, SCALING)
    _report("dspark_expert_ids[with_padding]", fused[1], ref_ids, exact=True)
    _report("dspark_weights[with_padding]", fused[0], ref_weights, exact=False)
    torch.testing.assert_close(fused[1], ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(fused[0], ref_weights, rtol=RTOL, atol=ATOL)
    # Outside verification the pool path is bypassed: merge_native keeps the native route.
    native = (
        torch.full((real + pad, TOP_K), 0.5, device=DEVICE),
        torch.full((real + pad, TOP_K), 999, dtype=torch.int32, device=DEVICE),
    )
    merged = pool.merge_native(fused, native)
    assert merged[0][real:].equal(native[0][real:]) and merged[1][real:].equal(native[1][real:])
    assert merged[0][:real].equal(fused[0][:real]) and merged[1][:real].equal(fused[1][:real])
    # Rerouting without the padding rows must be bit-identical for every real row.
    pool.prepare(_make_batch([0, 6, 12], [5, 5], real))
    trimmed = pool.route(_router(bias), logits[:real])
    assert fused[0][:real].equal(trimmed[0]) and fused[1][:real].equal(trimmed[1])
