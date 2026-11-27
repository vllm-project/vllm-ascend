# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU accuracy tests for DSpark expert-pool routing (DeepSeek verify batches)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.distributed.expert_transfer.dspark_expert_pool import DSparkExpertPool
from vllm_ascend.ops.triton.deepseek_expert_pool import CONTRIBUTOR, RESTRICTED, TOP_K, route_experts

DEVICE = "npu"
SUPPORTED_DEVICES = (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5)
EXPERTS, SUFFIX_EXPERTS, SCALING = 256, 2, 1.5
RTOL, ATOL = 2**-10, 2**-16
MIN_PASS_RATIO = 0.99
MAX_ABS_ERROR = 0.01


def _assert_routing_matches(name, actual, expected):
    weights, ids = (tensor.cpu() for tensor in actual)
    ref_weights, ref_ids = (tensor.cpu() for tensor in expected)
    assert ids.shape == ref_ids.shape and ids.dtype == ref_ids.dtype
    assert weights.shape == ref_weights.shape and weights.dtype == ref_weights.dtype == torch.float32
    mismatches = torch.count_nonzero(ids != ref_ids).item()
    abs_error = (weights - ref_weights).abs()
    within_tolerance = abs_error <= ATOL + RTOL * ref_weights.abs()
    passed = torch.count_nonzero(within_tolerance).item()
    total = weights.numel()
    max_abs_error = abs_error.max().item()
    print(
        f"[precision] {name}: id_mismatches={mismatches}/{ref_ids.numel()}, "
        f"weights_within_tolerance={passed}/{total} ({passed / total:.2%}), "
        f"max_abs_error={max_abs_error:.8e}, rtol={RTOL:.8e}, atol={ATOL:.8e}"
    )
    assert mismatches == 0
    assert passed / total >= MIN_PASS_RATIO
    assert max_abs_error <= MAX_ABS_ERROR


def _reference(logits, roles, bias, suffix_experts, scaling):
    """Pure-torch twin of _build_pool/_finish_route: identical ties and normalization.

    Selection ranks raw + bias (raw = sqrt(softplus(x))) and breaks ties on the
    lowest expert id. Pool votes: contributors add their top-K, restricted rows their
    top-`suffix_experts`, role-0 rows nothing. Final routing ranks the pooled experts
    for restricted rows (for other rows ranking all experts is equivalent, since a
    contributor's own votes always sit inside the pool). Weights renormalize the RAW
    (unbiased) scores: weight[i] = raw(id[i]) / max(sum_j raw(id[j]), 1e-20) * scaling.
    """
    x = logits.float()
    # Preserve the original small-score rounding; the large-score branch is stable.
    raw = torch.where(x > 20.0, x, torch.log(torch.exp(x.clamp_max(20.0)) + 1.0)).sqrt()
    score = raw + bias.float()
    valid = torch.isfinite(raw) & torch.isfinite(score)
    order = torch.sort(score.masked_fill(~valid, -float("inf")), dim=1, descending=True, stable=True).indices
    vote_limits = {CONTRIBUTOR: TOP_K, RESTRICTED: suffix_experts}
    pool = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
    for row, role in enumerate(roles.tolist()):
        if vote_limits.get(role):
            eligible = order[row][valid[row, order[row]]]
            pool[eligible[: vote_limits[role]]] = True
    ids = torch.zeros((logits.shape[0], TOP_K), dtype=torch.int64, device=logits.device)
    weights = torch.zeros_like(ids, dtype=torch.float32)
    for row, role in enumerate(roles.tolist()):
        ranked = order[row][valid[row, order[row]]]
        if role == RESTRICTED:
            ranked = ranked[pool[ranked]]
        chosen = ranked[:TOP_K]
        ids[row, : chosen.numel()] = chosen
        top = raw[row, chosen]
        weights[row, : chosen.numel()] = top / top.sum().clamp_min(1e-20) * scaling
    return weights, ids.to(torch.int32)


def _make_pool(padded_rows, suffix_experts=SUFFIX_EXPERTS):
    return DSparkExpertPool(
        torch.device(DEVICE),
        capacity=padded_rows,
        protected_positions=2,
        suffix_experts=suffix_experts,
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
        has_prefill=False,
    )


def _router(bias):
    return SimpleNamespace(num_logical_experts=EXPERTS, e_score_correction_bias=bias, routed_scaling_factor=SCALING)


@pytest.mark.skipif(
    get_ascend_device_type() not in SUPPORTED_DEVICES,
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("reqs,suffix_experts", [(1, 0), (1, 2), (1, TOP_K), (2, 2), (3, 2)])
def test_dspark_pool_routing_matches_reference_and_handles_strides(reqs, suffix_experts):
    torch.manual_seed(1)
    rows = 6 * reqs  # 12 rows reuse the cached workspace; 18 rows take the fresh-buffer path.
    pool = _make_pool(rows, suffix_experts)
    strided = torch.randn(rows, EXPERTS * 2, dtype=torch.float32, device=DEVICE)[:, ::2]
    strided[0, 0] = strided[0, 1] = 8.0  # exact tie: the lowest expert id must come first
    bias = torch.randn(EXPERTS, dtype=torch.float32, device=DEVICE) * 0.1
    bias[0] = bias[1] = 0.0
    pool.prepare(_make_batch(list(range(0, rows + 1, 6)), [5] * reqs, rows))
    unchanged = (strided.clone(), pool.roles.clone(), bias.clone())
    reference = _reference(strided, pool.roles[:rows], bias, suffix_experts, SCALING)
    outs = [pool.route(_router(bias), x) for x in (strided, strided.contiguous())]
    for layout, output in zip(("strided", "contiguous"), outs):
        _assert_routing_matches(f"{rows=} {suffix_experts=} {layout}", output, reference)
    # Strided logits are honoured directly: the kernels index via row/expert strides.
    assert outs[0][0].equal(outs[1][0]) and outs[0][1].equal(outs[1][1])
    for actual, original in zip((strided, pool.roles, bias), unchanged):
        assert actual.equal(original)


@pytest.mark.skipif(
    get_ascend_device_type() not in SUPPORTED_DEVICES,
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize("real,padded", [(6, 12), (12, 16)])
def test_dspark_padding_rows_neither_extend_the_pool_nor_get_routed(real, padded):
    torch.manual_seed(7)
    pool = _make_pool(padded)
    logits = torch.randn(padded, EXPERTS, dtype=torch.float32, device=DEVICE)
    # Real rows are kept off experts 198-206 so only padding rows favour them; if padding
    # could vote, expert 200 would pool-enter and reroute restricted row 2, which ranks it
    # just below its own two suffix votes (210, 211).
    logits[:real, 198:207] = -6.0
    logits[real:, 200:206] = 20.0
    logits[2, 200], logits[2, 210], logits[2, 211] = 9.0, 12.0, 11.5
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=DEVICE)
    boundaries = list(range(0, real + 1, 6))
    drafts = [5] * (real // 6)
    pool.prepare(_make_batch(boundaries, drafts, padded))
    expected_roles = [CONTRIBUTOR] * 2 + [RESTRICTED] * 4
    assert pool.roles[:padded].tolist() == expected_roles * (real // 6) + [0] * (padded - real)
    fused = pool.route(_router(bias), logits)
    reference = _reference(logits, pool.roles[:padded], bias, SUFFIX_EXPERTS, SCALING)
    _assert_routing_matches(f"{real=} {padded=}", fused, reference)
    # Outside verification the pool path is bypassed: merge_native keeps the native route.
    native = (
        torch.full((padded, TOP_K), 0.5, device=DEVICE),
        torch.full((padded, TOP_K), 999, dtype=torch.int32, device=DEVICE),
    )
    merged = pool.merge_native(fused, native)
    assert merged[0][real:].equal(native[0][real:]) and merged[1][real:].equal(native[1][real:])
    assert merged[0][:real].equal(fused[0][:real]) and merged[1][:real].equal(fused[1][:real])
    # Rerouting without the padding rows must be bit-identical for every real row.
    pool.prepare(_make_batch(boundaries, drafts, real))
    trimmed = pool.route(_router(bias), logits[:real])
    assert fused[0][:real].equal(trimmed[0]) and fused[1][:real].equal(trimmed[1])


@pytest.mark.skipif(get_ascend_device_type() not in SUPPORTED_DEVICES, reason="Requires Ascend A2/A3/A5")
@pytest.mark.parametrize("drafts,protected,suffix", [(1, 1, 0), (3, 1, TOP_K), (5, 6, 0)])
def test_public_wrapper_handles_verification_parameter_boundaries(drafts, protected, suffix):
    torch.manual_seed(19)
    rows = drafts + 1
    logits = torch.randn(rows, EXPERTS, dtype=torch.float32, device=DEVICE)
    roles = torch.full((rows,), RESTRICTED, dtype=torch.int32, device=DEVICE)
    roles[:protected] = CONTRIBUTOR
    bias = torch.randn(EXPERTS, dtype=torch.float32, device=DEVICE) * 0.1
    raw = torch.empty_like(logits)
    votes = torch.full((EXPERTS,), 123, dtype=torch.int32, device=DEVICE)
    originals = tuple(tensor.clone() for tensor in (logits, roles, bias))
    result = route_experts(logits, roles, bias, raw, votes, suffix, SCALING)
    _assert_routing_matches(
        f"{drafts=} {protected=} {suffix=}", result, _reference(logits, roles, bias, suffix, SCALING)
    )
    for tensor, original in zip((logits, roles, bias), originals):
        assert tensor.equal(original)
    assert torch.all(votes < 123)
    torch.testing.assert_close(raw, torch.sqrt(torch.log(torch.exp(logits) + 1.0)), rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(get_ascend_device_type() not in SUPPORTED_DEVICES, reason="Requires Ascend A2/A3/A5")
@pytest.mark.parametrize("experts", [TOP_K, EXPERTS])
def test_public_wrapper_zero_scores_preserve_ties_and_denominator_floor(experts):
    logits = torch.full((6, experts), -40.0, dtype=torch.float32, device=DEVICE)
    roles = torch.tensor([CONTRIBUTOR] * 2 + [RESTRICTED] * 4, dtype=torch.int32, device=DEVICE)
    bias = torch.zeros(experts, dtype=torch.float32, device=DEVICE)
    result = route_experts(
        logits,
        roles,
        bias,
        torch.empty_like(logits),
        torch.empty(experts, dtype=torch.int32, device=DEVICE),
        0,
        SCALING,
    )
    _assert_routing_matches(f"zero scores {experts=}", result, _reference(logits, roles, bias, 0, SCALING))
    assert torch.count_nonzero(result[0]).item() == 0
    assert result[1].equal(torch.arange(TOP_K, dtype=torch.int32, device=DEVICE).expand(6, -1))


@pytest.mark.skipif(get_ascend_device_type() not in SUPPORTED_DEVICES, reason="Requires Ascend A2/A3/A5")
@pytest.mark.parametrize("experts", [7, EXPERTS])
@pytest.mark.parametrize("logit", [100.0, 1000.0, 3e38])
def test_large_finite_logits_produce_finite_normalized_weights(experts, logit):
    logits = torch.full((2, experts), logit, dtype=torch.float32, device=DEVICE)
    roles = torch.tensor([CONTRIBUTOR, RESTRICTED], dtype=torch.int32, device=DEVICE)
    result = route_experts(
        logits,
        roles,
        torch.zeros(experts, device=DEVICE),
        torch.empty_like(logits),
        torch.empty(experts, dtype=torch.int32, device=DEVICE),
        0,
        SCALING,
    )
    expected = (
        torch.full((2, TOP_K), SCALING / TOP_K, dtype=torch.float32, device=DEVICE),
        torch.arange(TOP_K, dtype=torch.int32, device=DEVICE).expand(2, -1),
    )
    _assert_routing_matches(f"overflow {experts=} {logit=}", result, expected)
    assert torch.isfinite(result[0]).all()


@pytest.mark.skipif(get_ascend_device_type() not in SUPPORTED_DEVICES, reason="Requires Ascend A2/A3/A5")
@pytest.mark.parametrize("experts", [TOP_K, 7, EXPERTS])
@pytest.mark.parametrize("suffix", [0, 1])
def test_insufficient_pool_uses_safe_ids_and_zero_weight_for_missing_slots(experts, suffix):
    # Deliberately violate the production caller's P>=1 precondition to test safety.
    logits = torch.full((3, experts), -40.0, dtype=torch.float32, device=DEVICE)
    for row in range(3):
        logits[row, row + 1] = 4.0
    roles = torch.full((3,), RESTRICTED, dtype=torch.int32, device=DEVICE)
    bias = torch.zeros(experts, device=DEVICE)
    result = route_experts(
        logits,
        roles,
        bias,
        torch.empty_like(logits),
        torch.empty(experts, dtype=torch.int32, device=DEVICE),
        suffix,
        SCALING,
    )
    _assert_routing_matches(
        f"undersized {experts=} {suffix=}", result, _reference(logits, roles, bias, suffix, SCALING)
    )
    weights, ids = result
    assert torch.isfinite(weights).all() and torch.all((ids >= 0) & (ids < experts))
    if suffix:
        assert torch.all(ids[:, 3:] == 0) and torch.all(weights[:, 3:] == 0)
    else:
        assert torch.count_nonzero(weights).item() == 0 and torch.count_nonzero(ids).item() == 0


@pytest.mark.skipif(get_ascend_device_type() not in SUPPORTED_DEVICES, reason="Requires Ascend A2/A3/A5")
@pytest.mark.parametrize("experts", [7, EXPERTS])
@pytest.mark.parametrize("invalid", ["logit", "bias", "row", "infinite_logit", "infinite_bias"])
def test_nan_scores_do_not_create_invalid_ids_or_weights(experts, invalid):
    logits = torch.arange(experts, dtype=torch.float32, device=DEVICE).repeat(3, 1) / experts
    roles = torch.tensor([CONTRIBUTOR, CONTRIBUTOR, RESTRICTED], dtype=torch.int32, device=DEVICE)
    bias = torch.zeros(experts, device=DEVICE)
    if invalid == "bias":
        bias[3] = float("nan")
    elif invalid == "row":
        logits[2] = float("nan")
    elif invalid == "infinite_logit":
        logits[:, 3], logits[:, 4] = float("inf"), -float("inf")
    elif invalid == "infinite_bias":
        bias[3], bias[4] = float("inf"), -float("inf")
    else:
        logits[:, 3] = float("nan")
    result = route_experts(
        logits,
        roles,
        bias,
        torch.empty_like(logits),
        torch.empty(experts, dtype=torch.int32, device=DEVICE),
        2,
        SCALING,
    )
    _assert_routing_matches(f"NaN {experts=} {invalid=}", result, _reference(logits, roles, bias, 2, SCALING))
    assert torch.isfinite(result[0]).all() and torch.all((result[1] >= 0) & (result[1] < experts))
    if invalid == "row":
        assert torch.count_nonzero(result[0][2]).item() == 0


@pytest.mark.skipif(get_ascend_device_type() not in SUPPORTED_DEVICES, reason="Requires Ascend A2/A3/A5")
def test_mixed_large_and_small_logits_match_high_precision_weights():
    logits = torch.tensor(
        [[-40.0, -1.0, 0.0, 20.0, 21.0, 100.0, 1000.0], [100.0, 23.0, 21.0, -0.5, -2.0, -40.0, 1000.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    roles = torch.tensor([CONTRIBUTOR, RESTRICTED], dtype=torch.int32, device=DEVICE)
    result = route_experts(
        logits,
        roles,
        torch.zeros(7, device=DEVICE),
        torch.empty_like(logits),
        torch.empty(7, dtype=torch.int32, device=DEVICE),
        0,
        SCALING,
    )
    # Hand-selected IDs: the contributor excludes expert 0, so the restricted
    # row cannot choose expert 0 even though its own logit there is 100.
    ids = torch.tensor([[6, 5, 4, 3, 2, 1], [6, 1, 2, 3, 4, 5]], dtype=torch.int32, device=DEVICE)
    raw = torch.nn.functional.softplus(logits.cpu().double()).sqrt()
    top = raw.gather(1, ids.cpu().long())
    weights = (top / top.sum(dim=1, keepdim=True) * SCALING).float()
    _assert_routing_matches("mixed large/small high-precision", result, (weights, ids))


@pytest.mark.skipif(get_ascend_device_type() not in SUPPORTED_DEVICES, reason="Requires Ascend A2/A3/A5")
def test_real_zero_draft_decode_contributes_but_padding_does_not():
    pool = _make_pool(12, suffix_experts=0)
    pool.prepare(_make_batch([0, 1, 7], [0, 5], 12))
    assert pool.roles.tolist() == [CONTRIBUTOR] * 3 + [RESTRICTED] * 4 + [0] * 5
    assert not pool.has_native_rows
    logits = torch.full((12, EXPERTS), -40.0, dtype=torch.float32, device=DEVICE)
    logits[0, [90, 91, 92, 93, 94, 99]] = 5.0
    logits[1:7, 1:7] = 4.0
    logits[3:7, 99] = 6.0
    logits[7:, 200:206] = 20.0
    bias = torch.zeros(EXPERTS, device=DEVICE)
    result = pool.route(_router(bias), logits)
    _assert_routing_matches("real decode contribution", result, _reference(logits, pool.roles, bias, 0, SCALING))
    assert torch.all(result[1][3:7, 0] == 99)
    assert all(200 not in row for row in result[1][:7].tolist())
