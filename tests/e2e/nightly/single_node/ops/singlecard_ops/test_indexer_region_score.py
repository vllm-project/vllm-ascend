# SPDX-License-Identifier: Apache-2.0
"""Precision test for the Step5 indexer region score operator.

Calls ``torch.ops._C_ascend.npu_indexer_region_score.ids_out`` directly
(the exact call ``vllm_ascend/models/step5/ops/indexer_region_score.py``
makes for the Step5 DSA indexer: fused region scoring + causal top-k +
ascending logical-id sort, ids-only path with a preallocated
``selected_region_ids`` buffer, ``cache_mode=1, mask_mode=3, sort_mode=1``).

Reference semantics (csrc/attention/indexer_region_score/README.md): for
query token ``j`` of request ``n``, group ``g``, the visible region count is
``clamp((actual_kv_lengths[n] - q_len + j) // region_size, 0, max_region_len)``
and each visible region ``r`` scores
``sum_h(weights[t, g, h] * head_scale * relu(dot(query[t, g, h], key[n, r])))``
where ``key[n, r] = mean_cache[block_table[n, r // S], r % S, 0, :]``. IDs
are selected by descending score (ties -> smaller logical ID) and emitted in
ascending order with ``-1`` tail padding.

All inputs use binary fractions that are exact in FP16 / BF16 / FP8 E4M3 /
FP32, so the FP32 torch oracle is bit-exact and the INT32 ids are compared
exactly (integer/index output -> exact comparison). Query and key dtypes
follow each case (FP16, BF16 or FP8); weights stay FP32 and ``head_scale``
is a non-default 0.5. Coverage: causal visibility across a mixed batch
(including an empty request and a request whose packed query outruns its
history), grouped G=4/H=12 scoring, the model-scale ``top_k=512``
saturation, zero visible regions (all ``-1``), tie-breaking towards smaller
ids, a non-contiguous preallocated output view (SENTINEL-guarded), and the
schema's write-alias contract.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op

HEAD_DIM = 256
REGION_SIZE = 8
HEAD_SCALE = 0.5
SENTINEL = -313
GUARD_ELEMENTS = 8


def make_inputs(query_lengths, regions, page_size, lengths, *, groups, heads, dtype, ties=False):
    """Binary-fraction inputs (exact through every dtype here) and a poisoned
    2-layer cache backing whose layer-1 strided view is the operator input."""
    generator = torch.Generator().manual_seed(431)
    tokens = sum(query_lengths)
    columns = max(1, (regions + page_size - 1) // page_size)
    pages = len(query_lengths) * columns + 3
    # Small binary fractions are exactly representable even in E4M3FN. Every
    # product and sum also fits FP32 exactly, so accumulation order is irrelevant.
    query = (torch.randint(-2, 3, (tokens, groups, heads, HEAD_DIM), generator=generator).float() / 4).to(dtype)
    weights = torch.randint(-4, 5, (tokens, groups, heads), generator=generator).float() / 8
    if ties:
        weights.zero_()
    keys = (torch.randint(-2, 3, (pages, page_size, 1, HEAD_DIM), generator=generator).float() / 4).to(dtype)
    # Poison the unused layer so a wrong layer offset or page stride shows up.
    backing = torch.full((pages, 2, page_size, 1, HEAD_DIM), 31.0).to(dtype)
    backing[:, 1] = keys
    # Permuted pages expose affine-only cache addressing.
    table = torch.randperm(pages, generator=generator)[: len(query_lengths) * columns]
    table = table.reshape(len(query_lengths), columns).to(torch.int32)
    cu = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
    return query, weights, backing, torch.tensor(lengths, dtype=torch.int32), table, cu


def reference_ids(query, weights, backing, lengths, table, cu, *, regions, top_k):
    """FP32 oracle; dyadic inputs keep every score exact, including ties."""
    page_size = backing.shape[2]
    groups = query.shape[1]
    ids = torch.full((groups, query.shape[0], top_k), -1, dtype=torch.int32)
    logical = torch.arange(regions)
    for request in range(table.shape[0]):
        start, end = int(cu[request]), int(cu[request + 1])
        if start == end:
            continue
        # Reconstruct the request's logical sequence before computing attention.
        physical = table[request, logical // page_size].long()
        keys = backing[physical, 1, logical % page_size, 0].float()
        # [seg, G, H, R] -> weighted per-head ReLU scores summed over H.
        dots = torch.matmul(query[start:end].float(), keys.T).relu()
        values = (dots * weights[start:end, :, :, None] * HEAD_SCALE).sum(dim=2)
        history = int(lengths[request]) - (end - start)
        for token in range(start, end):
            pos = history + (token - start)
            visible = max(0, min(regions, pos // REGION_SIZE))
            if visible == 0:
                continue
            # Stable sorting starts in logical order: smaller logical IDs win
            # equal scores. The public result then sorts the selected IDs.
            selected = torch.argsort(values[token - start, :, :visible], descending=True, stable=True)[:, :top_k]
            ids[:, token, : selected.shape[1]] = selected.sort().values.to(torch.int32)
    return ids


def check_case(query_lengths, regions, page_size, top_k, lengths, *, groups, heads, dtype, ties=False, ids=None):
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("IndexerRegionScore requires Ascend 950 (A5)")
    enable_custom_op()
    query, weights, backing, lengths, table, cu = make_inputs(
        query_lengths, regions, page_size, lengths, groups=groups, heads=heads, dtype=dtype, ties=ties
    )
    expected = reference_ids(query, weights, backing, lengths, table, cu, regions=regions, top_k=top_k)

    tokens = sum(query_lengths)
    # The ids_out overload writes into the preallocated [G, T, top_k]
    # selected_region_ids buffer; Step5 always runs the ids-only path with
    # cache_mode=1, mask_mode=3, sort_mode=1.
    if ids is None:
        ids = torch.empty((groups, tokens, top_k), dtype=torch.int32, device="npu")
    torch.ops._C_ascend.npu_indexer_region_score.ids_out(
        query.npu(),
        weights.npu(),
        backing.npu()[:, 1],
        lengths.npu(),
        table.npu(),
        cu.npu(),
        max_region_len=regions,
        top_k=top_k,
        region_size=REGION_SIZE,
        head_scale=HEAD_SCALE,
        cache_mode=1,
        mask_mode=3,
        sort_mode=1,
        selected_region_ids=ids,
    )

    torch.npu.synchronize()
    assert ids.shape == (groups, tokens, top_k)
    assert ids.dtype == torch.int32
    torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0, msg="region ids mismatch")
    torch.npu.empty_cache()
    return ids


def test_causal_visibility_fp16():
    """Mixed batch (5 + 0 + 3 tokens): an empty request, a request whose
    packed query outruns its history (early tokens see nothing), and a long
    history whose causal visibility grows per token."""
    check_case((5, 0, 3), 37, 7, 17, (4, 0, 129), groups=1, heads=4, dtype=torch.float16)


def test_causal_visibility_bf16():
    """Same causal scenario as the FP16 case, in BF16."""
    check_case((5, 0, 3), 37, 7, 17, (4, 0, 129), groups=1, heads=4, dtype=torch.bfloat16)


def test_causal_visibility_fp8():
    """Same causal scenario as the FP16 case, in FP8 E4M3."""
    check_case((5, 0, 3), 37, 7, 17, (4, 0, 129), groups=1, heads=4, dtype=torch.float8_e4m3fn)


def test_grouped_requests_fp16():
    """G=4 groups x H=12 heads (README group/head constraints) over a single
    request with a near-full history."""
    check_case((2,), 19, 5, 7, (153,), groups=4, heads=12, dtype=torch.float16)


def test_grouped_requests_bf16():
    """Same grouped scenario as the FP16 case, in BF16."""
    check_case((2,), 19, 5, 7, (153,), groups=4, heads=12, dtype=torch.bfloat16)


def test_grouped_requests_fp8():
    """Same grouped scenario as the FP16 case, in FP8 E4M3."""
    check_case((2,), 19, 5, 7, (153,), groups=4, heads=12, dtype=torch.float8_e4m3fn)


def test_model_scale_topk_512():
    """Model-scale selection: 521 visible regions saturating top_k=512 with
    9 unseen candidates left as -1 padding."""
    check_case((2,), 521, 16, 512, (4170,), groups=4, heads=12, dtype=torch.bfloat16)


def test_zero_regions_all_padding():
    """max_region_len=0 leaves every request with zero visible regions, so
    every id row must be entirely -1."""
    check_case((2, 0, 1), 0, 7, 3, (100, 0, 100), groups=4, heads=12, dtype=torch.float16)


def test_tie_breaks_prefer_smaller_id():
    """Zeroed weights make every score 0: selection must break ties towards
    smaller logical ids (emitted ascending)."""
    check_case((3,), 19, 5, 7, (153,), groups=1, heads=4, dtype=torch.float8_e4m3fn, ties=True)


def test_output_written_through_strided_view():
    """selected_region_ids may be a non-contiguous view: the operator must
    write through its strides and never touch neighbouring elements
    (SENTINEL guards before, after, and in the interleaved gaps)."""
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("IndexerRegionScore requires Ascend 950 (A5)")
    groups, tokens, top_k = 4, 3, 7
    storage = torch.full((groups * tokens * top_k * 2 + 2 * GUARD_ELEMENTS,), SENTINEL, dtype=torch.int32, device="npu")
    ids = storage[GUARD_ELEMENTS:-GUARD_ELEMENTS:2].view(groups, tokens, top_k)
    assert not ids.is_contiguous()
    check_case((2, 0, 1), 19, 5, top_k, (153, 0, 9), groups=groups, heads=12, dtype=torch.bfloat16, ids=ids)
    actual = storage.cpu()
    assert torch.all(actual[:GUARD_ELEMENTS] == SENTINEL)
    assert torch.all(actual[-GUARD_ELEMENTS:] == SENTINEL)
    assert torch.all(actual[GUARD_ELEMENTS:-GUARD_ELEMENTS][1::2] == SENTINEL)


def test_schema_declares_write_alias_output():
    """The ids_out overload returns nothing: selected_region_ids is the
    kwarg-only write-alias output argument of the schema."""
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("IndexerRegionScore requires Ascend 950 (A5)")
    enable_custom_op()
    schema = torch.ops._C_ascend.npu_indexer_region_score.ids_out._schema
    assert not schema.returns
    output = schema.arguments[-1]
    assert output.name == "selected_region_ids"
    assert output.kwarg_only and not output.has_default_value()
    assert output.alias_info.is_write
