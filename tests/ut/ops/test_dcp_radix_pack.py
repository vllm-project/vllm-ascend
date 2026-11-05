# SPDX-License-Identifier: Apache-2.0
"""Validate exact digit ordering, FP32 key limits and original pair transport."""

import numpy as np
import pytest
import torch

from tests.ut.ops.test_dcp_kernels_cpu import array, load_kernels


@pytest.mark.parametrize("score_bits", [16, 32])
@pytest.mark.parametrize("id_bits", [14, 20, 23, 25])
@pytest.mark.parametrize("log_k", [3, 11])
@pytest.mark.parametrize("pattern", ["normal", "bits"])
def test_exact_radix_order(score_bits, id_bits, log_k, pattern):
    _, scope = load_kernels("vllm_ascend/ops/triton/sharded_indexer.py", ("_radix_sort_candidate_pairs",))

    def native_sort(value, descending):
        assert np.all((value >= 0) & (value < (1 << 24)))
        assert np.unique(value).size == value.size
        return array(np.sort(value))

    scope["_a5_sort"] = native_sort
    rng = np.random.default_rng(5311)
    k = 1 << log_k
    ids = rng.integers(0, min(1 << id_bits, 1 << 24), k, dtype=np.int32).astype(np.float32)
    ids[-1] = min((1 << id_bits) - 1, 1 << 24)
    values = rng.standard_normal(k).astype(np.float32)
    if pattern == "bits":
        values = rng.integers(0, 1 << 32, k, dtype=np.uint32).view(np.float32)
        values = np.where(np.isfinite(values), values, -np.inf).astype(np.float32)
    values[:8] = [0.0, -0.0, 1.0, -1.0, -np.inf, -np.inf, np.finfo(np.float32).tiny, np.finfo(np.float32).max]
    if score_bits == 16:
        values = (values.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    ids[:2] = 7
    actual_values, actual_ids = scope["_radix_sort_candidate_pairs"](
        array(values), array(ids), array(np.arange(k, dtype=np.int32)), log_k, id_bits, score_bits
    )
    order = np.lexsort((ids, -values))
    np.testing.assert_array_equal(actual_ids, ids[order])
    np.testing.assert_array_equal(actual_values.view(np.int32), values[order].view(np.int32))


def _expected_pack(ids, scores, world, interleave):
    valid = (ids >= 0) & torch.isfinite(scores)
    values = torch.where(valid, scores.float(), -torch.inf)
    keys = torch.where(valid, ids, torch.iinfo(torch.int32).max)
    by_id = keys.argsort(stable=True)
    order = by_id.gather(-1, values.gather(-1, by_id).argsort(descending=True, stable=True))
    mapped = ids + (ids // interleave * (world - 1) + world - 1) * interleave
    return torch.stack(
        (torch.where(valid, mapped, -1).gather(-1, order), values.gather(-1, order).contiguous().view(torch.int32)), -1
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "bound,world,rows,strided",
    [
        (1 << 14, 8, 3, False),
        (1 << 20, 16, 48, True),
        ((1 << 20) + 1, 3, 17, False),
        (1 << 23, 17, 65, True),
        ((1 << 23) + 1, 16, 3, True),
        ((1 << 24) + 1, 1, 3, False),
        ((1 << 24) + 1, 16, 3, True),
    ],
)
def test_npu_radix_pack_exact_and_graph(dtype, bound, world, rows, strided):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU hardware required")
    torch.npu.set_device(0)
    from vllm_ascend.ops.triton.sharded_indexer import _pack_sorted_candidates
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

    k, interleave = 2048, 768
    generator = torch.Generator().manual_seed(5311)
    ids = torch.randint(0, bound, (rows, k * 2), dtype=torch.int32, generator=generator)
    scores = torch.randint(-2, 3, ids.shape, generator=generator).to(dtype)
    # All score ties below have distinct IDs, so their original score bits are
    # observable even for signed zero. Include mapped payloads above 2**24.
    ids[:, :20] = torch.arange(20, dtype=torch.int32)
    ids[:, 20] = bound - 1
    scores[:, :20:2] = torch.tensor(
        [torch.nan, torch.inf, -torch.inf, 0.0, -0.0, 1.0, -1.0, 2.0, -2.0, torch.finfo(dtype).tiny]
    ).to(dtype)
    if dtype == torch.float32:
        scores[:, 22:26:2] = torch.tensor([1.0, torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))])
    ids[0] = -1
    ids, scores = ids[:, ::2], scores[:, ::2]
    if not strided:
        ids, scores = ids.contiguous(), scores.contiguous()
    # Transfer backing storage before slicing to preserve actual NPU strides.
    if strided:
        device_ids = torch.empty((rows, 2 * k), dtype=torch.int32, device="npu")[:, ::2]
        device_scores = torch.empty((rows, 2 * k), dtype=dtype, device="npu")[:, ::2]
        device_ids.copy_(ids)
        device_scores.copy_(scores)
    else:
        device_ids, device_scores = ids.npu(), scores.npu()
    output = torch.empty((rows, k, 2), dtype=torch.int32, device="npu")
    init_device_properties_triton()

    def run():
        _pack_sorted_candidates[(min(rows, get_vectorcore_num()),)](
            device_ids,
            device_scores,
            output,
            rows,
            device_ids.stride(0),
            device_ids.stride(-1),
            device_scores.stride(0),
            device_scores.stride(-1),
            world,
            world - 1,
            interleave,
            11,
            RADIX=True,
            multibuffer=False,
        )

    run()
    torch.testing.assert_close(output.cpu(), _expected_pack(ids, scores, world, interleave), rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            run()
        # Change key range across the on-device fast-path bound and then make
        # every row invalid, with the captured pointers and strides unchanged.
        for changed in ("large", "invalid"):
            if changed == "large":
                ids = ids.flip(-1).clone()
                ids[1, 0] = 1 << 24
                scores = -scores.flip(-1)
            else:
                ids.fill_(-1)
            device_ids.copy_(ids)
            device_scores.copy_(scores)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), _expected_pack(ids, scores, world, interleave), rtol=0, atol=0)
    finally:
        graph.reset()
