# SPDX-License-Identifier: Apache-2.0
"""Check pair loads at short/tail tiles and under changed-input graph replay."""

import pytest
import torch

pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("NPU required", allow_module_level=True)

from vllm_ascend.ops.triton.sharded_indexer import _merge_candidates
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


def expected_indices(ids, scores, owner, world):
    rows = ids.shape[1]
    result = torch.full((rows, 2048), -1, dtype=torch.int32)
    for row in range(rows):
        row_ids = ids[:, row].reshape(-1)
        row_scores = scores[:, row].reshape(-1)
        valid = (row_ids >= 0) & torch.isfinite(row_scores)
        keys = row_ids[valid]
        values = row_scores[valid]
        by_id = keys.argsort(stable=True)
        order = values[by_id].argsort(descending=True, stable=True)[:2048]
        selected = keys[by_id][order]
        if owner >= 0:
            selected = selected[(selected // 768) % world == owner]
            selected = selected // (world * 768) * 768 + selected % 768
        result[row, : selected.numel()] = selected
    return result


@pytest.mark.parametrize(
    "world,rows,local_k", [(3, 3, 7), (3, 2, 2051), (16, 48, 2048), (8, 128, 2048), (17, 49, 2048)]
)
@pytest.mark.parametrize("return_local", [False, True])
def test_merge_pair_tail_and_graph(world, rows, local_k, return_local):
    generator = torch.Generator().manual_seed(53)
    ids = torch.randint(0, 1 << 24, (world, rows, local_k), dtype=torch.int32, generator=generator)
    scores = torch.randint(-3, 4, ids.shape, generator=generator).float()
    ids[:, 0] = -1
    scores[..., :3] = torch.tensor([torch.nan, torch.inf, -torch.inf])
    owner = world - 1 if return_local else -1
    valid = (ids >= 0) & torch.isfinite(scores)
    safe_scores = torch.where(valid, scores, -torch.inf)
    safe_ids = torch.where(valid, ids, torch.iinfo(torch.int32).max)
    by_id = safe_ids.argsort(stable=True)
    ordered_scores = safe_scores.gather(-1, by_id)
    by_score = ordered_scores.argsort(descending=True, stable=True)
    order = by_id.gather(-1, by_score)
    packed_cpu = torch.stack((ids.gather(-1, order), scores.gather(-1, order).contiguous().view(torch.int32)), -1)
    packed = packed_cpu.npu()
    output = torch.full((rows, 2048), -123, device="npu", dtype=torch.int32)
    init_device_properties_triton()
    grid = min(rows, get_vectorcore_num())

    def run():
        _merge_candidates[(grid,)](packed, output, rows, local_k, 2048, world, owner, 768, 11, True, multibuffer=False)

    for _ in range(3):
        run()
    torch.testing.assert_close(output.cpu(), expected_indices(ids, scores, owner, world), rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            run()
        packed.copy_(packed_cpu.flip(0))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            output.cpu(), expected_indices(ids.flip(0), scores.flip(0), owner, world), rtol=0, atol=0
        )
    finally:
        graph.reset()
