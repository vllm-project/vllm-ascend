"""Synthetic NPU differential cases; run separately on A3 and A5."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.eplb.tensor_ops import count_allgather_tokens as count_allgather_tokens_triton


def reference(ids, mask, start, count):
    ids = ids.to(torch.int64) - start
    eligible = mask[:, None] & (ids >= 0) & (ids < count)
    out = torch.zeros(count, dtype=torch.int64)
    return out.scatter_add_(0, ids.clamp(0, count - 1).reshape(-1), eligible.to(torch.int64).reshape(-1))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "t,k,c",
    [(0, 8, 128), (4, 0, 3), (1, 1, 1), (65, 8, 17), (511, 1, 128), (512, 1, 128), (513, 1, 129), (10240, 8, 128)],
)
@pytest.mark.parametrize("pattern", ["mixed", "hot", "disabled", "all_valid", "remote"])
def test_exact(dtype, t, k, c, pattern):
    start = 7
    ids = (torch.arange(t * k).reshape(t, k) % (c + 4) + start - 2).to(dtype)
    if pattern in ("hot", "all_valid"):
        ids.fill_(start)
    elif pattern == "remote":
        ids.fill_(start + c)
    if t * k and dtype == torch.int64:
        ids[-1, -1] = 2**40
    mask = torch.arange(t) % 3 != 0
    if pattern == "disabled":
        mask.fill_(False)
    elif pattern == "all_valid":
        mask.fill_(True)
    expected = reference(ids, mask, start, c)
    # Exercise strided IDs and mask without changing their logical values.
    storage = torch.empty((t, k * 2), dtype=dtype, device="npu")
    view = storage[:, ::2]
    view.copy_(ids)
    mask_storage = torch.empty(t * 2, dtype=torch.bool, device="npu")
    mask_view = mask_storage[::2]
    mask_view.copy_(mask)
    actual = count_allgather_tokens_triton(view, mask_view, start, c)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_mutable_graph_replay():
    ids = torch.full((65, 8), 3, dtype=torch.int64, device="npu")
    mask = torch.ones(65, dtype=torch.bool, device="npu")
    for _ in range(3):
        count_allgather_tokens_triton(ids, mask, 3, 17)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = count_allgather_tokens_triton(ids, mask, 3, 17)
    for expert, enabled in [(4, True), (4, False), (19, True)]:
        ids.fill_(expert)
        mask.fill_(enabled)
        graph.replay()
        expected = reference(ids.cpu(), mask.cpu(), 3, 17)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
