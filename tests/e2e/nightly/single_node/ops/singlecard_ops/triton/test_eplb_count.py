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
    [
        (0, 8, 128),
        (4, 0, 3),
        (1, 1, 1),
        (1, 8, 128),
        (32, 2, 8),
        (64, 4, 32),
        (65, 8, 17),
        (128, 3, 64),
        (256, 8, 64),
        (511, 1, 128),
        (512, 1, 128),
        (513, 1, 129),
        (512, 8, 128),
        (2048, 8, 128),
        (10240, 1, 128),
        (10240, 4, 128),
        (10240, 8, 32),
        (10240, 8, 64),
        (10240, 8, 128),
        (10240, 8, 256),
        (10240, 16, 128),
        (32768, 8, 128),
        (2, 1, 1),
        (16, 2, 16),
        (127, 8, 127),
        (129, 8, 129),
        (1024, 2, 128),
        (1024, 8, 128),
        (2559, 8, 128),
        (2560, 8, 128),
        (2561, 8, 128),
        (4095, 8, 128),
        (4096, 8, 128),
        (4097, 8, 128),
        (8192, 8, 128),
        (16384, 8, 128),
        (65536, 8, 128),
        (131072, 8, 128),
        (10240, 2, 128),
        (10240, 32, 128),
        (10240, 8, 16),
        (10240, 8, 512),
        (960, 8, 512),
        (1024, 8, 512),
        (1088, 8, 512),
    ],
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
    # IDs are contiguous by contract; the token mask may still be strided.
    device_ids = ids.npu()
    mask_storage = torch.empty(t * 2, dtype=torch.bool, device="npu")
    mask_view = mask_storage[::2]
    mask_view.copy_(mask)
    actual = count_allgather_tokens_triton(device_ids, mask_view, start, c)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_reject_noncontiguous_ids():
    ids = torch.zeros((4, 16), dtype=torch.int64, device="npu")[:, ::2]
    mask = torch.ones(4, dtype=torch.bool, device="npu")
    with pytest.raises(ValueError, match="contiguous physical IDs"):
        count_allgather_tokens_triton(ids, mask, 0, 128)


@pytest.mark.parametrize("tokens,experts,calls", [(65, 17, 1), (10240, 128, 20), (1088, 512, 20), (10240, 512, 20)])
def test_mutable_graph_replay(tokens, experts, calls):
    ids = torch.full((tokens, 8), 3, dtype=torch.int64, device="npu")
    mask = torch.ones(tokens, dtype=torch.bool, device="npu")
    for _ in range(3):
        count_allgather_tokens_triton(ids, mask, 3, experts)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls):
            out = count_allgather_tokens_triton(ids, mask, 3, experts)
    for expert, enabled in [(4, True), (4, False), (19, True)] * 3:
        ids.fill_(expert)
        mask.fill_(enabled)
        graph.replay()
        expected = reference(ids.cpu(), mask.cpu(), 3, experts)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
