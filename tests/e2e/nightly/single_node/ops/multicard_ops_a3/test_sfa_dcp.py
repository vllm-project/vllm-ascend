# SPDX-License-Identifier: Apache-2.0
"""DCP2/4/8 precision tests; launch with torchrun and pytest --noconftest."""

import gc
import math
import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")

from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

# Importing the module registers torch.ops.vllm.dcp_a2a_fused.
from vllm_ascend.ops.triton.dcp import dcp_a2a  # noqa: E402,F401
from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import can_exchange, exchange  # noqa: E402
from vllm_ascend.ops.triton.query_gather_prep import prep_query_head_major  # noqa: E402

RANKS = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(RANKS not in (2, 4, 8), reason="requires isolated DCP2/4/8")


@pytest.fixture(scope="module")
def context():
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    init_distributed_environment(
        world_size=RANKS, rank=rank, local_rank=local_rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(RANKS))],
        local_rank=local_rank,
        backend="hccl",
        group_name="generalized_dcp",
        use_device_communicator=False,
    )
    retained: list[tuple] = []
    yield rank, group, retained
    # Capture buffers must outlive all graphs sharing this communicator.
    for graph, checks, _owners in reversed(retained):
        graph.replay()
        for actual, expected, budget in checks:
            assert ((actual.cpu().double() - expected.double()).abs() <= budget).all()
    torch.npu.synchronize()
    dist.barrier(group=group.cpu_group)
    retained.clear()
    gc.collect()
    group.destroy()
    destroy_distributed_environment()


@pytest.mark.parametrize("local_heads", [3, 8, 32])
@pytest.mark.parametrize("column_stride", [1, 2])
def test_query_group_sizes_and_head_counts(context, local_heads, column_stride):
    rank, group, retained = context
    patterns = torch.tensor([0, -32768, 1, 127, 32640, -128, 32705, -63], dtype=torch.int16)
    steps = 3
    for tokens in (2, 6, 12):

        def bits(sender, width, step, tokens=tokens):
            rows = torch.arange(tokens * local_heads).view(tokens, local_heads, 1)
            columns = torch.arange(width).view(1, 1, width)
            values = ((rows * 257 + columns * 17 + sender * 4099 + step * 8191) % 65536 - 32768).to(torch.int16)
            values[..., : patterns.numel()] = patterns.roll(sender + step)
            assert not torch.equal(values[0, 0], values[0, 1])
            assert not torch.equal(values[0, 0], values[1, 0])
            return values

        n_owner = bits(rank, 512 * column_stride, 0).transpose(0, 1).contiguous().view(torch.bfloat16).npu()
        r_owner = bits(rank, 64 * column_stride, 0).view(torch.bfloat16).npu()
        qn, qr = n_owner.transpose(0, 1)[..., ::column_stride], r_owner[..., ::column_stride]

        def invoke(qn=qn, qr=qr, tokens=tokens):
            send = prep_query_head_major(qn, qr)
            gathered = torch.empty((RANKS * local_heads, tokens, 576), dtype=qn.dtype, device=qn.device)
            dist.all_gather_into_tensor(gathered, send, group=group.device_group)
            return gathered.transpose(0, 1).split((512, 64), dim=-1)

        for _ in range(5):
            invoke()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = invoke()
        for step in range(steps):
            n_owner.copy_(bits(rank, 512 * column_stride, step).transpose(0, 1).contiguous().view(torch.bfloat16))
            r_owner.copy_(bits(rank, 64 * column_stride, step).view(torch.bfloat16))
            graph.replay()
            for actual, width in zip(result, (512, 64)):
                expected = torch.cat(
                    [bits(s, width * column_stride, step)[..., ::column_stride] for s in range(RANKS)], dim=1
                )
                assert torch.equal(actual.cpu().view(torch.int16), expected)
        checks = [
            (
                actual.view(torch.int16),
                torch.cat(
                    [bits(s, width * column_stride, steps - 1)[..., ::column_stride] for s in range(RANKS)], dim=1
                ),
                0,
            )
            for actual, width in zip(result, (512, 64))
        ]
        retained.append((graph, checks, (n_owner, r_owner)))


@pytest.mark.parametrize(
    "local_heads,tokens,from_stats",
    [(3, 1, False), (8, 2, True), (32, 13, True), (8, 48, False), (32, 191, True), (3, 192, False)],
)
def test_registered_exchange_group_sizes_against_fp64(context, local_heads, tokens, from_stats):
    rank, group, retained = context
    heads = RANKS * local_heads

    def inputs(step):
        generator = torch.Generator().manual_seed(16350 + step)
        outputs = torch.randn(RANKS, tokens, heads, 512, generator=generator).to(torch.bfloat16)
        lses = torch.randn(RANKS, tokens, heads, 1, generator=generator) * 4
        lses[:, :, ::local_heads] = -torch.inf
        outputs[:, :, ::local_heads] = torch.nan
        lses[0, :, 1::local_heads] = torch.nan
        outputs[0, :, 1::local_heads] = torch.nan
        return outputs, lses

    outputs, lses = inputs(0)
    owner = outputs[rank].transpose(0, 1).contiguous().npu()
    output = owner.transpose(0, 1)
    max_offset = math.log(2) if from_stats else 0.0
    lse_owner = (lses[rank] - max_offset).transpose(0, 1).contiguous().npu()
    lse = lse_owner.transpose(0, 1)
    sums = torch.full_like(lse, 2.0) if from_stats else None

    optimized = can_exchange(output, lse, RANKS)

    def invoke():
        fp32 = exchange(output, lse, group.device_group, softmax_sum=sums) if optimized else None
        return fp32, torch.ops.vllm.dcp_a2a_fused(output, lse, RANKS, 1, group.unique_name, softmax_sum=sums)

    for _ in range(3):
        invoke()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fp32, bf16 = invoke()
    for step in range(3):
        outputs, lses = inputs(step)
        owner.copy_(outputs[rank].transpose(0, 1).contiguous())
        lse_owner.copy_((lses[rank] - max_offset).transpose(0, 1).contiguous())
        graph.replay()
        values = outputs[:, :, rank * local_heads : (rank + 1) * local_heads].double()
        stats = lses[:, :, rank * local_heads : (rank + 1) * local_heads].double()
        valid = torch.isfinite(stats)
        weights = torch.nan_to_num(torch.softmax(stats.masked_fill(~valid, -torch.inf), dim=0), nan=0.0)
        terms = values.masked_fill(~valid, 0.0) * weights
        reference = terms.sum(0)
        budget = 1e-6 + 8 * torch.finfo(torch.float32).eps * terms.abs().sum(0)
        rounded = reference.to(torch.bfloat16)
        upper = torch.nextafter(rounded, torch.full_like(rounded, torch.inf)).double()
        lower = torch.nextafter(rounded, torch.full_like(rounded, -torch.inf)).double()
        half_ulp = torch.maximum(upper - rounded.double(), rounded.double() - lower) / 2
        if optimized:
            assert ((fp32.cpu().double() - reference).abs() <= budget).all()
        assert ((bf16.cpu().double() - reference).abs() <= half_ulp + budget).all()
    checks = [(bf16, reference, half_ulp + budget)]
    if optimized:
        checks.append((fp32, reference, budget))
    retained.append((graph, checks, (owner, lse_owner, sums)))


@pytest.mark.parametrize("layout", ["ordinary", "odd_singleton", "odd_offset"])
def test_exchange_t1_singleton_stride_graph(context, layout):
    rank, group, retained = context
    heads = RANKS * 8
    if layout == "odd_singleton":
        owner = torch.empty((heads, 512, 1), dtype=torch.bfloat16, device="npu")
        output = owner.permute(2, 0, 1)
        assert output.is_contiguous() and output.stride(0) == 1
    elif layout == "odd_offset":
        owner = torch.empty(heads * 512 + 1, dtype=torch.bfloat16, device="npu")
        output = owner[1:].view(1, heads, 512)
    else:
        owner = output = torch.empty((1, heads, 512), dtype=torch.bfloat16, device="npu")
    lse = torch.zeros((1, heads, 1), device="npu")
    base = (torch.arange(heads * 512).reshape(1, heads, 512) % 31 - 15).float() / 32
    output.copy_(base + rank / 16)

    def invoke():
        return torch.ops.vllm.dcp_a2a_fused(output, lse, RANKS, 1, group.unique_name)

    for _ in range(3):
        invoke()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = invoke()
    for step in range(3):
        output.copy_(base + rank / 16 + step / 64)
        lse.fill_(step / 4)
        graph.replay()
        expected = (base[:, rank * 8 : (rank + 1) * 8] + (RANKS - 1) / 32 + step / 64).bfloat16()
        assert torch.equal(result.cpu().view(torch.int16), expected.view(torch.int16))
    retained.append((graph, [(result, expected, 0)], (owner, lse)))


@pytest.mark.parametrize(
    "tokens,local_heads,head_dim",
    [(512, 32, 512), (513, 32, 512), (1025, 16, 512), (256, 64, 512), (257, 64, 512), (1, 64, 1024), (2, 64, 1024)],
)
def test_exchange_tiled_resource_graph(context, tokens, local_heads, head_dim):
    rank, group, retained = context
    heads = RANKS * local_heads
    owner = torch.empty((heads, tokens, head_dim), dtype=torch.bfloat16, device="npu")
    output = owner.transpose(0, 1)
    lse = torch.zeros((tokens, heads, 1), device="npu")
    assert can_exchange(output, lse, RANKS)

    def invoke():
        return torch.ops.vllm.dcp_a2a_fused(output, lse, RANKS, 1, group.unique_name)

    output.fill_(rank / 16)
    for _ in range(3):
        invoke()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = invoke()
    for step in range(3):
        output.fill_(rank / 16 + step / 64)
        lse.fill_(step / 4)
        graph.replay()
        expected = torch.full((tokens, local_heads, head_dim), (RANKS - 1) / 32 + step / 64, dtype=torch.bfloat16)
        assert torch.equal(result.cpu().view(torch.int16), expected.view(torch.int16))
    retained.append((graph, [(result, expected, 0)], (owner, lse)))
