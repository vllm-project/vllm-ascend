# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --nproc-per-node=2,4,8 -m pytest --noconftest."""

import gc
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

from vllm_ascend.ops.triton.dcp import dcp_a2a  # noqa: E402
from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import can_exchange, exchange  # noqa: E402
from vllm_ascend.ops.triton.dcp.sfa_dcp_query import prepare_query_head_major, unpack_query  # noqa: E402

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
    for tokens in (2, 12):

        def bits(sender, width, step, tokens=tokens):
            values = torch.arange(tokens * local_heads * width).reshape(tokens, local_heads, width)
            return ((values * 17 + sender * 4099 + step * 8191) % 65536 - 32768).to(torch.int16)

        n_owner = bits(rank, 512 * column_stride, 0).transpose(0, 1).contiguous().view(torch.bfloat16).npu()
        r_owner = bits(rank, 64 * column_stride, 0).view(torch.bfloat16).npu()
        qn, qr = n_owner.transpose(0, 1)[..., ::column_stride], r_owner[..., ::column_stride]

        def invoke(qn=qn, qr=qr, tokens=tokens):
            send = prepare_query_head_major(qn, qr)
            gathered = torch.empty((RANKS * local_heads, tokens, 576), dtype=qn.dtype, device=qn.device)
            dist.all_gather_into_tensor(gathered, send, group=group.device_group)
            return unpack_query(gathered)

        for _ in range(3):
            invoke()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = invoke()
        for step in range(3):
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
                torch.cat([bits(s, width * column_stride, 2)[..., ::column_stride] for s in range(RANKS)], dim=1),
                0,
            )
            for actual, width in zip(result, (512, 64))
        ]
        retained.append((graph, checks, (n_owner, r_owner)))


@pytest.mark.parametrize("local_heads", [3, 8, 32])
@pytest.mark.parametrize("tokens", [1, 2, 13, 48, 191, 192])
def test_registered_exchange_group_sizes_against_fp64(context, local_heads, tokens, monkeypatch):
    rank, group, retained = context
    heads = RANKS * local_heads
    calls = []

    def observed(*args):
        calls.append(True)
        return exchange(*args)

    monkeypatch.setattr(dcp_a2a, "exchange", observed)

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
    lse_owner = lses[rank].transpose(0, 1).contiguous().npu()
    lse = lse_owner.transpose(0, 1)

    optimized = can_exchange(output, lse, RANKS)

    def invoke():
        fp32 = exchange(output, lse, group.device_group) if optimized else None
        return fp32, torch.ops.vllm.dcp_a2a_fused(output, lse, RANKS, 1, group.unique_name)

    for _ in range(3):
        invoke()
    graph = torch.npu.NPUGraph()
    before = len(calls)
    with torch.npu.graph(graph):
        fp32, bf16 = invoke()
    assert (len(calls) > before) == optimized, "registered capture ignored tensor eligibility"
    for step in range(3):
        outputs, lses = inputs(step)
        owner.copy_(outputs[rank].transpose(0, 1).contiguous())
        lse_owner.copy_(lses[rank].transpose(0, 1).contiguous())
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
    retained.append((graph, checks, (owner, lse_owner)))
