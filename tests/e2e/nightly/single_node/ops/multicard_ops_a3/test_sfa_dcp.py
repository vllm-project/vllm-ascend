# SPDX-License-Identifier: Apache-2.0
"""DCP2/4/8 precision tests; launch with torchrun and pytest --noconftest."""

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
from vllm_ascend.ops.triton.dcp.dcp_a2a import (  # noqa: E402
    dcp_a2a_fused_combine,
    fused_dcp_lse_combine,
    pack_dcp_output_lse,
)
from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import can_exchange, exchange  # noqa: E402
from vllm_ascend.ops.triton.dcp.sfa_dcp_query import prepare_query_head_major, unpack_query  # noqa: E402

RANKS = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = pytest.mark.skipif(RANKS not in (2, 4, 8), reason="requires isolated DCP2/4/8")


@pytest.mark.skipif(RANKS != 8, reason="DCP8-specific precision regression")
def test_exchange_matches_upstream_and_reference_with_changed_input_graphs():
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    retained = []

    def inputs(tokens, step):
        generator = torch.Generator().manual_seed(2026 + step)
        outputs = torch.randn(8, tokens, 64, 512, generator=generator).to(torch.bfloat16)
        lses = torch.randn(8, tokens, 64, 1, generator=generator)
        # All-invalid and mixed-invalid heads cover NaN*0 contamination.
        lses[:, :, 0] = -torch.inf
        outputs[:, :, 0] = torch.nan
        for sender, value in enumerate((torch.nan, torch.inf, -torch.inf)):
            lses[sender, :, 1] = value
            outputs[sender, :, 1] = torch.nan
        return outputs, lses

    for tokens in (1, 6, 12):
        outputs, lses = inputs(tokens, 0)
        # Head-major storage exercises a strided token-major view.
        owner = outputs[rank].transpose(0, 1).contiguous().npu()
        output = owner.transpose(0, 1)
        lse = lses[rank].npu()

        def invoke(output=output, lse=lse):
            return dcp_a2a_fused_combine(output, lse, 8, 1, dist.group.WORLD)

        for _ in range(5):
            invoke()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = invoke()
        for step in range(8):
            outputs, lses = inputs(tokens, step)
            owner.copy_(outputs[rank].transpose(0, 1).contiguous())
            lse.copy_(lses[rank])
            # These unchanged upstream helpers bypass the new dispatch guard.
            send = pack_dcp_output_lse(output, lse, 8, 1)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send)
            upstream = fused_dcp_lse_combine(recv, 512, 1)
            graph.replay()
            torch.npu.synchronize()
            finite = torch.isfinite(lses)
            weights = torch.nan_to_num(torch.softmax(lses.masked_fill(~finite, -torch.inf), dim=0), nan=0.0)
            expected = (outputs.float().masked_fill(~finite, 0.0) * weights).sum(0)
            expected = expected[:, rank * 8 : (rank + 1) * 8].to(torch.bfloat16)
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(result, upstream, rtol=1e-3, atol=1e-3)
        retained.append((graph, result, owner, lse))
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(RANKS != 8, reason="DCP8-specific precision regression")
def test_large_batch_registered_graph_matches_exact_uniform_merge(monkeypatch):
    """Exercise the actual registered op with an exactly representable oracle.

    The existing random strict-BF16 test above is retained unchanged. This
    regression targets new large-batch routing, group lookup and graph reuse.
    """
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        init_distributed_environment,
        init_model_parallel_group,
    )

    from vllm_ascend.ops.triton.dcp import dcp_a2a as dispatch
    from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import can_exchange

    calls: dict[int, int] = {}
    original_exchange = dispatch.exchange

    def observed_exchange(output, lse, group):
        tokens = output.shape[0]
        calls[tokens] = calls.get(tokens, 0) + 1
        return original_exchange(output, lse, group)

    monkeypatch.setattr(dispatch, "exchange", observed_exchange)

    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed_environment(
        world_size=8, rank=rank, local_rank=rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(8))],
        local_rank=rank,
        backend="hccl",
        group_name="large_dcp8_regression",
        use_device_communicator=False,
    )
    retained = []
    for tokens in (48, 192):
        base = (torch.arange(tokens * 64 * 512).reshape(tokens, 64, 512) % 31 - 15).float() / 32
        output = (base + rank / 16).to(torch.bfloat16).npu()
        lse = torch.zeros((tokens, 64, 1), device="npu", dtype=torch.float32)
        assert can_exchange(output, lse, group.world_size)

        def invoke(output=output, lse=lse):
            return torch.ops.vllm.dcp_a2a_fused(output, lse, 8, 1, group.unique_name)

        for _ in range(5):
            invoke()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        calls_before_capture = calls.get(tokens, 0)
        with torch.npu.graph(graph):
            result = invoke()
        assert calls.get(tokens, 0) > calls_before_capture, "registered capture bypassed the new exchange"
        for step in range(8):
            output.copy_((base + rank / 16 + step / 64).to(torch.bfloat16))
            lse.fill_(step / 4)
            graph.replay()
            torch.npu.synchronize()
            # Equal LSEs give equal weights. Every arithmetic term is binary-exact.
            expected = (base[:, rank * 8 : (rank + 1) * 8] + 3.5 / 16 + step / 64).to(torch.bfloat16)
            assert torch.equal(result.cpu().view(torch.int16), expected.contiguous().view(torch.int16))
        retained.append((graph, result, output, lse))
    dist.barrier()
    retained.clear()
    del graph, result, output, lse
    gc.collect()
    torch.npu.synchronize()
    group.destroy()
    destroy_distributed_environment()


def _nonuniform_inputs(tokens, step):
    rng = torch.Generator().manual_seed(16350 + step)
    outputs = torch.randn(8, tokens, 64, 512, generator=rng).to(torch.bfloat16)
    lses = torch.randn(8, tokens, 64, 1, generator=rng) * 4
    # Repeat special cases in every receiver's head shard.
    lses[:, :, 0::8] = -torch.inf
    outputs[:, :, 0::8] = torch.nan
    lses[:4, :, 1::8] = -torch.inf
    outputs[:4, :, 1::8] = torch.nan
    lses[:, :, 2::8] = -80
    lses[step % 8, :, 2::8] = 80
    lses[:, :, 3::8] = torch.arange(8).view(8, 1, 1, 1) * 1e-3
    for sender, value in enumerate((torch.nan, torch.inf, -torch.inf)):
        lses[sender, :, 4::8] = value
        outputs[sender, :, 4::8] = torch.nan
    return outputs, lses


def _fp64_merge_reference(outputs, lses, rank):
    values = outputs[:, :, rank * 8 : (rank + 1) * 8].double()
    stats = lses[:, :, rank * 8 : (rank + 1) * 8].double()
    valid = torch.isfinite(stats)
    safe = stats.masked_fill(~valid, -torch.inf)
    maximum = safe.amax(dim=0, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    exponent = torch.exp(safe - maximum)
    denominator = exponent.sum(dim=0, keepdim=True)
    weights = exponent / torch.where(denominator > 0, denominator, 1.0)
    terms = values.masked_fill(~valid, 0.0) * weights
    return terms.sum(dim=0), terms.abs().sum(dim=0)


@pytest.mark.skipif(RANKS != 8, reason="DCP8-specific precision regression")
def test_nonuniform_large_batch_registered_graph_against_fp64(monkeypatch):
    """Check FP32 math separately from the registered BF16 output rounding.

    FP32 error budget: 1e-6 + 8 * eps32 * sum(abs(weighted contributions)).
    BF16 output additionally permits half of one local BF16 ULP. This is a
    dtype-rounding bound against FP64, not a relaxation of the old strict
    BF16-to-BF16 test above (which remains unchanged).
    """
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        init_distributed_environment,
        init_model_parallel_group,
    )

    from vllm_ascend.ops.triton.dcp import dcp_a2a as dispatch
    from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import exchange

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    init_distributed_environment(
        world_size=8, rank=rank, local_rank=local_rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(8))],
        local_rank=local_rank,
        backend="hccl",
        group_name="nonuniform_dcp8",
        use_device_communicator=False,
    )
    calls = []

    def observed(output, lse, process_group):
        calls.append(output.shape[0])
        return exchange(output, lse, process_group)

    monkeypatch.setattr(dispatch, "exchange", observed)
    retained = []
    for tokens in (13, 191, 192):
        outputs, lses = _nonuniform_inputs(tokens, 0)
        owner = outputs[rank].transpose(0, 1).contiguous().npu()
        output = owner.transpose(0, 1)
        head_major_lse = tokens != 191
        lse_owner = (lses[rank].transpose(0, 1).contiguous() if head_major_lse else lses[rank]).npu()
        lse = lse_owner.transpose(0, 1) if head_major_lse else lse_owner

        def invoke(output=output, lse=lse):
            fp32 = exchange(output, lse, group.device_group)
            bf16 = torch.ops.vllm.dcp_a2a_fused(output, lse, 8, 1, group.unique_name)
            return fp32, bf16

        for _ in range(3):
            invoke()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        before = len(calls)
        with torch.npu.graph(graph):
            fp32, bf16 = invoke()
        assert len(calls) > before, "registered capture bypassed optimized exchange"
        for step in range(4):
            outputs, lses = _nonuniform_inputs(tokens, step)
            owner.copy_(outputs[rank].transpose(0, 1).contiguous())
            lse_owner.copy_(lses[rank].transpose(0, 1).contiguous() if head_major_lse else lses[rank])
            graph.replay()
            reference, magnitude = _fp64_merge_reference(outputs, lses, rank)
            actual32, actual16 = fp32.cpu().double(), bf16.cpu().double()
            roundoff = 1e-6 + 8 * torch.finfo(torch.float32).eps * magnitude
            rounded = reference.to(torch.bfloat16)
            up = torch.nextafter(rounded, torch.full_like(rounded, torch.inf)).double()
            down = torch.nextafter(rounded, torch.full_like(rounded, -torch.inf)).double()
            half_ulp = torch.maximum(up - rounded.double(), rounded.double() - down) / 2
            assert torch.isfinite(actual32).all() and torch.isfinite(actual16).all()
            assert ((actual32 - reference).abs() <= roundoff).all()
            assert ((actual16 - reference).abs() <= half_ulp + roundoff).all()
            assert torch.count_nonzero(actual32[:, 0]) == torch.count_nonzero(actual16[:, 0]) == 0
        retained.append((graph, fp32, bf16, owner, lse, reference, roundoff, half_ulp))
    # Revisit older buckets after later captures share the same runtime kernels.
    for saved_graph, saved32, saved16, _owner, _lse, reference, roundoff, half_ulp in reversed(retained):
        saved_graph.replay()
        torch.npu.synchronize()
        assert ((saved32.cpu().double() - reference).abs() <= roundoff).all()
        assert ((saved16.cpu().double() - reference).abs() <= half_ulp + roundoff).all()
    dist.barrier()
    retained.clear()
    del graph, fp32, bf16, owner, output, lse
    gc.collect()
    torch.npu.synchronize()
    group.destroy()
    destroy_distributed_environment()


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
    steps = 16 if RANKS == 8 and local_heads == 8 else 3
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
            send = prepare_query_head_major(qn, qr)
            gathered = torch.empty((RANKS * local_heads, tokens, 576), dtype=qn.dtype, device=qn.device)
            dist.all_gather_into_tensor(gathered, send, group=group.device_group)
            return unpack_query(gathered, (512, 64))

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
                assert actual.is_contiguous()
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
def test_exchange_tiled_resource_graph(context, tokens, local_heads, head_dim, monkeypatch):
    rank, group, retained = context
    heads = RANKS * local_heads
    owner = torch.empty((heads, tokens, head_dim), dtype=torch.bfloat16, device="npu")
    output = owner.transpose(0, 1)
    lse = torch.zeros((tokens, heads, 1), device="npu")
    assert can_exchange(output, lse, RANKS)

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("large vectors must stay on the native exchange path")

    monkeypatch.setattr(dcp_a2a, "pack_dcp_output_lse", unexpected_fallback)

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
