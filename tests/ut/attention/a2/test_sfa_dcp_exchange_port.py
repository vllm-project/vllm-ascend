# SPDX-License-Identifier: Apache-2.0
"""DCP8 exchange comparison; run via torchrun on an isolated eight-NPU group."""

import gc
import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.sfa_cp import (  # noqa: E402
    fused_sfa_dcp_lse_combine,
    pack_sfa_dcp_output_lse,
    sfa_dcp_a2a_fused_combine,
)

pytestmark = pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="requires isolated DCP8")


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
            return sfa_dcp_a2a_fused_combine(output, lse, 8, 1, dist.group.WORLD)

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
            send = pack_sfa_dcp_output_lse(output, lse, 8, 1)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send)
            upstream = fused_sfa_dcp_lse_combine(recv, 512, 1)
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

    from vllm_ascend.ops.triton import sfa_cp as dispatch
    from vllm_ascend.ops.triton.sfa_dcp_exchange import can_exchange

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
        assert can_exchange(output, lse)

        def invoke(output=output, lse=lse):
            return torch.ops.vllm.sfa_dcp_a2a_fused(output, lse, 8, 1, group.unique_name, decode_token_budget=192)

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

    from vllm_ascend.ops.triton import sfa_cp as dispatch
    from vllm_ascend.ops.triton.sfa_dcp_exchange import exchange

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
        lse = lses[rank].npu()

        def invoke(output=output, lse=lse):
            fp32 = exchange(output, lse, group.device_group)
            bf16 = torch.ops.vllm.sfa_dcp_a2a_fused(output, lse, 8, 1, group.unique_name, decode_token_budget=192)
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
            lse.copy_(lses[rank])
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
        retained.append((graph, fp32, bf16, owner, lse))
    dist.barrier()
    retained.clear()
    del graph, fp32, bf16, owner, output, lse
    gc.collect()
    torch.npu.synchronize()
    group.destroy()
    destroy_distributed_environment()
