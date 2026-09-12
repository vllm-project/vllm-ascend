# SPDX-License-Identifier: Apache-2.0
"""DCP8 exchange comparison; run via torchrun on an isolated eight-NPU group."""

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
