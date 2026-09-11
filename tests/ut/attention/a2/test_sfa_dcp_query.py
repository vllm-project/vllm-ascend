# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=8 -m pytest --noconftest."""

import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.sfa_dcp_query import (  # noqa: E402
    can_prepare_query,
    prepare_query_head_major,
    unpack_query,
)

pytestmark = pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 8, reason="requires an isolated DCP8 group")


def test_query_gather_raw_bits_and_changed_input_graphs():
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    retained = []
    patterns = torch.tensor([0, -32768, 1, 127, 32640, -128, 32705, -63], dtype=torch.int16)

    def bits(sender, tokens, width, step):
        return patterns.roll(sender + step).repeat(tokens * 8 * width // patterns.numel()).view(tokens, 8, width)

    for tokens in (2, 6, 12):
        n_owner = bits(rank, tokens, 512, 0).transpose(0, 1).contiguous().view(torch.bfloat16).npu()
        r_owner = bits(rank, tokens, 64, 0).view(torch.bfloat16).npu()
        qn, qr = n_owner.transpose(0, 1), r_owner
        assert can_prepare_query(qn, qr)

        def invoke(qn=qn, qr=qr, tokens=tokens):
            send = prepare_query_head_major(qn, qr)
            received = torch.empty((64, tokens, 576), dtype=qn.dtype, device=qn.device)
            dist.all_gather_into_tensor(received, send, async_op=True).wait()
            return unpack_query(received)

        for _ in range(5):
            invoke()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = invoke()
        for step in range(16):
            n_owner.copy_(bits(rank, tokens, 512, step).transpose(0, 1).contiguous().view(torch.bfloat16))
            r_owner.copy_(bits(rank, tokens, 64, step).view(torch.bfloat16))
            graph.replay()
            torch.npu.synchronize()
            for actual, width in zip(result, (512, 64)):
                expected = torch.cat([bits(sender, tokens, width, step) for sender in range(8)], dim=1)
                assert actual.is_contiguous()
                assert torch.equal(actual.view(torch.int16).cpu(), expected)
        retained.append((graph, result, n_owner, r_owner))
    dist.barrier()
    dist.destroy_process_group()
