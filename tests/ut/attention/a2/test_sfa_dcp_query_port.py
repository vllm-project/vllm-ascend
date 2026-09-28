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


@pytest.fixture(scope="module")
def dcp_context():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    # HCCL capture buffers must remain alive across all parameterized cases
    # sharing this communicator, just as serving retains its graph buckets.
    retained: list[tuple] = []
    yield dist.group.WORLD, retained
    # Later captures must not overwrite runtime arguments of older buckets.
    for graph, result, _n_owner, _r_owner, expected_pair in reversed(retained):
        graph.replay()
        torch.npu.synchronize()
        for actual, expected in zip(result, expected_pair):
            assert torch.equal(actual.view(torch.int16).cpu(), expected)
    torch.npu.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    retained.clear()


@pytest.mark.parametrize("column_stride", [1, 2])
def test_query_gather_raw_bits_and_changed_input_graphs(column_stride, dcp_context):
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dcp_group, retained = dcp_context
    patterns = torch.tensor([0, -32768, 1, 127, 32640, -128, 32705, -63], dtype=torch.int16)

    def bits(sender, tokens, width, step):
        rows = torch.arange(tokens * 8).view(tokens, 8, 1)
        columns = torch.arange(width).view(1, 1, width)
        value = ((rows * 257 + columns * 17 + sender * 4099 + step * 8191) % 65536 - 32768).to(torch.int16)
        # Preserve special BF16 bit patterns, but make token/head addressing
        # observable instead of repeating the same row everywhere.
        value[..., : patterns.numel()] = patterns.roll(sender + step)
        assert not torch.equal(value[0, 0], value[0, 1])
        assert not torch.equal(value[0, 0], value[1, 0])
        return value

    for tokens in (2, 6, 12):
        n_owner = bits(rank, tokens, 512 * column_stride, 0).transpose(0, 1).contiguous().view(torch.bfloat16).npu()
        r_owner = bits(rank, tokens, 64 * column_stride, 0).view(torch.bfloat16).npu()
        qn, qr = n_owner.transpose(0, 1)[..., ::column_stride], r_owner[..., ::column_stride]
        assert can_prepare_query(qn, qr)

        def invoke(qn=qn, qr=qr, tokens=tokens):
            send = prepare_query_head_major(qn, qr)
            received = torch.empty((64, tokens, 576), dtype=qn.dtype, device=qn.device)
            dist.all_gather_into_tensor(received, send, group=dcp_group, async_op=True).wait()
            return unpack_query(received)

        for _ in range(5):
            invoke()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = invoke()
        for step in range(16):
            n_owner.copy_(
                bits(rank, tokens, 512 * column_stride, step).transpose(0, 1).contiguous().view(torch.bfloat16)
            )
            r_owner.copy_(bits(rank, tokens, 64 * column_stride, step).view(torch.bfloat16))
            graph.replay()
            torch.npu.synchronize()
            for actual, width in zip(result, (512, 64)):
                expected = torch.cat(
                    [bits(sender, tokens, width * column_stride, step)[..., ::column_stride] for sender in range(8)],
                    dim=1,
                )
                assert actual.is_contiguous()
                assert torch.equal(actual.view(torch.int16).cpu(), expected)
        expected_pair = tuple(
            torch.cat(
                [bits(sender, tokens, width * column_stride, 15)[..., ::column_stride] for sender in range(8)], dim=1
            )
            for width in (512, 64)
        )
        retained.append((graph, result, n_owner, r_owner, expected_pair))
