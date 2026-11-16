# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate compact SP exchanges on HCCL, including empty send/receive ranks."""

from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.decoder_replay_sp import SPReplayPlan


def _hccl_replay_worker(rank, init_method):
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    dist.init_process_group("hccl", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        for lengths, kept in (([4096], [128]), ([4096, 1], [128, 1]), ([33, 35, 2], [30, 2, 2]), ([16], [1])):
            offsets = np.insert(np.cumsum(lengths), 0, 0)
            original_width = (sum(lengths) + 1) // 2
            compact_width = (sum(kept) + 1) // 2
            plan = SPReplayPlan.build(offsets, np.array(kept), sum(lengths), compact_width * 2, 2, rank, device)
            rows = torch.tensor([row for end, count in zip(offsets[1:], kept) for row in range(end - count, end)])
            compact_rows = rows[rank * compact_width : (rank + 1) * compact_width]
            local_rows = rows[(rows >= rank * original_width) & (rows < (rank + 1) * original_width)]
            local_rows -= rank * original_width
            # Hidden states, pre-mix, routing IDs and padding masks keep their
            # actual communication dtypes; BF16 values stay exactly representable.
            for dtype, shape in (
                (torch.bfloat16, (4, 128)),
                (torch.float32, (4,)),
                (torch.int64, ()),
                (torch.int32, ()),
                (torch.int8, ()),
            ):
                full = (torch.arange(original_width * 2) % 31).to(dtype)
                full = full.reshape(-1, *(1 for _ in shape)).expand(-1, *shape).clone()
                source = full[rank * original_width : (rank + 1) * original_width].to(device)
                compact = plan.compact(source, dist.group.WORLD)
                expected = torch.zeros((compact_width, *shape), dtype=dtype)
                expected[: len(compact_rows)] = full[compact_rows]
                torch.testing.assert_close(compact.cpu(), expected, rtol=0, atol=0)

                restored = plan.restore(compact, dist.group.WORLD)
                expected_original = torch.zeros((original_width, *shape), dtype=dtype)
                expected_original[local_rows] = source.cpu()[local_rows]
                torch.testing.assert_close(restored.cpu(), expected_original, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_compact_sp_replay_hccl(tmp_path):
    mp.spawn(_hccl_replay_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
