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
            rows = torch.tensor([row for end, count in zip(offsets[1:], kept) for row in range(end - count, end)])
            device_rows = torch.nn.functional.pad(rows, (0, compact_width * 2 - len(rows))).to(device)
            plan = SPReplayPlan.build(offsets, np.array(kept), sum(lengths), compact_width * 2, 2, rank, device_rows)
            compact_rows = rows[rank * compact_width : (rank + 1) * compact_width]
            local_rows = rows[(rows >= rank * original_width) & (rows < (rank + 1) * original_width)]
            local_rows -= rank * original_width
            # BF16 hidden values stay exactly representable; FP32 pre-mix
            # fractions must survive the shared int8 payload bit-for-bit.
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                tokens = torch.arange(original_width * 2)
                full_hidden = (tokens % 31).to(dtype).reshape(-1, 1, 1).expand(-1, 4, 128).clone()
                full_mix = (tokens.float() / 7).reshape(-1, 1).expand(-1, 4).clone()
                full_mix.view(torch.int32)[:, 0] = -2147483648  # negative zero
                full_mix.view(torch.int32)[:, 1] = 0x7FC01234  # NaN payload
                shard = slice(rank * original_width, (rank + 1) * original_width)
                hidden, mix = full_hidden[shard].to(device), full_mix[shard].to(device)
                compact_hidden, compact_mix = plan.compact(hidden, mix, dist.group.WORLD)
                restored_hidden, restored_mix = plan.restore(compact_hidden, compact_mix, dist.group.WORLD)
                for compact, restored, full, source in (
                    (compact_hidden, restored_hidden, full_hidden, hidden),
                    (compact_mix, restored_mix, full_mix, mix),
                ):
                    assert compact.is_contiguous() and restored.is_contiguous()
                    expected = full.new_zeros((compact_width, *full.shape[1:]))
                    expected[: len(compact_rows)] = full[compact_rows]
                    assert torch.equal(compact.cpu().view(torch.int8), expected.view(torch.int8))
                    expected_original = source.cpu().new_zeros(source.shape)
                    expected_original[local_rows] = source.cpu()[local_rows]
                    assert torch.equal(restored.cpu().view(torch.int8), expected_original.view(torch.int8))
    finally:
        dist.destroy_process_group()


def test_compact_sp_replay_hccl(tmp_path):
    mp.spawn(_hccl_replay_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
