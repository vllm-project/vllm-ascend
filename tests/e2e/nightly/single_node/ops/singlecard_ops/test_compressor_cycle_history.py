# SPDX-License-Identifier: Apache-2.0
"""CYCLE must read immutable old history while other cores write the new tail.

The 261-token continuation reproduces overlapping old reads and tail writes.
Compare every valid compressed element and retained state with disjoint
CONTINUOUS pages, including partial groups and empty batch entries.
"""

from itertools import accumulate

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

CASES = [([0], [1]), ([0], [261]), ([256], [261])]
CASES += [([start], [length]) for start in (256, 257) for length in (1, 2, 5, 257, 258)]
CASES += [([256, 259], [261, 258]), ([256, 259, 256], [261, 0, 2])]


@pytest.mark.parametrize("ratio,coff,head", [(4, 2, 128), (4, 2, 512), (128, 1, 512)])
@pytest.mark.parametrize("starts,lengths", CASES)
def test_cycle_history_matches_continuous(ratio, coff, head, starts, lengths):
    assert enable_custom_op()
    torch.npu.set_device(0)
    torch.manual_seed(7)
    batch = len(starts)
    ring = coff * ratio
    width = 2 * coff * head
    bs = 32
    device = "npu:0"
    x = torch.randn(sum(lengths), 4096, dtype=torch.bfloat16).to(device)
    wkv = (torch.randn(coff * head, 4096) * 0.02).to(dtype=torch.bfloat16, device=device)
    wgate = (torch.randn(coff * head, 4096) * 0.02).to(dtype=torch.bfloat16, device=device)
    ape = (torch.randn(ratio, coff * head) * 0.01).to(device)
    norm = torch.ones(head, dtype=torch.bfloat16, device=device)
    counts = [(s + n) // ratio - s // ratio for s, n in zip(starts, lengths)]
    groups = sum(counts)
    capacity = min(sum(lengths), sum(lengths) // ratio + batch)
    sin = torch.zeros(capacity, 64, dtype=torch.float32, device=device)
    cos = torch.ones_like(sin)
    cu = torch.tensor([0] + list(accumulate(lengths)), dtype=torch.int32, device=device)
    positions = torch.tensor(starts, dtype=torch.int32, device=device)
    ring_pages = (ring + bs - 1) // bs
    linear_pages = (max(s + n for s, n in zip(starts, lengths)) + bs - 1) // bs
    compact = torch.zeros(1 + batch * ring_pages, bs, width, dtype=torch.float32, device=device)
    continuous = torch.zeros(1 + batch * linear_pages, bs, width, dtype=torch.float32, device=device)
    ct = torch.arange(1, 1 + batch * ring_pages, dtype=torch.int32, device=device).view(batch, ring_pages)
    lt = torch.arange(1, 1 + batch * linear_pages, dtype=torch.int32, device=device).view(batch, linear_pages)
    for b, start in enumerate(starts):
        old = torch.randn(ring, width, device=device)
        for logical in range(max(0, start - ring), start):
            slot = logical % ring
            compact[1 + b * ring_pages + slot // bs, slot % bs] = old[slot]
            continuous[1 + b * linear_pages + logical // bs, logical % bs] = old[slot]
    initial = compact.clone()

    def invoke(cache, table, mode):
        return torch.ops._C_ascend.compressor(
            x,
            wkv,
            wgate,
            cache,
            ape,
            norm,
            sin,
            cos,
            state_block_table=table,
            cu_seqlens=cu,
            seqused=None,
            start_pos=positions,
            rope_head_dim=64,
            cmp_ratio=ratio,
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=1,
            cache_mode=mode,
        )

    expected = invoke(continuous, lt, 1)
    assert expected.ndim == 2 and expected.shape == (capacity, head), expected.shape
    outputs = []
    repeat = 3 if ratio == 4 and head == 128 and starts == [256] and lengths == [261] else 1
    for i in range(repeat):
        compact.copy_(initial)
        actual = invoke(compact, ct, 2)
        torch.npu.synchronize()
        assert actual.ndim == 2 and actual.shape == (capacity, head), actual.shape
        a = actual[:groups].cpu()
        e = expected[:groups].cpu()
        assert torch.isfinite(a).all() and torch.equal(a, e), (
            ratio,
            head,
            starts,
            lengths,
            "output",
            (a.float() - e.float()).abs().max().item() if groups else None,
        )
        outputs.append(a)
        for b, (start, length) in enumerate(zip(starts, lengths)):
            end = start + length
            tail_start = max(0, (end // ratio) * ratio - (coff - 1) * ratio)
            for logical in range(tail_start, end):
                slot = logical % ring
                a = compact[1 + b * ring_pages + slot // bs, slot % bs]
                e = continuous[1 + b * linear_pages + logical // bs, logical % bs]
                assert torch.equal(a, e), (ratio, head, starts, lengths, "tail", logical)
    assert all(torch.equal(outputs[0], a) for a in outputs[1:])
