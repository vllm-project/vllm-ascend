#!/usr/bin/env python3
"""Sparse load / KV gather validation: independent FP32 oracle + optional package A/B.

Run in separate processes with baseline/new ASCEND_CUSTOM_OPP_PATH. Pass --save
for the baseline and --compare for the new package; no installed OPP is modified.
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch_npu

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "build.sh").exists())
sys.path[:0] = [str(ROOT / "torch_extension"),
                str(ROOT / "attention/generic_block_sparse_attention/torch_extension")]
import generic_block_sparse_attention as ext

builder = ext.generic_block_sparse_attention_op_builder
builder.category = None
builder.sources = lambda: [str(ROOT / "attention/generic_block_sparse_attention/torch_extension/csrc/generic_block_sparse_attention.cpp")]


def cases():
    # qs, kvs, Y, GQA, D, precision, BF16, PA. Include ring wrap and tight UB.
    result = [
        (17, 17, 16, 8, 128, 0, False, False),
        (20, 257, 1, 16, 128, 0, False, False),
        (20, 509, 3, 16, 64, 1, True, False),
        (20, 777, 31, 32, 256, 0, False, True),
        (20, 1031, 127, 64, 512, 0, False, True),
        (20, 1537, 95, 128, 512, 0, True, False),
        (20, 1537, 128, 128, 256, 0, False, False),
        (1, 4097, 16, 16, 128, 0, False, False),
        (1, 4097, 31, 64, 512, 1, False, True),
        (2, 8193, 128, 128, 512, 0, False, False),
        (20, 4097, 16, 16, 128, 0, False, False),
        (20, 4097, 16, 32, 256, 1, True, True),
    ]
    for i, d in enumerate((1, 31, 32, 33, 63, 65, 127, 129, 255, 257, 383, 384, 511)):
        result.append((12, 521, (3, 16, 63)[i % 3], (16, 32, 64, 128)[i % 4],
                       d, i % 2, i % 3 == 0, i % 2 == 0))
    result += [
        (1, 16384, 16, 1, 128, 0, False, False),  # AIV with no Q rows still prepares addresses.
        (20, 1025, 17, 3, 129, 1, True, True),
        (20, 2049, 1, 128, 512, 0, False, False),  # Maximum descriptor count + LO GM.
        (20, 257, 128, 1, 512, 0, False, False),  # Empty/residual-only tasks.
    ]
    result += [
        (20, 8193, 128, 16, 128, 0, False, True),  # UB capacity below blockY: split a full block.
        (20, 8193, 127, 16, 128, 0, False, True),  # UB limit + page + AIV-half boundaries.
        (20, 4097, 65, 16, 128, 0, True, True),
        (1, 8193, 127, 16, 128, 0, False, True),   # Same boundaries within FD slices.
    ]
    return result


def run(case, case_id, mode, device, kv_heads=None):
    qs, kvs, y, group, d, precision, bf16, paged = case
    dtype = torch.bfloat16 if bf16 else torch.float16
    # Two batches and two KV heads exercise storage/head bases and multiple tasks.
    q_lengths, kv_lengths, heads = ([qs], [kvs], 1) if qs <= 2 else ([qs, 3], [kvs, 137], 2)
    heads = heads if kv_heads is None else kv_heads
    generator = torch.Generator().manual_seed(240926 + case_id)
    q = (torch.randn(sum(q_lengths), heads * group, d, generator=generator) * .2).to(dtype)
    logical_k = [(torch.randn(k, heads, d, generator=generator) * .2).to(dtype) for k in kv_lengths]
    logical_v = [(torch.randn(k, heads, d, generator=generator) * .2).to(dtype) for k in kv_lengths]
    page_size = 128
    if paged:
        pages_per_batch = [math.ceil(k / page_size) for k in kv_lengths]
        total_pages = sum(pages_per_batch)
        permutation = torch.randperm(total_pages, generator=generator)
        table = torch.zeros((len(q_lengths), max(pages_per_batch)), dtype=torch.int32)
        k_host = torch.zeros(total_pages, page_size, heads, d, dtype=dtype)
        v_host = torch.zeros_like(k_host)
        pos = 0
        for b, num in enumerate(pages_per_batch):
            for page in range(num):
                physical = permutation[pos].item()
                pos += 1
                table[b, page] = physical
                size = min(page_size, kv_lengths[b] - page * page_size)
                k_host[physical, :size] = logical_k[b][page * page_size:page * page_size + size]
                v_host[physical, :size] = logical_v[b][page * page_size:page * page_size + size]
        # Different physical page strides for K and V, not merely a page permutation.
        k = torch.zeros(total_pages * 2, page_size, heads, d, dtype=dtype, device=device)[::2]
        v = torch.zeros(total_pages * 3, page_size, heads, d, dtype=dtype, device=device)[::3]
        k.copy_(k_host)
        v.copy_(v_host)
        block_table, layout = table.to(device), "PA_BBND"
    else:
        k, v = torch.cat(logical_k).to(device), torch.cat(logical_v).to(device)
        block_table, layout = None, "TND"
    total_q = sum(q_lengths)
    capacity = max(math.ceil(k / y) for k in kv_lengths)
    idx = torch.full((heads, total_q, capacity), -1, dtype=torch.int32)
    count = torch.zeros(heads, total_q, dtype=torch.int32)
    golden = torch.zeros_like(q, dtype=torch.float32)
    golden_lse = torch.full((total_q, heads * group), -float("inf"))
    base = 0
    for b, (qlen, klen) in enumerate(zip(q_lengths, kv_lengths)):
        for row in range(qlen):
            visible = klen - qlen + row + 1
            for head in range(heads):
                selected = list(range(0, visible // y, head + 2))
                if mode == 0 and visible % y:
                    selected.append(visible // y)
                if case_id == 28 and row % 3 == 0:
                    selected = []
                if selected:
                    idx[head, base + row, :len(selected)] = torch.tensor(selected)
                count[head, base + row] = len(selected)
                full = selected + ([visible // y] if mode == 1 and visible % y else [])
                tokens = [t for block in full for t in range(block * y, min((block + 1) * y, visible))]
                if not tokens:
                    continue
                sl = slice(head * group, (head + 1) * group)
                scores = q[base + row, sl].float() @ logical_k[b][tokens, head].float().T / math.sqrt(d)
                golden[base + row, sl] = scores.softmax(-1) @ logical_v[b][tokens, head].float()
                golden_lse[base + row, sl] = scores.logsumexp(-1)
        base += qlen
    cu_q = torch.tensor([0] + list(torch.tensor(q_lengths).cumsum(0).tolist()), dtype=torch.int64, device=device)
    cu_kv = torch.tensor([0] + list(torch.tensor(kv_lengths).cumsum(0).tolist()), dtype=torch.int64, device=device)
    used = torch.tensor(kv_lengths, dtype=torch.int32, device=device)
    idx, count = idx.to(device), count.to(device)
    common = dict(cu_seqlens_q=cu_q, cu_seqlens_kv=None if paged else cu_kv, seqused_kv=used,
                  layout_q="TND", layout_kv=layout, layout_sparse_pattern=4, mask_mode=1,
                  quant_mode=0, softmax_precision=precision, win_left=-1, win_right=-1,
                  residual_block_mode=mode, is_consistent_topk=False)
    metadata = ext.generic_block_sparse_attention_metadata(
        idx, count, heads * group, heads, d, [1, y], max_seqlen_q=max(q_lengths),
        max_seqlen_kv=max(kv_lengths), **common)
    output, lse = ext.generic_block_sparse_attention(
        q.to(device), k, v, idx, count, [1, y], metadata=metadata, block_table=block_table,
        softmax_scale=1 / math.sqrt(d), return_softmax_lse=False, **common)
    torch_npu.npu.synchronize()
    output = output.cpu().float()
    # Softmax precision differs by route; an independent golden catches mapping
    # errors. Exact baseline equality is tested separately with --compare.
    tolerance = .006 if bf16 else .002
    torch.testing.assert_close(output, golden, atol=tolerance, rtol=tolerance)
    print(f"PASS case={case_id} residual={mode} config={case} kv_heads={heads} FD={int(metadata[5].cpu())} "
          f"golden_max_abs={(output - golden).abs().max().item():.7g}", flush=True)
    return (output,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--cases", type=int, nargs="*")
    parser.add_argument("--kv-heads", type=int, choices=(1, 2), help="override KV heads for contiguous DMA coverage")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch_npu.npu.set_device(args.device)
    baseline = torch.load(args.compare, weights_only=True) if args.compare else None
    outputs = {}
    for i, case in enumerate(cases()):
        if args.cases is not None and i not in args.cases:
            continue
        for mode in (0, 1):
            key = f"{i}:{mode}"
            print(f"RUN {key} {case}", flush=True)
            outputs[key] = run(case, i, mode, f"npu:{args.device}", args.kv_heads)
            if baseline is not None:
                for actual, expected in zip(outputs[key], baseline[key]):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                print(f"BASELINE EXACT {key}", flush=True)
    if args.save:
        torch.save(outputs, args.save)
    print(f"ALL PASS: {len(outputs)} cases", flush=True)


if __name__ == "__main__":
    main()
