# SPDX-License-Identifier: Apache-2.0
"""Independent CPU formula oracle from the public compressor specification."""

import torch


def compressor_reference(args, *, ratio, coff, eps=1e-6, rotary=2, cache_mode=1):
    x, wk, wg, cache, ape, gamma, sin, cos, table, cu, used, starts = args
    x, wk, wg = x.float(), wk.float(), wg.float()
    state = cache.clone(memory_format=torch.preserve_format)
    dim = gamma.numel()
    width = coff * dim
    bsz = cu.numel() - 1 if x.ndim == 2 else x.shape[0]
    seq = x.shape[1] if x.ndim == 3 else 0
    kv = x.reshape(-1, x.shape[-1]) @ wk.t()
    scores = x.reshape(-1, x.shape[-1]) @ wg.t()
    output = torch.zeros((*sin.shape[:-1], dim), dtype=torch.float32)
    packed_row = 0

    def cache_location(batch, pos):
        if cache_mode == 1:
            logical, slot = divmod(pos, cache.shape[1])
            if table is None:
                page = batch + logical
            elif logical >= table.shape[1]:
                return None
            else:
                page = int(table[batch, logical])
                if page == 0:
                    return None
        else:
            page = batch if table is None else int(table[batch])
            slot = pos % cache.shape[1]
        if not 0 <= page < cache.shape[0]:
            return None
        return page, slot

    for b in range(bsz):
        start = int(starts[b]) if starts is not None else 0
        begin = int(cu[b]) if cu is not None else b * seq
        n = int(used[b]) if used is not None else (int(cu[b + 1]) - begin if cu is not None else seq)
        count = (start + n) // ratio - start // ratio
        for group in range(count):
            end = (start // ratio + group + 1) * ratio
            values, logits = [], []
            for pos in range(max(0, end - coff * ratio), end):
                half = int(coff == 2 and pos >= end - ratio)
                columns = slice(half * dim, (half + 1) * dim)
                if pos >= start:
                    row = begin + pos - start
                    values.append(kv[row, columns])
                    logits.append(scores[row, columns] + ape[pos % ratio, columns])
                else:
                    location = cache_location(b, pos)
                    if location is None:
                        continue
                    values.append(cache[location][columns])
                    logits.append(cache[location][width:][columns])
            if values:
                pooled = (torch.stack(values) * torch.stack(logits).softmax(dim=0)).sum(dim=0)
                normed = pooled * (pooled.square().mean() + eps).rsqrt() * gamma
                row = packed_row if x.ndim == 2 else b * output.shape[1] + group
                tail = normed[-64:]
                if rotary == 2:
                    rotated = torch.stack((-tail[1::2], tail[::2]), dim=-1).flatten()
                else:
                    rotated = torch.cat((-tail[32:], tail[:32]))
                normed[-64:] = tail * cos.reshape(-1, 64)[row] + rotated * sin.reshape(-1, 64)[row]
                output.reshape(-1, dim)[row] = normed
            packed_row += 1
        end = start + n
        tail_start = start
        for pos in range(tail_start, end):
            location = cache_location(b, pos)
            if location is not None:
                state[location][:width] = kv[begin + pos - start]
                state[location][width:] = scores[begin + pos - start] + ape[pos % ratio]
    return output, state
