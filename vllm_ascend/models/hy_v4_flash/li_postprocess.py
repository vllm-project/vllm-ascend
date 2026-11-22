"""Integer index expansion used by the HY4 Prefill LightningIndexer adapter."""

import torch


def expand_blocks(groups, offsets, ratio, use_alpha):
    if use_alpha:
        return torch.add(offsets, groups[..., None], alpha=ratio)
    return groups[..., None] * ratio + offsets


def current_tail(positions, offsets, ratio, use_alpha):
    positions = positions.to(torch.int32)
    tail = expand_blocks(positions // ratio, offsets, ratio, use_alpha)
    return tail.masked_fill_(tail > positions[:, None], -1)


def merge_group_rows(chunks, sentinel, offsets, tail, ratio, use_alpha):
    groups = torch.cat(chunks, dim=0)
    ordered = groups.masked_fill(groups < 0, sentinel).sort(dim=-1).values
    history = expand_blocks(ordered, offsets, ratio, use_alpha)
    history.masked_fill_(ordered[..., None] == sentinel, -1)
    return torch.cat((history.flatten(1), tail), dim=-1)
