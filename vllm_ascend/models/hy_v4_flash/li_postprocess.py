"""Integer index expansion used by the HY4 Prefill LightningIndexer adapter."""
import torch


def expand_blocks(groups, offsets, ratio):
    return torch.add(offsets, groups[..., None], alpha=ratio)


def current_tail(positions, offsets, ratio):
    positions = positions.to(torch.int32)
    tail = expand_blocks(positions // ratio, offsets, ratio)
    return tail.masked_fill_(tail > positions[:, None], -1)


def merge_group_rows(groups, sentinel, offsets, tail, ratio):
    ordered = groups.masked_fill(groups < 0, sentinel).sort(dim=-1).values
    history = expand_blocks(ordered, offsets, ratio)
    history.masked_fill_(ordered[..., None] == sentinel, -1)
    return torch.cat((history.flatten(1), tail), dim=-1)
