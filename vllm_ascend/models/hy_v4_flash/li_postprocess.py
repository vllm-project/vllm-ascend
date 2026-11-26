"""Integer index expansion used by the HY4 Prefill LightningIndexer adapter."""

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import is_950


def expand_blocks(groups, offsets, ratio):
    return torch.add(offsets, groups[..., None], alpha=ratio)


def current_tail(positions, offsets, ratio):
    positions = positions.to(torch.int32)
    tail = expand_blocks(positions // ratio, offsets, ratio)
    return tail.masked_fill_(tail > positions[:, None], -1)


def merge_group_rows(groups, sentinel, offsets, tail, ratio, *, consume_groups=False):
    use_fused = (
        HAS_TRITON
        and groups.device.type == "npu"
        and is_950()
        and groups.dtype == tail.dtype == torch.int32
        and groups.ndim == 2
        and groups.shape[1] == 511
        and (16 <= groups.shape[0] <= 512 or (consume_groups and 768 <= groups.shape[0] <= 4096))
        and groups.is_contiguous()
        and tail.device == groups.device
        and tail.shape == (groups.shape[0], 4)
        and tail.is_contiguous()
        and ratio == 4
    )
    use_float_sort = (
        groups.device.type == "npu"
        and is_950()
        and groups.dtype == torch.int32
        and groups.ndim == 2
        and groups.shape[1] == 511
        and groups.shape[0] >= 1872
        and 0 <= sentinel <= 2**24
    )
    if use_float_sort:
        # Legal LI group IDs are below sentinel and remain exact in FP32.
        if HAS_TRITON and groups.is_contiguous():
            from .prefill_ops.li_tail import mask_group_ids

            sortable = mask_group_ids(groups, sentinel, torch.float32)
        else:
            sortable = groups.masked_fill(groups < 0, sentinel).float()
        ordered = sortable.sort(dim=-1).values
    elif consume_groups and groups.device.type == "npu" and is_950() and 768 <= groups.shape[0] < 1872:
        # LI returns private group IDs. The out-of-place mask first clones them.
        ordered = groups.masked_fill_(groups < 0, sentinel).sort(dim=-1).values
    else:
        ordered = groups.masked_fill(groups < 0, sentinel).sort(dim=-1).values
    if use_fused:
        from .prefill_ops.li_tail import expand_sorted_group_ids

        return expand_sorted_group_ids(ordered, sentinel, tail)
    history = expand_blocks(ordered.to(groups.dtype), offsets, ratio)
    history.masked_fill_(ordered[..., None] == sentinel, -1)
    return torch.cat((history.flatten(1), tail), dim=-1)
