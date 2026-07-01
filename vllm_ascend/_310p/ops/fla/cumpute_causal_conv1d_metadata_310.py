import numpy as np
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def compute_causal_conv1d_metadata(
    query_start_loc_p_cpu: torch.Tensor,
    *,
    device: torch.device,
):
    assert query_start_loc_p_cpu.device.type == "cpu"
    seqlens = query_start_loc_p_cpu.diff()
    nums_dict: dict[int, dict[str, object]] = {}
    batch_ptr = None
    token_chunk_offset_ptr = None
    batch_ptr_cpu = None
    token_chunk_offset_ptr_cpu = None

    for BLOCK_M in [8]:
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums.numpy()))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(mlist)
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2

        offset_items: list[int] = []
        for idx, num in enumerate(nums):
            offset_items.extend(range(num))
        offsetlist = torch.tensor(offset_items, dtype=torch.int32)

        if batch_ptr is None or batch_ptr.numel() < MAX_NUM_PROGRAMS:
            batch_ptr_cpu = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32)
            token_chunk_offset_ptr_cpu = torch.full((MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32)
            if device.type == "cpu":
                batch_ptr = batch_ptr_cpu
                token_chunk_offset_ptr = token_chunk_offset_ptr_cpu
            else:
                batch_ptr = batch_ptr_cpu.to(device, non_blocking=False)
                token_chunk_offset_ptr = token_chunk_offset_ptr_cpu.to(device, non_blocking=False)
        else:
            batch_ptr_cpu.fill_(PAD_SLOT_ID)
            token_chunk_offset_ptr_cpu.fill_(PAD_SLOT_ID)

        batch_ptr_cpu[:mlist_len].copy_(mlist.to(torch.int32))
        token_chunk_offset_ptr_cpu[:mlist_len].copy_(offsetlist)
        if device.type != "cpu":
            batch_ptr.copy_(batch_ptr_cpu, non_blocking=True)
            token_chunk_offset_ptr.copy_(token_chunk_offset_ptr_cpu, non_blocking=True)

        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr

    return nums_dict, batch_ptr, token_chunk_offset_ptr
