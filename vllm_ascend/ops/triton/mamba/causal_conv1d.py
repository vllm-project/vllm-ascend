# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore

__all__ = ["PAD_SLOT_ID", "extract_last_width"]


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)
