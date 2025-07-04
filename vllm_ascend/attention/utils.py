from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AscendCommonAttentionMetadata:
    """
    Attention metadata attributes that can be shared by layers in different KV
    cache groups and thus having different block table.
    """

    query_start_loc: torch.Tensor = None
    """(batch_size + 1,), the start location of each request in query Tensor"""
    seq_lens: Optional[torch.Tensor] = None
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""
    query_lens: Optional[torch.Tensor] = None
    """(batch_size,), the length of each request including only the newly
    scheduled tokens"""
    seq_lens_list: Optional[list] = None
    """(num_input_tokens,), note that this is specifically for FIA kernel"""
