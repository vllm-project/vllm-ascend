from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MambaTerminalSlotView:
    """Worker-local tensors needed to resolve terminal Mamba slots exactly."""

    req_ids: tuple[str, ...]
    idx_mapping: torch.Tensor
    state_indices: torch.Tensor
    group_ids: tuple[int, ...]
    block_tables: tuple[torch.Tensor, ...]
