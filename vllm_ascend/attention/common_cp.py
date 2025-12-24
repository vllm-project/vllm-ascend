from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AscendPCPMetadata:
    q_head_idx: torch.Tensor = None
    q_tail_idx: torch.Tensor = None
    kv_with_q_head_nomask_idx: torch.Tensor = None
    kv_with_q_head_mask_idx: torch.Tensor = None
    kv_with_q_tail_nomask_idx: torch.Tensor = None
    kv_with_q_tail_mask_idx: torch.Tensor = None
    attn_mask_seqlens: torch.Tensor = None
    head_attn_nomask_seqlens: torch.Tensor = None
    tail_attn_nomask_seqlens: torch.Tensor = None
    q_full_idx: torch.Tensor = None
    pcp_prefill_mask: torch.Tensor = None
    pcp_allgather_restore_idx: Optional[list[int]] = None


@dataclass
class CPChunkedContextMetadata:
    # New for MLA (compared to FlashAttention)
    # For handling chunked prefill
    cu_seq_lens: torch.Tensor
    starts: torch.Tensor
    seq_tot: list[int]
    max_seq_lens: list[int]
    workspace: torch.Tensor
    chunk_seq_lens: torch.Tensor
    chunk_seq_lens_npu: torch.Tensor
    # for mla DCP & PCP
    padded_chunk_seq_lens_npu: torch.Tensor = None
    padded_local_chunk_seq_lens: Optional[list[list[int]]] = None
    local_context_lens_allranks: Optional[list[list[int]]] = None
    padded_local_cu_seq_lens: torch.Tensor = None
    cu_seq_lens_lst: Optional[list[list[int]]] = None
    chunk_size: Optional[int] = None
