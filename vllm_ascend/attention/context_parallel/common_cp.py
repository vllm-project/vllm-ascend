from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu
from vllm.distributed import get_dcp_group, get_decode_context_model_parallel_world_size, get_pcp_group


@dataclass
class AscendPCPMetadata:
    """
    Metadata for Prefill Context Parallelism (PCP) on Ascend devices.

    Stores index tensors and sequence lengths for routing attention
    computations across PCP ranks during long sequence processing.
    """

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
    pcp_allgather_restore_idx: list[int] | None = None


@dataclass
class CPChunkedContextMetadata:
    """
    Metadata for chunked context handling in Context Parallelism (CP).

    Extends chunked prefill with per-rank chunk information for PCP/DCP.
    """

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
    padded_local_chunk_seq_lens: list[list[int]] | None = None
    local_context_lens_allranks: list[list[int]] | None = None
    padded_local_cu_seq_lens: torch.Tensor = None
    cu_seq_lens_lst: list[list[int]] | None = None
    chunk_size: int | None = None


@dataclass
class AscendMetadataForPrefill:
    """Prefill-specific metadata for Ascend attention with Context Parallelism."""

    @dataclass
    class ChunkedContextMetadata:
        """Metadata for chunked context processing within prefill phase."""

        actual_chunk_seq_lengths: torch.Tensor
        actual_seq_lengths_kv: torch.Tensor
        starts: torch.Tensor
        chunk_seq_mask_filtered_indices: torch.Tensor
        chunked_req_mask: list[bool] | None = None
        local_context_lens_allranks: list[list[int]] | None = None
        cp_kv_recover_idx_for_chunk: list[int] | None = None
        kv_inverse_idx_for_chunk: list[int] | None = None
        batch_chunk_seq_mask: list[bool] | None = None
        local_total_toks: int | None = None

    """ Prefill Specific Metadata for Ascend"""
    pcp_metadata: AscendPCPMetadata | None = None
    chunked_context: ChunkedContextMetadata | None = None
    block_tables: torch.Tensor = None
    actual_seq_lengths_q: torch.Tensor = None


@dataclass
class AscendMetadataForDecode:
    """Decode-specific metadata for Ascend attention with Context Parallelism."""

    num_computed_tokens_of_pcp_dcp: list[list[list[int]]] | None = None
    block_tables: torch.Tensor = None


def _process_attn_out_lse(attn_output: torch.Tensor, softmax_lse: torch.Tensor) -> torch.Tensor:
    pcp_size = get_pcp_group().world_size
    dcp_size = get_decode_context_model_parallel_world_size()
    dcp_group = get_dcp_group().device_group if dcp_size > 1 else None
    softmax_lse = softmax_lse.to(torch.float32)
    attn_output = attn_output.to(torch.float32)
    # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
    attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
    if dcp_size > 1:
        # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
        attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
        attn_out_lse_all2all = torch.empty_like(attn_out_lse)
        dist.all_to_all_single(attn_out_lse_all2all, attn_out_lse, group=dcp_group)
        attn_out_lse = attn_out_lse_all2all.permute([2, 0, 1])

    if pcp_size > 1:
        # AllGather out&lse within CP group
        attn_out_lse = get_pcp_group().all_gather(attn_out_lse.contiguous(), dim=0)

    return attn_out_lse


def _npu_attention_update(head_size, attn_out_lse: torch.Tensor) -> torch.Tensor:
    pcp_size = get_pcp_group().world_size
    dcp_size = get_decode_context_model_parallel_world_size()
    # [PCP * S, DCP * H, D+1]
    B_total, H_total, D_plus_1 = attn_out_lse.shape
    S = B_total // pcp_size
    H = H_total // dcp_size
    D = head_size
    assert D_plus_1 == D + 1
    # [PCP, S, DCP, H, D+1]
    x = attn_out_lse.view(pcp_size, S, dcp_size, H, D_plus_1)
    # [PCP, DCP, S, H, D+1]
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    # Flatten [N, S, H, D+1], N = pcp_size * dcp_size
    x = x.view(-1, S, H, D_plus_1)
    # Split out lse
    out_flat, lse_flat = torch.split(x, [D, 1], dim=-1)  # [N, S, H, D], [N, S, H, 1]
    #    out: [N, S, H, D] -> [N, S*H, D]
    #    lse: [N, S, H, 1] -> [N, S*H]
    out_flat = out_flat.flatten(1, 2)  # [N, S*H, D]
    lse_flat = lse_flat.flatten(1, -1)  # [N, S*H]
    #  unbind to list
    out_list = out_flat.unbind(0)  # [S*H, D]
    lse_list = lse_flat.unbind(0)  # [S*H]
    attn_out, _ = torch_npu.npu_attention_update(lse_list, out_list, 0)
    attn_out = attn_out.view(-1, H, D)
    return attn_out
