from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from vllm.distributed import get_dcp_group, get_pcp_group, get_tp_group

import vllm_ascend.ops.triton.dcp.dcp_a2a  # noqa: F401
from vllm_ascend.distributed.utils import get_decode_context_model_parallel_world_size
from vllm_ascend.utils import model_uses_kpool_indexer


def is_pcp_decode_sharding_enabled(vllm_config) -> bool:
    """Shard decode requests for PCP without speculation.

    Graph execution follows the sharded path as well. Speculative decoding
    stays on the replicated path in this change. The decision is derived from
    declared vLLM fields so cloning a draft ``ParallelConfig`` with
    ``replace()`` does not see an undeclared attribute. KPool models keep
    replicated decodes so every PCP rank advances its TP-sharded Mamba state
    in the same request order.
    """
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.prefill_context_parallel_size > 1
        and parallel_config.decode_context_parallel_size == 1
        and vllm_config.speculative_config is None
        and not model_uses_kpool_indexer(getattr(vllm_config, "model_config", None))
    )


def get_pcp_num_replicated_tokens(num_decode_tokens: int, is_decode_sharded: bool) -> int:
    """Return the leading rank-local tokens that every PCP rank computes identically.

    Replicated decodes need no KV gather. A sharded decode belongs to one PCP
    rank, so its KV is gathered together with the prefill tokens.
    """
    return 0 if is_decode_sharded else num_decode_tokens


class CPKVScope(str, Enum):
    """KV ranges used by context-parallel attention."""

    HISTORY = "history"
    CURRENT = "current"
    FULL = "full"


def use_history_current_split_decode(
    attn_metadata: Any,
    *,
    is_draft_model: bool = False,
    is_draft_model_prefill: bool = False,
    use_spec_decode: bool = False,
) -> bool:
    """Choose one split policy for MLA and GQA decode attention."""
    if not attn_metadata.causal or attn_metadata.decode is None:
        return False
    # Preserve the task layout of speculative target and draft-prefill graphs.
    if is_draft_model_prefill or (use_spec_decode and not is_draft_model):
        return True
    decode = attn_metadata.decode
    assert decode is not None and decode.actual_seq_lengths_q is not None
    query_ends = decode.actual_seq_lengths_q
    return any(end - start > 1 for start, end in zip([0] + query_ends[:-1], query_ends))


def get_cp_local_query_key_lens(
    query_start_loc: torch.Tensor,
    cum_query_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    local_start: int,
    local_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cumulative query lengths and causal KV lengths for a token shard.

    ``local_end`` includes CP padding. Requests with no local queries must
    have zero KV length, including graph-padding requests with zero seq_lens.
    Callers copy the results into their own graph-stable metadata buffers.
    """
    global_start = query_start_loc[: cum_query_lens.shape[0]]
    req_local_start = global_start.clamp(min=local_start)
    req_local_end = cum_query_lens.clamp(max=local_end)
    num_local_tokens = req_local_end - req_local_start
    local_query_lens = torch.cumsum(num_local_tokens.clamp(min=0), dim=0)
    offset = cum_query_lens - req_local_end
    local_key_lens = torch.where(num_local_tokens > 0, torch.clamp_min(seq_lens - offset, 0), 0)
    return local_query_lens, local_key_lens


def build_pcp_ordered_slot_mapping(
    global_slot_mapping: torch.Tensor,
    pcp_context: Any,
    slot_mapping_buffer: torch.Tensor,
) -> torch.Tensor:
    """Write slots in PCP gather order into a caller-owned output buffer."""
    gather_idx = pcp_context.padded_gather_idx
    write_mask = pcp_context.gathered_kv_write_mask
    if gather_idx is None or write_mask is None:
        raise RuntimeError("PCP+DCP prefill requires the PCP gathered-token layout.")
    num_tokens = gather_idx.numel()
    if slot_mapping_buffer.shape[0] < num_tokens:
        raise RuntimeError(
            f"PCP+DCP indexer slot buffer is too small: capacity={slot_mapping_buffer.shape[0]}, required={num_tokens}."
        )
    slot_mapping = slot_mapping_buffer[:num_tokens]
    torch.index_select(global_slot_mapping, 0, gather_idx, out=slot_mapping)
    slot_mapping.masked_fill_(~write_mask, -1)
    return slot_mapping


class DCPMetadataBuilderMixin:
    """Shared DCP metadata access for backend-specific metadata builders."""

    dcp_size: int
    dcp_rank: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dcp_group = get_dcp_group()
        self.dcp_size = dcp_group.world_size
        self.dcp_rank = dcp_group.rank_in_group

    @staticmethod
    def _require_dcp_metadata(
        common_attn_metadata: Any,
    ) -> Any:
        dcp_metadata = common_attn_metadata.context_parallel_metadata
        if dcp_metadata is None or dcp_metadata.num_computed_tokens_of_dcp is None:
            raise AssertionError("DCP metadata must be populated.")
        return dcp_metadata

    def _get_dcp_context_lens(
        self,
        common_attn_metadata: Any,
        *,
        start: int = 0,
        end: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        dcp_metadata = self._require_dcp_metadata(common_attn_metadata)
        context_lens = torch.as_tensor(
            dcp_metadata.num_computed_tokens_of_dcp[start:end],
            dtype=torch.int32,
            device=device,
        )
        return context_lens.reshape(-1, self.dcp_size)

    def _get_dcp_rank_context_lens(
        self,
        common_attn_metadata: Any,
        *,
        start: int = 0,
        end: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return self._get_dcp_context_lens(
            common_attn_metadata,
            start=start,
            end=end,
            device=device,
        )[:, self.dcp_rank]


class DCPImplMixin:
    """Shared DCP group lifecycle and collectives for attention backends."""

    # vLLM #55780 defaults implementations to no DCP; these implement it.
    supports_dcp = True

    dcp_size: int
    dcp_rank: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dcp_group = get_dcp_group()
        self.pcp_group = get_pcp_group()
        self.tp_group = get_tp_group()
        self.dcp_size = self.dcp_group.world_size
        self.dcp_rank = self.dcp_group.rank_in_group
        self.dcp_device_group = self.dcp_group.device_group if self.dcp_size > 1 else None

    def _dcp_all_gather(
        self,
        tensor: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        if self.dcp_size == 1:
            return tensor
        return self.dcp_group.all_gather(tensor.contiguous(), dim)

    def _dcp_all_gather_fragments(
        self,
        *tensors: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, ...]:
        if not tensors:
            return ()
        split_sizes = [tensor.shape[-1] for tensor in tensors]
        pcp_group = getattr(self, "pcp_group", None)
        if dim == 1 and pcp_group is not None and pcp_group.world_size > 1:
            # Query heads are replicated across PCP ranks. Gather only
            # distinct TP heads when DCP spans the full TP x PCP domain.
            if self.dcp_size == pcp_group.world_size:
                return tensors
            gathered = self.tp_group.all_gather(torch.cat(tensors, dim=-1).contiguous(), dim=dim)
        else:
            gathered = self._dcp_all_gather(
                torch.cat(tensors, dim=-1),
                dim,
            )
        return torch.split(gathered, split_sizes, dim=-1)

    def _local_decode_query(self, *queries: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, ...]:
        """Select this TP rank's query heads after a DCP query gather."""
        pcp_group = getattr(self, "pcp_group", None)
        if pcp_group is not None and pcp_group.world_size > 1:
            head_rank = self.tp_group.rank_in_group if self.dcp_size > pcp_group.world_size else 0
        else:
            head_rank = self.dcp_rank
        head_start = head_rank * num_heads
        head_end = head_start + num_heads
        return tuple(query[:, head_start:head_end].contiguous() for query in queries)

    def _merge_dcp_attention_output(self, attn_output, softmax_lse, *, defer_combine=False):
        pcp_group = getattr(self, "pcp_group", None)
        if pcp_group is not None and pcp_group.world_size > 1:
            # Match SFA: TP exchanges distinct head shards, then PCP gathers
            # all KV contributions for that TP-local head range.
            tp_size = self.tp_group.world_size if self.dcp_size > pcp_group.world_size else 1
            return torch.ops.vllm.dcp_a2a_fused(
                attn_output,
                softmax_lse,
                tp_size,
                1,
                self.tp_group.unique_name,
                pcp_group.unique_name,
                defer_combine=defer_combine,
            )
        return torch.ops.vllm.dcp_a2a_fused(
            attn_output,
            softmax_lse,
            self.dcp_size,
            1,
            self.dcp_group.unique_name if self.dcp_size > 1 else "",
            defer_combine=defer_combine,
        )


def _process_attn_out_lse(
    attn_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    *,
    dcp_size: int | None = None,
    dcp_device_group=None,
) -> torch.Tensor:
    if dcp_size is None:
        dcp_size = get_decode_context_model_parallel_world_size()
    if dcp_size > 1 and dcp_device_group is None:
        dcp_device_group = get_dcp_group().device_group
    softmax_lse = softmax_lse.to(torch.float32)
    attn_output = attn_output.to(torch.float32)
    # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
    attn_out_lse = torch.cat([attn_output, softmax_lse], dim=-1)
    if dcp_size > 1:
        # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
        attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
        attn_out_lse_all2all = torch.empty_like(attn_out_lse)
        dist.all_to_all_single(
            attn_out_lse_all2all,
            attn_out_lse,
            group=dcp_device_group,
        )
        attn_out_lse = attn_out_lse_all2all.permute([2, 0, 1])

    return attn_out_lse


def _npu_attention_update(
    head_size,
    attn_out_lse: torch.Tensor,
    *,
    dcp_size: int | None = None,
) -> torch.Tensor:
    if dcp_size is None:
        dcp_size = get_decode_context_model_parallel_world_size()
    # [S, DCP * H, D+1]
    S, H_total, D_plus_1 = attn_out_lse.shape
    H = H_total // dcp_size
    D = head_size
    assert D_plus_1 == D + 1
    # [S, DCP, H, D+1] -> [DCP, S, H, D+1]
    x = attn_out_lse.view(S, dcp_size, H, D_plus_1)
    x = x.permute(1, 0, 2, 3).contiguous()
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


def _npu_attn_out_lse_update(attn_lse_mask, attn_lse_nomask, attn_out_mask, attn_out_nomask):
    T = attn_out_mask.shape[0]
    N = attn_out_mask.shape[1]
    D = attn_out_mask.shape[2]
    attn_out_mask, attn_lse_mask = _out_lse_reshape(attn_out_mask, attn_lse_mask)
    attn_out_nomask, attn_lse_nomask = _out_lse_reshape(attn_out_nomask, attn_lse_nomask)
    attn_out_mask = attn_out_mask.to(torch.float32)
    attn_out_nomask = attn_out_nomask.to(torch.float32)
    attn_lse_mask = attn_lse_mask.to(torch.float32)
    attn_lse_nomask = attn_lse_nomask.to(torch.float32)
    attn_output = [attn_out_nomask, attn_out_mask]
    attn_lse = [attn_lse_nomask, attn_lse_mask]
    update_type = 0
    output, _ = torch_npu.npu_attention_update(attn_lse, attn_output, update_type)
    output = output.view(T, N, D)
    return output


def _out_lse_reshape(attn_out: torch.Tensor, attn_lse: torch.Tensor) -> torch.Tensor:
    attn_out = attn_out.contiguous().view(attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
    attn_lse = attn_lse.contiguous().view(attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
    return attn_out, attn_lse


def _update_out_and_lse(out_list: torch.Tensor, lse_list: torch.Tensor) -> torch.Tensor:
    """LSE_final = log(sum(exp(LSE_i))), O_final = sum(exp(LSE_i - LSE_final) * O_i)
    Args:
        out_list: shape = [N, batch_size, num_heads, head_size]
        lse_list: shape = [N, batch_size, num_heads, 1]
    Returns:
        out_final: shape = [batch_size, num_heads, head_size]
        lse_final: shape = [batch_size, num_heads, 1]
    """
    lse_final = torch.logsumexp(lse_list, dim=0, keepdim=False)
    out_final = torch.sum(torch.exp(lse_list - lse_final) * out_list, dim=0)
    return out_final, lse_final
