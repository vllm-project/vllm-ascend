from enum import Enum
from typing import Any

import torch
from vllm import envs
from vllm.distributed import get_dcp_group, get_pcp_group, get_tp_group
from vllm.logger import logger

import vllm_ascend.ops.triton.dcp.dcp_a2a  # noqa: F401
from vllm_ascend.utils import enable_sfa, model_uses_kpool_indexer


def use_dcp_q_replicate(vllm_config) -> bool:
    """Resolve dense MLA Q replication before creating Q projections."""
    parallel = vllm_config.parallel_config
    requested = (
        envs.VLLM_DCP_Q_REPLICATE
        if envs.is_set("VLLM_DCP_Q_REPLICATE")
        else bool(getattr(parallel, "dcp_q_replicate", False))
    )
    if not requested or parallel.decode_context_parallel_size <= 1 or parallel.prefill_context_parallel_size > 1:
        return False
    if enable_sfa(vllm_config):
        logger.warning_once("Ascend dcp_q_replicate is not supported for SFA; disabling Q replication.")
        return False
    # LoRA wrappers do not implement the replicated-Q projection layout.
    if vllm_config.lora_config is not None:
        logger.warning_once("Ascend dcp_q_replicate is not supported for LoRA adapters; disabling Q replication.")
        return False
    return True


def is_pcp_decode_sharding_enabled(vllm_config) -> bool:
    """Shard PCP decode requests, including speculative target verification.

    The decision uses declared vLLM fields so cloning a draft
    ``ParallelConfig`` with ``replace()`` remains supported. KPool models
    keep replicated decodes so every PCP rank advances its head-sharded
    Mamba state in the same request order.
    """
    if model_uses_kpool_indexer(getattr(vllm_config, "model_config", None)):
        # TODO: Adapt PCP decode request sharding for GLM-5.3-Flash, including
        # recurrent-state ownership and request ordering across PCP ranks.
        return False
    parallel_config = vllm_config.parallel_config
    return parallel_config.prefill_context_parallel_size > 1 and parallel_config.decode_context_parallel_size == 1


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
