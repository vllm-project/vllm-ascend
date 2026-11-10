from vllm import envs
from vllm.logger import logger

from vllm_ascend.attention.context_parallel.metadata import (
    CPKVScope,
    DCPImplMixin,
    DCPMetadataBuilderMixin,
    _npu_attention_update,
    _npu_attn_out_lse_update,
    _out_lse_reshape,
    _process_attn_out_lse,
    _update_out_and_lse,
    build_pcp_ordered_slot_mapping,
    use_history_current_split_decode,
)
from vllm_ascend.attention.context_parallel.topology import resolve_cp_topology
from vllm_ascend.utils import enable_sfa, model_uses_kpool_indexer

__all__ = [
    "CPKVScope",
    "DCPImplMixin",
    "DCPMetadataBuilderMixin",
    "_npu_attention_update",
    "_npu_attn_out_lse_update",
    "_out_lse_reshape",
    "_process_attn_out_lse",
    "_update_out_and_lse",
    "build_pcp_ordered_slot_mapping",
    "use_history_current_split_decode",
]


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
    topology = resolve_cp_topology(vllm_config)
    return topology.pcp_size > 1 and topology.dcp_size == 1


def get_pcp_num_replicated_tokens(num_decode_tokens: int, is_decode_sharded: bool) -> int:
    """Return the leading rank-local tokens that every PCP rank computes identically.

    Replicated decodes need no KV gather. A sharded decode belongs to one PCP
    rank, so its KV is gathered together with the prefill tokens.
    """
    return 0 if is_decode_sharded else num_decode_tokens
