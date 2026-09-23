from math import gcd

import torch
import vllm.distributed.parallel_state as vllm_parallel_state
from vllm.config import ParallelConfig, get_current_vllm_config, get_current_vllm_config_or_none
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_node_count,
    get_world_group,
    in_the_same_node_as,
    init_model_parallel_group,
)
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config

# Newer vLLM revisions build the node-local Engram DP group themselves (#56512).
# Until the pinned revision has it, Ascend backports the same grouping below and
# this stays None so the group is never created twice.
_UPSTREAM_ENGRAM_DP_GETTER = getattr(vllm_parallel_state, "get_engram_dp_group", None)

# Currently, mc2 op need their own group coordinator.
_MC2: GroupCoordinator | None = None

# Module specific tensor parallel groups
_MLP_TP: GroupCoordinator | None = None
_OTP: GroupCoordinator | None = None
_LMTP: GroupCoordinator | None = None
_EMBED_TP: GroupCoordinator | None = None

_P_TP: GroupCoordinator | None = None

_DYNAMIC_EPLB: GroupCoordinator | None = None
_KVPP: GroupCoordinator | None = None

# Node-local DP replicas that shard or share one Engram table.
_ENGRAM_DP: GroupCoordinator | None = None


class ReplicatedGroup:
    """A no-communication stand-in for a TP group of world_size 1.

    Used to replicate a parameter across ranks (e.g. a disable_tp layer such
    as the DSpark Markov lm_head): every rank holds the full weight, so there
    is nothing to all-reduce / all-gather. Unlike a real ``GroupCoordinator``,
    constructing this does **not** call ``hcclCommInitRootInfoConfig`` — it is
    a pure logical object exposing just the attributes
    ``AscendVocabParallelEmbedding`` reads (``world_size``, ``rank_in_group``)
    plus no-op comm methods for safety.
    """

    world_size = 1
    rank_in_group = 0
    device_group = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return input_

    def destroy(self) -> None:
        pass


# Singleton: identical on every rank, no HCCL comm created.
_REPLICATED = ReplicatedGroup()


def _engram_dp_shard_size(world_size: int, data_parallel_size: int, replica_size: int) -> int:
    """Size of the node-local Engram DP shards.

    Backport of vLLM's split: a shard has to divide the DP group evenly, to
    align with complete replicas and to stay inside one node. A layout that
    cannot guarantee all three gets size 1, i.e. one shard per replica, so no
    shard ever spans a node boundary.
    """
    node_count = get_node_count()
    replicas_per_node, remainder = divmod(world_size, node_count * replica_size)
    if remainder:
        return 1
    shard_size = gcd(data_parallel_size, replicas_per_node)
    if shard_size == 1 or node_count == 1:
        return shard_size
    ranks_per_node = replicas_per_node * replica_size
    for start in range(0, world_size, ranks_per_node):
        same_node = in_the_same_node_as(get_world_group().cpu_group, start)
        node_ranks = [rank for rank, local in enumerate(same_node) if local]
        if node_ranks != list(range(start, start + ranks_per_node)):
            return 1
    return shard_size


def _engram_dp_group_ranks(all_ranks: torch.Tensor, engram_dp_size: int) -> list[list[int]]:
    """The EDP groups: DP runs of one TP/PP/PCP position in shard-size chunks."""
    group_ranks = all_ranks.permute(0, 2, 3, 4, 1).reshape(-1, engram_dp_size).unbind(0)
    return [ranks.tolist() for ranks in group_ranks]


def _engram_dp_size_for(parallel_config: ParallelConfig, world_size: int) -> int:
    """Engram DP shard size for this launch, or 1 when sharding is not safe."""
    vllm_config = get_current_vllm_config_or_none()
    if (
        vllm_config is None
        or getattr(vllm_config, "engram_config", None) is None
        or vllm_config.model_config is None
        or vllm_config.model_config.architecture != "DeepseekV41ForCausalLM"
        or parallel_config.enable_elastic_ep
    ):
        return 1
    return _engram_dp_shard_size(
        world_size,
        parallel_config.data_parallel_size,
        parallel_config.tensor_parallel_size
        * parallel_config.pipeline_parallel_size
        * parallel_config.prefill_context_parallel_size,
    )


def init_ascend_model_parallel(
    parallel_config: ParallelConfig,
):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    global_tp_size = parallel_config.tensor_parallel_size
    global_dp_size = parallel_config.data_parallel_size
    global_pp_size = parallel_config.pipeline_parallel_size
    global_pcp_size = parallel_config.prefill_context_parallel_size

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(
        -1,
        global_dp_size,
        global_pp_size,
        global_pcp_size,
        global_tp_size,
    )

    kvpp_size = get_ascend_config().kvpp_config.size
    global _KVPP
    assert _KVPP is None, "KV layer parallel group is already initialized"
    if kvpp_size > 1:
        # One cache-replica domain per DP replica and PP stage. PCP's
        # prefill gather replicates MLA KV across PCP as well as TP ranks.
        assert kvpp_size == global_pcp_size * global_tp_size
        kvpp_group_ranks = all_ranks.flatten(-2).reshape(-1, kvpp_size).unbind(0)
        _KVPP = init_model_parallel_group(
            [ranks.tolist() for ranks in kvpp_group_ranks],
            get_world_group().local_rank,
            backend,
            group_name="kvpp",
        )

    pd_tp_ratio = get_ascend_config().pd_tp_ratio
    pd_head_ratio = get_ascend_config().pd_head_ratio
    global _P_TP
    assert _P_TP is None, "distributed prefill tensor parallel group is already initialized"
    prefill_tensor_model_parallel_size = pd_tp_ratio
    # divide alltoall groups
    if pd_head_ratio > 1 and get_current_vllm_config().kv_transfer_config.is_kv_producer:
        num_head_replica = get_ascend_config().num_head_replica
        remote_tp_size = global_tp_size // pd_tp_ratio
        if num_head_replica <= 1:
            group_ranks = all_ranks.view(-1, prefill_tensor_model_parallel_size).unbind(0)
        else:
            group_ranks = all_ranks.clone().view(
                global_dp_size * global_pp_size * global_pcp_size, -1, num_head_replica
            )  # [DP_size, num_head, num_head_replica]
            group_ranks = group_ranks.permute(0, 2, 1)
            group_ranks = group_ranks.reshape(-1, group_ranks.size(-1))  # [DP_size * num_head_replica, num_head]
            alltoall_group_size = group_ranks.size(-1) // remote_tp_size
            group_ranks = group_ranks.unsqueeze(-1).view(
                global_dp_size * global_pp_size * global_pcp_size,
                num_head_replica,
                -1,
                alltoall_group_size,
            )  # [DP_size, num_head_replica, num_alltoall_group, alltoall_group_size]
            group_ranks = group_ranks.reshape(-1, alltoall_group_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        local_rank = get_world_group().local_rank
        num = next((i for i, ranks in enumerate(group_ranks) if local_rank in ranks), None)
        _P_TP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=f"p_tp_{num}")

    # EP like group ranks
    group_ranks = (
        all_ranks.transpose(1, 2)
        .reshape(
            -1,
            global_dp_size * global_pcp_size * global_tp_size,
        )
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]

    global _MC2
    _MC2 = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="mc2")

    if get_ascend_config().eplb_config.dynamic_eplb:
        global _DYNAMIC_EPLB
        _DYNAMIC_EPLB = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dynamic_eplb"
        )

    # Initialize fine-grained TP process groups on Ascend for four components:
    # 1. LM Head: output logits projection (`lmhead_tensor_parallel_size`)
    # 2. O Proj: attention output projection (`oproj_tensor_parallel_size`)
    # 3. Embedding: The token embedding table at the input of the model (`embedding_tensor_parallel_size`)
    # 4. MLP: feed-forward network in transformer blocks (`mlp_tensor_parallel_size`)
    _group_cache = {}

    def _create_or_get_group(group_size: int, group_name: str) -> GroupCoordinator:
        if group_size is None:
            return None
        if group_size not in _group_cache:
            rank_grid = torch.arange(world_size).reshape(global_pp_size, global_dp_size, global_tp_size)
            num_chunks = global_dp_size // group_size
            group_ranks = []
            for pp_idx in range(global_pp_size):
                stage_ranks = rank_grid[pp_idx]  # (dp, tp)
                for chunk in range(num_chunks):
                    for tp_idx in range(global_tp_size):
                        group = stage_ranks[chunk * group_size : (chunk + 1) * group_size, tp_idx].tolist()
                        group_ranks.append(group)
            pg = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name=group_name)
            _group_cache[group_size] = pg

        return _group_cache[group_size]

    otp_size = get_ascend_config().finegrained_tp_config.oproj_tensor_parallel_size
    lmhead_tp_size = get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size
    embedding_tp_size = get_ascend_config().finegrained_tp_config.embedding_tensor_parallel_size
    mlp_tp_size = get_ascend_config().finegrained_tp_config.mlp_tensor_parallel_size

    global _OTP, _LMTP, _EMBED_TP, _MLP_TP

    if otp_size > 0:
        _OTP = _create_or_get_group(otp_size, "otp")
    if lmhead_tp_size > 0:
        _LMTP = _create_or_get_group(lmhead_tp_size, "lmheadtp")
    if embedding_tp_size > 0:
        _EMBED_TP = _create_or_get_group(embedding_tp_size, "emtp")
    if mlp_tp_size > 0:
        _MLP_TP = _create_or_get_group(mlp_tp_size, "mlptp")

    # Engram shards one table over TP x node-local EDP, and neither the shards
    # nor their per-step exchange may cross a node, so the group comes from the
    # physical placement instead of the global DP topology.
    global _ENGRAM_DP
    assert _ENGRAM_DP is None, "Engram data parallel group is already initialized"
    if _UPSTREAM_ENGRAM_DP_GETTER is None:
        engram_dp_size = _engram_dp_size_for(parallel_config, world_size)
        if engram_dp_size > 1:
            _ENGRAM_DP = init_model_parallel_group(
                _engram_dp_group_ranks(all_ranks, engram_dp_size),
                get_world_group().local_rank,
                backend,
                group_name="edp",
            )
            logger.info_once("Engram node-local DP group size: %d (TP=%d)", engram_dp_size, global_tp_size)


def model_parallel_initialized():
    return _MC2 is not None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, "mc2 group is not initialized"
    return _MC2


def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, "mlp group is not initialized"
    return _MLP_TP


def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, "output tensor parallel group is not initialized"
    return _OTP


def get_lmhead_tp_group() -> GroupCoordinator:
    assert _LMTP is not None, "lm head tensor parallel group is not initialized"
    return _LMTP


def get_embed_tp_group() -> GroupCoordinator:
    assert _EMBED_TP is not None, "emtp group is not initialized"
    return _EMBED_TP


def get_replicated_group() -> ReplicatedGroup:
    return _REPLICATED


def get_p_tp_group() -> GroupCoordinator:
    assert _P_TP is not None, "distributed prefill tensor parallel group is not initialized"
    return _P_TP


def get_dynamic_eplb_group() -> GroupCoordinator:
    assert _DYNAMIC_EPLB is not None, "Dynamic eplb group is not initialized"
    return _DYNAMIC_EPLB


def get_kvpp_group() -> GroupCoordinator:
    assert _KVPP is not None, "KV layer parallel group is not initialized"
    return _KVPP


def get_engram_dp_group() -> GroupCoordinator | None:
    """The node-local Engram DP group (EDP), when one exists for this launch."""
    if _UPSTREAM_ENGRAM_DP_GETTER is not None:
        return _UPSTREAM_ENGRAM_DP_GETTER()
    return _ENGRAM_DP


def destroy_ascend_model_parallel():
    global _KVPP
    if _KVPP:
        _KVPP.destroy()
    _KVPP = None

    global _MC2
    if _MC2:
        _MC2.destroy()
    _MC2 = None

    global _MLP_TP
    if _MLP_TP:
        _MLP_TP.destroy()
    _MLP_TP = None

    global _LMTP
    if _LMTP:
        _LMTP.destroy()
    _LMTP = None

    global _EMBED_TP
    if _EMBED_TP:
        _EMBED_TP.destroy()
    _EMBED_TP = None

    global _OTP
    if _OTP:
        _OTP.destroy()
    _OTP = None

    global _P_TP
    if _P_TP:
        _P_TP.destroy()
    _P_TP = None

    global _DYNAMIC_EPLB
    if _DYNAMIC_EPLB:
        _DYNAMIC_EPLB.destroy()
    _DYNAMIC_EPLB = None

    global _ENGRAM_DP
    if _ENGRAM_DP:
        _ENGRAM_DP.destroy()
    _ENGRAM_DP = None


def get_global_rank(parallel_config: ParallelConfig | None = None) -> int:
    """Return a globally unique rank for the current worker across all parallel
     dimensions (TP/PP/CP/DP), compatible with both dense and MoE models.

     vLLM does not expose a single ready-to-use cross-DP global rank:
       - For dense models each DP rank is launched as an independent DP=1 engine,
         so ``data_parallel_rank`` is reset to 0 and ``get_world_group()`` only
         spans one replica (``rank_in_group`` is the local rank in the replica).
       - For MoE DP / external_launcher the world group spans all DP ranks, so
         ``rank_in_group`` already encodes the DP offset.

     ``data_parallel_index`` always keeps the true DP rank (it is never reset),
     and ``rank_in_group % replica_size`` yields the local rank within a replica
     in both cases, so the formula below is correct everywhere. It mirrors vLLM's
     own ``data_parallel_rank * world_size + rank`` (see
     vllm/distributed/parallel_state.py).

    Note: DCP (decode context parallel) reuses the TP NPUs and EP overlays
    TP/DP, so neither adds new ranks and they are intentionally excluded from
    ``replica_size``.
    """
    if parallel_config is None:
        parallel_config = get_current_vllm_config().parallel_config
    # Number of NPUs in a single DP replica (TP * PP * prefill-CP).
    replica_size = (
        parallel_config.tensor_parallel_size
        * parallel_config.pipeline_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    rank_in_replica = get_world_group().rank_in_group % replica_size
    return parallel_config.data_parallel_index * replica_size + rank_in_replica
