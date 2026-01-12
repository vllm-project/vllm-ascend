from typing import Optional

import torch
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (GroupCoordinator, get_tp_group,
                                             get_world_group,
                                             init_model_parallel_group)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import enable_dsa_cp, flashcomm2_enable

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None

# Module specific tensor parallel groups
_MLP_TP: Optional[GroupCoordinator] = None
_OTP: Optional[GroupCoordinator] = None
_LMTP: Optional[GroupCoordinator] = None
_EMBED_TP: Optional[GroupCoordinator] = None

# flashcomm specific groups
_FLASHCOMM2_OTP: Optional[GroupCoordinator] = None
_FLASHCOMM2_ODP: Optional[GroupCoordinator] = None
_FC3_QUANT_X: Optional[GroupCoordinator] = None

# shard_weight across rank groups
_SHARD_WEIGHT: Optional[GroupCoordinator] = None

_P_TP: Optional[GroupCoordinator] = None


def init_ascend_model_parallel(parallel_config: ParallelConfig, ):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    global_tp_size = parallel_config.tensor_parallel_size
    global_dp_size = parallel_config.data_parallel_size
    global_pp_size = parallel_config.pipeline_parallel_size

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(
        -1, global_dp_size * parallel_config.prefill_context_parallel_size *
        global_tp_size)
    #TODO: all_ranks should be the same as vllm_all_ranks, all_ranks needs to be removed in the future.
    vllm_all_ranks = torch.arange(world_size).reshape(
        -1,
        global_dp_size,
        global_pp_size,
        parallel_config.prefill_context_parallel_size,
        global_tp_size,
    )

    pd_tp_ratio = get_ascend_config().pd_tp_ratio
    pd_head_ratio = get_ascend_config().pd_head_ratio
    global _P_TP
    assert _P_TP is None, (
        "distributed prefill tensor parallel group is already initialized")
    prefill_tensor_model_parallel_size = pd_tp_ratio
    # divide alltoall groups
    if pd_head_ratio > 1 and get_current_vllm_config(
    ).kv_transfer_config.is_kv_producer:
        num_head_replica = get_ascend_config().num_head_replica
        remote_tp_size = global_tp_size // pd_tp_ratio
        if num_head_replica <= 1:
            group_ranks = all_ranks.view(
                -1, prefill_tensor_model_parallel_size).unbind(0)
        else:
            group_ranks = all_ranks.clone().view(
                global_dp_size, -1,
                num_head_replica)  # [DP_size, num_head, num_head_replica]
            group_ranks = group_ranks.permute(0, 2, 1)
            group_ranks = group_ranks.reshape(
                -1,
                group_ranks.size(-1))  # [DP_size * num_head_replica, num_head]
            alltoall_group_size = group_ranks.size(-1) // remote_tp_size
            group_ranks = group_ranks.unsqueeze(-1).view(
                global_dp_size, num_head_replica, -1, alltoall_group_size
            )  # [DP_size, num_head_replica, num_alltoall_group, alltoall_group_size]
            group_ranks = group_ranks.reshape(-1,
                                              alltoall_group_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        local_rank = get_world_group().local_rank
        num = next(
            (i for i, ranks in enumerate(group_ranks) if local_rank in ranks),
            None)
        _P_TP = init_model_parallel_group(group_ranks,
                                          get_world_group().local_rank,
                                          backend,
                                          group_name=f"p_tp_{num}")

    global _MC2
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    _MC2 = init_model_parallel_group(group_ranks,
                                     get_world_group().local_rank,
                                     backend,
                                     group_name="mc2")

    # Initialize fine-grained TP process groups on Ascend for four components:
    # 1. LM Head: output logits projection (`lmhead_tensor_parallel_size`)
    # 2. O Proj: attention output projection (`oproj_tensor_parallel_size`)
    # 3. Embedding: The token embedding table at the input of the model (`embedding_tensor_parallel_size`)
    # 4. MLP: feed-forward network in transformer blocks (`mlp_tensor_parallel_size`)
    _group_cache = {}

    def _create_or_get_group(group_size: int,
                             group_name: str) -> GroupCoordinator:
        if group_size is None:
            return None
        if group_size not in _group_cache:

            rank_grid = torch.arange(world_size).reshape(
                global_pp_size, global_dp_size, global_tp_size)
            num_chunks = global_dp_size // group_size
            group_ranks = []
            for pp_idx in range(global_pp_size):
                stage_ranks = rank_grid[pp_idx]  # (dp, tp)
                for chunk in range(num_chunks):
                    for tp_idx in range(global_tp_size):
                        group = stage_ranks[chunk * group_size:(chunk + 1) *
                                            group_size, tp_idx].tolist()
                        group_ranks.append(group)
            pg = init_model_parallel_group(group_ranks,
                                           get_world_group().local_rank,
                                           backend,
                                           group_name=group_name)
            _group_cache[group_size] = pg

        return _group_cache[group_size]

    otp_size = get_ascend_config(
    ).finegrained_tp_config.oproj_tensor_parallel_size
    lmhead_tp_size = get_ascend_config(
    ).finegrained_tp_config.lmhead_tensor_parallel_size
    embedding_tp_size = get_ascend_config(
    ).finegrained_tp_config.embedding_tensor_parallel_size
    mlp_tp_size = get_ascend_config(
    ).finegrained_tp_config.mlp_tensor_parallel_size

    global _OTP, _LMTP, _EMBED_TP, _MLP_TP

    if otp_size > 0:
        _OTP = _create_or_get_group(otp_size, "otp")
    if lmhead_tp_size > 0:
        _LMTP = _create_or_get_group(lmhead_tp_size, "lmheadtp")
    if embedding_tp_size > 0:
        _EMBED_TP = _create_or_get_group(embedding_tp_size, "emtp")
    if mlp_tp_size > 0:
        _MLP_TP = _create_or_get_group(mlp_tp_size, "mlptp")

    # TODO: Extract and unify the logic across different communication group.
    flashcomm2_otp_group_ranks = []
    if flashcomm2_enable():
        flashcomm2_otp_size = get_ascend_config(
        ).flashcomm2_oproj_tensor_parallel_size
        num_fc2_oproj_tensor_parallel_groups: int = (global_tp_size //
                                                     flashcomm2_otp_size)
        global _FLASHCOMM2_OTP
        global _FLASHCOMM2_ODP

        _FLASHCOMM2_OTP = None
        _FLASHCOMM2_ODP = get_tp_group()

        if flashcomm2_otp_size > 1:
            odp_group_ranks: list[list[int]] = [
                [] for _ in range(flashcomm2_otp_size * global_dp_size *
                                  global_pp_size)
            ]
            for dp_group_index in range(global_dp_size):
                for pp_group_index in range(global_pp_size):
                    dp_pp_serial_index = dp_group_index * global_pp_size + pp_group_index
                    tp_base_rank = dp_pp_serial_index * global_tp_size
                    odp_base_index = dp_pp_serial_index * flashcomm2_otp_size

                    for i in range(num_fc2_oproj_tensor_parallel_groups):
                        ranks = []
                        for j in range(flashcomm2_otp_size):
                            tp_local_rank = i + j * num_fc2_oproj_tensor_parallel_groups
                            assert tp_local_rank < global_tp_size
                            global_rank = tp_base_rank + tp_local_rank
                            ranks.append(global_rank)

                            odp_group_index = odp_base_index + j
                            odp_group_ranks[odp_group_index].append(
                                global_rank)
                        flashcomm2_otp_group_ranks.append(ranks)

            _FLASHCOMM2_OTP = init_model_parallel_group(
                flashcomm2_otp_group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="flashcomm2_otp")
            _FLASHCOMM2_ODP = init_model_parallel_group(
                odp_group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="flashcomm2_odp")

    def create_shard_weight_group(
            module_tp_group_ranks: None) -> GroupCoordinator:
        # Argument module_tp_group_ranks: The module specific tensor parallel group.
        # There are three situations.
        # 1. If it is None, then the TP_size of the specific module is 1 and is replicated linear layer.
        # 2. If it is not None, and the module tp_group is same as the global tp_group.
        # 3. If it is not None, and the module tp_group is different from the global tp_group.(eg. flashcomm2_otp)
        group_ranks = []
        pp_group_ranks = vllm_all_ranks.transpose(2, 4).reshape(
            -1, global_pp_size)
        if module_tp_group_ranks is None:
            # If it is None, then the TP_size of this shard weight is 1.
            shard_weight_group_ranks = pp_group_ranks.transpose(0, 1).unbind(0)
            group_ranks = [x.tolist() for x in shard_weight_group_ranks]
        else:
            # combine standard tp group and non-standard tp group to build  shard_weight comm_group
            module_tp_tanspose_ranks = module_tp_group_ranks.transpose(0, 1)
            G = world_size // (global_pp_size * module_tp_group_ranks.size(1))
            shard_weight_group_ranks = torch.stack(
                [t.view(global_pp_size, G) for t in module_tp_tanspose_ranks],
                dim=1)
            group_ranks = shard_weight_group_ranks.view(-1, G).tolist()
        return init_model_parallel_group(group_ranks,
                                         get_world_group().local_rank,
                                         backend,
                                         group_name="shard_weight")

    # Create shard weight group if enabled
    if get_ascend_config().layer_sharding is not None:
        global _SHARD_WEIGHT
        if flashcomm2_enable():
            if len(flashcomm2_otp_group_ranks) == 0:
                FC2_group_ranks = None
            else:
                FC2_group_ranks = torch.tensor(
                    flashcomm2_otp_group_ranks).squeeze(0)
            _SHARD_WEIGHT = create_shard_weight_group(FC2_group_ranks)
        elif enable_dsa_cp():
            # For dsa_cp, all shard layers are replicated.
            _SHARD_WEIGHT = create_shard_weight_group(None)
        else:
            # For standard tp, use global tp group_ranks
            tp_group_ranks = vllm_all_ranks.view(-1, global_tp_size)
            _SHARD_WEIGHT = create_shard_weight_group(tp_group_ranks)

    if get_ascend_config().multistream_overlap_gate:
        global _FC3_QUANT_X
        group_ranks = all_ranks.unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _FC3_QUANT_X = init_model_parallel_group(group_ranks,
                                                 get_world_group().local_rank,
                                                 backend,
                                                 group_name="fc3_quant_x")


def model_parallel_initialized():
    return (_MC2 is not None)


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, ("mlp group is not initialized")
    return _MLP_TP


def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, (
        "output tensor parallel group is not initialized")
    return _OTP


def get_lmhead_tp_group() -> GroupCoordinator:
    assert _LMTP is not None, (
        "lm head tensor parallel group is not initialized")
    return _LMTP


def get_embed_tp_group() -> GroupCoordinator:
    assert _EMBED_TP is not None, ("emtp group is not initialized")
    return _EMBED_TP


def get_flashcomm2_otp_group() -> GroupCoordinator:
    return _FLASHCOMM2_OTP


def get_flashcomm2_odp_group() -> GroupCoordinator:
    assert _FLASHCOMM2_ODP is not None, (
        "output data parallel group for flashcomm2 is not initialized")
    return _FLASHCOMM2_ODP


def get_shard_weight_group() -> GroupCoordinator:
    assert _SHARD_WEIGHT is not None, (
        "output shard weight parallel group for flashcomm2 is not initialized")
    return _SHARD_WEIGHT


def get_p_tp_group() -> GroupCoordinator:
    assert _P_TP is not None, (
        "distributed prefill tensor parallel group is not initialized")
    return _P_TP


def get_fc3_quant_x_group() -> GroupCoordinator:
    assert _FC3_QUANT_X is not None, ("fc3 quant x group is not initialized")
    return _FC3_QUANT_X


def destroy_ascend_model_parallel():
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

    global _FLASHCOMM2_OTP
    if _FLASHCOMM2_OTP and get_ascend_config(
    ).flashcomm2_oproj_tensor_parallel_size != 1:
        _FLASHCOMM2_OTP.destroy()
        _FLASHCOMM2_OTP = None

    global _FLASHCOMM2_ODP
    if _FLASHCOMM2_ODP and get_ascend_config(
    ).flashcomm2_oproj_tensor_parallel_size != 1:
        _FLASHCOMM2_ODP.destroy()
        _FLASHCOMM2_ODP = None

    global _SHARD_WEIGHT
    if _SHARD_WEIGHT:
        _SHARD_WEIGHT.destroy()
    _SHARD_WEIGHT = None

    global _FC3_QUANT_X
    if _FC3_QUANT_X:
        _FC3_QUANT_X.destroy()
    _FC3_QUANT_X = None
