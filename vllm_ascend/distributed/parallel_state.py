from typing import Optional

import torch
from vllm.config import ParallelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (GroupCoordinator, get_dp_group,
                                             get_pp_group, get_tp_group,
                                             get_world_group,
                                             init_model_parallel_group)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import enable_sp, flashcomm2_enable

# Currently, mc2 op need their own group coordinator.
_MC2: Optional[GroupCoordinator] = None
_MLP_TP: Optional[GroupCoordinator] = None
_OTP: Optional[GroupCoordinator] = None
_LMTP: Optional[GroupCoordinator] = None
_P_TP: Optional[GroupCoordinator] = None
_FLASHCOMM2_OTP: Optional[GroupCoordinator] = None
_FLASHCOMM2_ODP: Optional[GroupCoordinator] = None
_SHARED_WEIGHT: Optional[GroupCoordinator] = None


def get_mc2_group() -> GroupCoordinator:
    assert _MC2 is not None, ("mc2 group is not initialized")
    return _MC2


def get_otp_group() -> GroupCoordinator:
    assert _OTP is not None, (
        "output tensor parallel group is not initialized")
    return _OTP


def get_lmhead_tp_group() -> GroupCoordinator:
    assert _LMTP is not None, (
        "lm head tensor parallel group is not initialized")
    return _LMTP


def get_flashcomm2_otp_group() -> GroupCoordinator:
    return _FLASHCOMM2_OTP


def get_flashcomm2_odp_group() -> GroupCoordinator:
    assert _FLASHCOMM2_ODP is not None, (
        "output data parallel group for flashcomm2 is not initialized")
    return _FLASHCOMM2_ODP


def get_shared_weight_group() -> GroupCoordinator:
    assert _SHARED_WEIGHT is not None, (
        "output shared weight parallel group for flashcomm2 is not initialized"
    )
    return _SHARED_WEIGHT


def get_mlp_tp_group() -> GroupCoordinator:
    assert _MLP_TP is not None, ("mlp group is not initialized")
    return _MLP_TP


def get_p_tp_group() -> GroupCoordinator:
    assert _P_TP is not None, (
        "distributed prefill tensor parallel group is not initialized")
    return _P_TP


def model_parallel_initialized():
    return (_MC2 is not None)


def init_ascend_model_parallel(parallel_config: ParallelConfig, ):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(
        -1, parallel_config.data_parallel_size *
        parallel_config.prefill_context_parallel_size *
        parallel_config.tensor_parallel_size)

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
        remote_tp_size = parallel_config.tensor_parallel_size // pd_tp_ratio
        if num_head_replica <= 1:
            group_ranks = all_ranks.view(
                -1, prefill_tensor_model_parallel_size).unbind(0)
        else:
            group_ranks = all_ranks.clone().view(
                parallel_config.data_parallel_size, -1,
                num_head_replica)  # [DP_size, num_head, num_head_replica]
            group_ranks = group_ranks.permute(0, 2, 1)
            group_ranks = group_ranks.reshape(
                -1,
                group_ranks.size(-1))  # [DP_size * num_head_replica, num_head]
            alltoall_group_size = group_ranks.size(-1) // remote_tp_size
            group_ranks = group_ranks.unsqueeze(-1).view(
                parallel_config.data_parallel_size, num_head_replica, -1,
                alltoall_group_size
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
    if envs_ascend.VLLM_ASCEND_ENABLE_MLP_OPTIMIZE:
        global _MLP_TP
        assert _MLP_TP is None, (
            "mlp tensor model parallel group is already initialized")

        mlp_tp = parallel_config.data_parallel_size

        all_ranks_mlp_head = torch.arange(world_size).reshape(
            -1, mlp_tp, parallel_config.pipeline_parallel_size, 1)  # noqa
        group_ranks = all_ranks_mlp_head.view(-1, mlp_tp).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        # message queue broadcaster is only used in tensor model parallel group
        _MLP_TP = init_model_parallel_group(group_ranks,
                                            get_world_group().local_rank,
                                            backend,
                                            group_name="mlp_tp")

    # If oproj tensor parallel size is set, we will create a group for it.
    otp_size = get_ascend_config().oproj_tensor_parallel_size
    if otp_size is not None:
        group_ranks = []
        global _OTP
        num_oproj_tensor_parallel_groups: int = (world_size // otp_size)
        for i in range(num_oproj_tensor_parallel_groups):
            ranks = list(range(i * otp_size, (i + 1) * otp_size))
            group_ranks.append(ranks)
        _OTP = init_model_parallel_group(group_ranks,
                                         get_world_group().local_rank,
                                         backend,
                                         group_name="otp")

    lmhead_tensor_parallel_size = get_ascend_config(
    ).lmhead_tensor_parallel_size
    if lmhead_tensor_parallel_size is not None:
        group_ranks = []
        global _LMTP
        num_lmhead_tensor_parallel_groups: int = (world_size //
                                                  lmhead_tensor_parallel_size)
        for i in range(num_lmhead_tensor_parallel_groups):
            ranks = list(
                range(i * lmhead_tensor_parallel_size,
                      (i + 1) * lmhead_tensor_parallel_size))
            group_ranks.append(ranks)
        _LMTP = init_model_parallel_group(group_ranks,
                                          get_world_group().local_rank,
                                          backend,
                                          group_name="lmheadtp")

    # TODO: Extract and unify the logic across different communication group.
    if flashcomm2_enable():
        flashcomm2_otp_size = get_ascend_config(
        ).flashcomm2_oproj_tensor_parallel_size
        global_tp_size = get_tp_group().world_size
        global_dp_size = get_dp_group().world_size
        global_pp_size = get_pp_group().world_size
        num_fc2_oproj_tensor_parallel_groups: int = (global_tp_size //
                                                     flashcomm2_otp_size)

        global _FLASHCOMM2_OTP
        global _FLASHCOMM2_ODP

        _FLASHCOMM2_OTP = None
        _FLASHCOMM2_ODP = get_tp_group()

        if flashcomm2_otp_size > 1:
            otp_group_ranks = []
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
                        otp_group_ranks.append(ranks)

            _FLASHCOMM2_OTP = init_model_parallel_group(
                otp_group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="flashcomm2_otp")
            _FLASHCOMM2_ODP = init_model_parallel_group(
                odp_group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="flashcomm2_odp")

    vllm_config = get_current_vllm_config()
    # TODO: Check if the model is Deepseek V3.2 with enabled SFA CP and activated shared weights. It will then be normalized within the PCP parameters. -- clrs97
    is_ds_v32 = hasattr(vllm_config.model_config.hf_config, "index_topk")
    if enable_sp() and is_ds_v32:
        global _SHARED_WEIGHT
        group_ranks = [list(range(torch.distributed.get_world_size()))]
        _SHARED_WEIGHT = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="CP_shared_weight")


def get_mlp_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_mlp_tp_group().world_size


def get_mlp_tensor_model_parallel_rank():
    """Return world size for the tensor model parallel group."""
    return get_mlp_tp_group().rank_in_group


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

    global _SHARED_WEIGHT
    if _SHARED_WEIGHT:
        _SHARED_WEIGHT.destroy()
    _SHARED_WEIGHT = None
