# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional

import torch
import torchair as tng
from omni.adaptors.vllm.distributed.parallel_state import (
    GroupCoordinator, get_cross_group_from_list, get_far_cross_group_from_list,
    get_local_group_from_list, get_mlp_tp_group,
    get_near_cross_group_from_list, get_round_cross_group_from_list,
    get_world_group_from_list)
from omni.models.common.config.model_config import model_extra_config
from vllm.distributed import get_tp_group


def reduce_scatter_two_stage(input_: torch.Tensor, idx: int, reverse=False) -> torch.Tensor:
    if model_extra_config.operator_opt_config.two_stage_comm == 0:
        return get_world_group_from_list(idx).reduce_scatter(input_)
    if reverse:
        stage1 = get_cross_group_from_list(idx).reduce_scatter(input_)
        return get_local_group_from_list(idx).reduce_scatter(stage1)
    stage1 = get_local_group_from_list(idx).reduce_scatter(input_)
    return get_cross_group_from_list(idx).reduce_scatter(stage1)


def all_gather_two_stage(input_: torch.Tensor, idx: int, dim=-1, reverse=False) -> torch.Tensor:
    if model_extra_config.operator_opt_config.two_stage_comm == 0:
        return get_world_group_from_list(idx).all_gather(input_, dim)
    if reverse:
        stage1 = get_local_group_from_list(idx).all_gather(input_, dim)
        return get_cross_group_from_list(idx).all_gather(stage1, dim)
    stage1 = get_cross_group_from_list(idx).all_gather(input_, dim)
    return get_local_group_from_list(idx).all_gather(stage1, dim)


def reduce_scatter_local(input_: torch.Tensor, idx: int) -> torch.Tensor:
    return get_local_group_from_list(idx).reduce_scatter(input_)


def reduce_scatter_cross(input_: torch.Tensor, idx: int) -> torch.Tensor:
    return get_cross_group_from_list(idx).reduce_scatter(input_)


def all_gather_local(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_local_group_from_list(idx).all_gather(input_, dim)


def all_gather_cross(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_cross_group_from_list(idx).all_gather(input_, dim)


def mlp_all_gather(input_: torch.Tensor, dim=-1, comm_group: Optional[GroupCoordinator] = None):
    if comm_group is None:
        return get_mlp_tp_group().all_gather(input_, dim)
    else:
        return comm_group.all_gather(input_, dim)


def mlp_reduce_scatter(input_: torch.Tensor, comm_group: Optional[GroupCoordinator] = None) -> torch.Tensor:
    if comm_group is None:
        return get_mlp_tp_group().reduce_scatter(input_)
    else:
        return comm_group.reduce_scatter(input_)


def mla_tensor_model_parallel_reduce_scatter(input_: torch.Tensor, comm_group: Optional[GroupCoordinator] = None) -> torch.Tensor:
    if comm_group is None:
        return get_tp_group().reduce_scatter(input_)
    else:
        return comm_group.reduce_scatter(input_)


def mla_tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1, comm_group: Optional[GroupCoordinator] = None) -> torch.Tensor:
    if comm_group is None:
        return get_tp_group().all_gather(input_, dim)
    else:
        return comm_group.all_gather(input_, dim)


def reduce_scatter_world(input_: torch.Tensor, idx:int) -> torch.Tensor:
    return get_world_group_from_list(idx).reduce_scatter(input_)


def all_gather_world(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_world_group_from_list(idx).all_gather(input_, dim)


def reduce_scatter_pipeline(input_: torch.Tensor, idx: int, which_half: int, dtype=torch.bfloat16, reverse=False) -> torch.Tensor:
    if reverse:
        return reduce_scatter_two_stage(input_, idx, reverse)

    if which_half == 0:
        stage1, stage2 = torch.split(input_, input_.shape[0] // 2, dim=0)
    else:
        stage2, stage1 = torch.split(input_, input_.shape[0] // 2, dim=0)

    if dtype != torch.bfloat16:
        stage2 = stage2.to(torch.bfloat16)
        if model_extra_config.operator_opt_config.moe_multi_stream_tune:
            with tng.scope.npu_stream_switch('67'):
                stage1 = stage1.to(torch.bfloat16)
        else:
            stage1 = stage1.to(torch.bfloat16)

    stage2_local_rs = get_local_group_from_list(idx).reduce_scatter(stage2)
    if model_extra_config.operator_opt_config.moe_multi_stream_tune:
        stage1 = tng.scope.npu_wait_tensor(stage1, stage2_local_rs)
    stage1_near_cross_rs = get_local_group_from_list(idx).reduce_scatter(stage1)
    if model_extra_config.operator_opt_config.moe_multi_stream_tune:
        with tng.scope.npu_stream_switch('77'):
            stage2_far_cross_rs = get_far_cross_group_from_list(idx).swap(stage2_local_rs, method='all2allv')
    else:
        stage2_far_cross_rs = get_far_cross_group_from_list(idx).swap(stage2_local_rs, method='all2allv')

    output = get_near_cross_group_from_list(idx).reduce_scatter(stage1_near_cross_rs + stage2_far_cross_rs)

    return output


def all_gather_pipeline(input_:torch.Tensor, idx: int, which_half: int, dim=-1, reverse=False) -> torch.Tensor:
    if dim != 0 or reverse:
        return all_gather_two_stage(input_, dim, reverse)

    stage1_near_cross = get_near_cross_group_from_list(idx).all_gather(input_, dim)

    stage1_near_cross_ag = get_local_group_from_list(idx).all_gather(stage1_near_cross, dim)
    if model_extra_config.operator_opt_config.moe_multi_stream_tune:
        with tng.scope.npu_stream_switch('37'):
            stage2_far_cross = get_far_cross_group_from_list(idx).swap(stage1_near_cross, method='all2allv')
        stage2_far_cross = tng.scope.npu_wait_tensor(stage2_far_cross, stage1_near_cross_ag)
    else:
        stage2_far_cross = get_far_cross_group_from_list(idx).swap(stage1_near_cross, method='all2allv')

    stage2_far_cross_ag = get_local_group_from_list(idx).all_gather(stage2_far_cross, dim)

    if which_half == 0:
        output = torch.cat([stage1_near_cross_ag, stage2_far_cross_ag], dim=0)
    else:
        output = torch.cat([stage2_far_cross_ag, stage1_near_cross_ag], dim=0)

    return output

def prefill_reduce_scatter_pipeline(input_:torch.Tensor, idx: int, which_half: int) -> torch.Tensor:
    if which_half == 0:
        stage1, stage2 = torch.split(input_, input_.shape[0] // 2, dim=0)
    else:
        stage2, stage1 = torch.split(input_, input_.shape[0] // 2, dim=0)

    stage2_local_rs = get_local_group_from_list(idx).reduce_scatter(stage2)

    rs_stream = torch.npu.Stream()
    curr_stream = torch.npu.current_stream()

    rs_stream.wait_stream(curr_stream)
    with torch.npu.stream(rs_stream):
        stage2_far_cross_rs = get_cross_group_from_list(idx).swap(stage2_local_rs, method="allgather")
    stage1_near_cross_rs = get_local_group_from_list(idx).reduce_scatter(stage1)

    torch.npu.current_stream().wait_stream(rs_stream)
    rs_stream.wait_stream(torch.npu.current_stream())

    output = stage1_near_cross_rs + stage2_far_cross_rs

    return output


def reduce_scatter_round_pipeline(input_: torch.Tensor, idx: int, node_rank: int, dtype=torch.bfloat16, reverse=False) -> torch.Tensor:
    if reverse:
        return reduce_scatter_two_stage(input_, idx, reverse)

    if node_rank == 0:
        input_self, round0, round1, round2 = torch.split(input_, input_.shape[0] // 4, dim=0)
    elif node_rank == 1:
        round0, input_self, round2, round1 = torch.split(input_, input_.shape[0] // 4, dim=0)
    elif node_rank == 2:
        round1, round2, input_self, round0 = torch.split(input_, input_.shape[0] // 4, dim=0)
    elif node_rank == 3:
        round2, round1, round0, input_self = torch.split(input_, input_.shape[0] // 4, dim=0)
    else:
        return None

    if dtype != torch.bfloat16:
        round2 = round2.to(torch.bfloat16)
        if model_extra_config.operator_opt_config.moe_multi_stream_tune:
            with tng.scope.npu_stream_switch('67'):
                round1 = round1.to(torch.bfloat16)
                round0 = round0.to(torch.bfloat16)
                input_self = input_self.to(torch.bfloat16)
        else:
            round1 = round1.to(torch.bfloat16)
            round0 = round0.to(torch.bfloat16)
            input_self = input_self.to(torch.bfloat16)
    if model_extra_config.operator_opt_config.moe_multi_stream_tune:
        round2_rs = get_local_group_from_list(idx).reduce_scatter(round2)
        round1 = tng.scope.npu_wait_tensor(round1, round2_rs)
        round1_rs = get_local_group_from_list(idx).reduce_scatter(round1)
        round0 = tng.scope.npu_wait_tensor(round0, round2_rs)
        round0_rs = get_local_group_from_list(idx).reduce_scatter(round0)
        input_self = tng.scope.npu_wait_tensor(input_self, round0_rs)
        input_rs = get_local_group_from_list(idx).reduce_scatter(input_self)
        with tng.scope.npu_stream_switch('77'):
            round2_swp = get_round_cross_group_from_list(round=2).swap(round2_rs, method="all2allv")
        with tng.scope.npu_stream_switch('87'):
            round1_swp = get_round_cross_group_from_list(round=1).swap(round1_rs, method="all2allv")
        with tng.scope.npu_stream_switch('97'):
            round0_swp = get_round_cross_group_from_list(round=0).swap(round0_rs, method="all2allv")
    else:
        round2_rs = get_local_group_from_list(idx).reduce_scatter(round2)
        round1_rs = get_local_group_from_list(idx).reduce_scatter(round1)
        round0_rs = get_local_group_from_list(idx).reduce_scatter(round0)
        input_rs = get_local_group_from_list(idx).reduce_scatter(input_self)

        round2_swp = get_round_cross_group_from_list(round=2).swap(round2_rs, method="all2allv")
        round1_swp = get_round_cross_group_from_list(round=1).swap(round1_rs, method="all2allv")
        round0_swp = get_round_cross_group_from_list(round=0).swap(round0_rs, method="all2allv")

    output = input_rs + round0_swp + round1_swp + round2_swp

    return output


def all_gather_round_pipeline(input_: torch.Tensor, idx: int, node_rank: int, dim=-1, reverse=False) -> torch.Tensor:
    if dim != 0 or reverse:
        return all_gather_two_stage(input_, dim, reverse)

    input_ag = get_local_group_from_list(idx).all_gather(input_, dim)
    if model_extra_config.operator_opt_config.moe_multi_stream_tune:
        with tng.scope.npu_stream_switch('37'):
            round0_swp = get_round_cross_group_from_list(round=0).swap(input_, method="all2allv")
        with tng.scope.npu_stream_switch('47'):
            round1_swp = get_round_cross_group_from_list(round=1).swap(input_, method="all2allv")
        with tng.scope.npu_stream_switch('57'):
            round2_swp = get_round_cross_group_from_list(round=2).swap(input_, method="all2allv")

        round0_swp = tng.scope.npu_wait_tensor(round0_swp, input_ag)
        round0_ag = get_local_group_from_list(idx).all_gather(round0_swp, dim)
        round1_swp = tng.scope.npu_wait_tensor(round1_swp, round0_ag)
        round1_ag = get_local_group_from_list(idx).all_gather(round1_swp, dim)
        round2_swp = tng.scope.npu_wait_tensor(round2_swp, round1_ag)
        round2_ag = get_local_group_from_list(idx).all_gather(round2_swp, dim)
    else:
        round0_swp = get_round_cross_group_from_list(round=0).swap(input_, method="all2allv")
        round1_swp = get_round_cross_group_from_list(round=1).swap(input_, method="all2allv")
        round2_swp = get_round_cross_group_from_list(round=2).swap(input_, method="all2allv")
        round0_ag = get_local_group_from_list(idx).all_gather(round0_swp, dim)
        round1_ag = get_local_group_from_list(idx).all_gather(round1_swp, dim)
        round2_ag = get_local_group_from_list(idx).all_gather(round2_swp, dim)

    if node_rank == 0:
        output = torch.cat([input_ag, round0_ag, round1_ag, round2_ag], dim=0)
    elif node_rank == 1:
        output = torch.cat([round0_ag, input_ag, round2_ag, round1_ag], dim=0)
    elif node_rank == 2:
        output = torch.cat([round1_ag, round2_ag, input_ag, round0_ag], dim=0)
    elif node_rank == 3:
        output = torch.cat([round2_ag, round1_ag, round0_ag, input_ag], dim=0)
    else:
        output = None

    return output


def all_to_all_local(input_: torch.Tensor, idx: int) -> torch.Tensor:
    return get_local_group_from_list(idx).all_to_all(input_)
