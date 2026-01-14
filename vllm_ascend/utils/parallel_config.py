#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallelism configuration utilities for vLLM Ascend.

This module provides functionality for:
- Managing tensor parallelism flags (LM head, embedding, O-proj, MLP)
- Managing sequence parallelism (FlashComm, FlashComm2)
- Managing context parallelism
- Data parallelism for shared experts
"""

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch
import torch_npu  # noqa: F401

from vllm_ascend import envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

_DEFAULT_BUFFER_SIZE = 200
_MIN_DP_BUFFER_SIZE = 50


def lmhead_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.lmhead_tensor_parallel_size > 0


def embedding_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.embedding_tensor_parallel_size > 0


def oproj_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.oproj_tensor_parallel_size > 0


def mlp_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.mlp_tensor_parallel_size > 0


def matmul_allreduce_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE


def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    from vllm_ascend.utils.model_utils import is_moe_model
    global _ENABLE_SP
    if _ENABLE_SP is None:
        if vllm_config is None:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
        _ENABLE_SP = (
            vllm_config.compilation_config.pass_config.enable_sp
            or envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
            # Flash comm 1 should be enabled by env VLLM_ASCEND_ENABLE_FLASHCOMM1
            # We retain the env VLLM_ASCEND_ENABLE_FLASHCOMM here for backward compatibility.
            or bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", '0'))))

        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info(
                "shared_expert_dp requires enable_sp = True. has set enable_sp to True"
            )

        if not _ENABLE_SP:
            return _ENABLE_SP

        assert vllm_config.parallel_config.tensor_parallel_size > 1, \
            "Flash Comm v1 (Sequence Parallelism) is only supported when tp_size > 1."

        assert (
            not is_moe_model(vllm_config)
            or vllm_config.parallel_config.enable_expert_parallel
        ), "Flash Comm v1 (Sequence Parallelism) requires enable_expert_parallel=True for MoE models."

    return _ENABLE_SP


# TODO remove it after vllm has this func
def shared_expert_dp_enabled() -> bool:
    return get_ascend_config().enable_shared_expert_dp or enable_sp()


def prefill_context_parallel_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL


def flashcomm2_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE > 0


def o_shard_enable() -> bool:
    layer_sharding = get_ascend_config().layer_sharding
    if layer_sharding is None:
        return False
    return "o_proj" in layer_sharding


def get_flashcomm2_config_and_validate(ascend_config, vllm_config):
    flashcomm2_oproj_tp_size = envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
    global_tp_size = vllm_config.parallel_config.tensor_parallel_size

    if not flashcomm2_enable():
        return 0

    logger.info(
        f"Enable FLASHCOMM2 with flashcomm2_oproj_tensor_parallel_size = {flashcomm2_oproj_tp_size}"
    )

    layer_sharding = ascend_config.layer_sharding or []
    if layer_sharding:
        if layer_sharding == ["o_proj"]:
            logger.info_once(
                "Enable FLASHCOMM2 with o_proj layer sharding for reduced memory consumption."
            )
        else:
            raise ValueError(
                "FLASHCOMM2 only supports 'o_proj' as the sole layer sharding configuration! "
                f"Found invalid layer_sharding: {layer_sharding}")
    if not envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1:
        logger.warning_once(
            "It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance."
        )
    if ascend_config.finegrained_tp_config.oproj_tensor_parallel_size > 0:
        raise AssertionError(
            "flashcomm2_oproj_tensor_parallel_size cannot be enabled simultaneously with oproj_tensor_parallel_size"
        )
    if global_tp_size <= flashcomm2_oproj_tp_size:
        raise AssertionError(
            f"flashcomm2_oproj_tensor_parallel_size ({flashcomm2_oproj_tp_size}) cannot exceed global tensor parallel size ({global_tp_size})"
        )
    if global_tp_size % flashcomm2_oproj_tp_size != 0:
        raise AssertionError(
            f"Global tensor parallel size ({global_tp_size}) must be divisible by flashcomm2_oproj_tensor_parallel_size ({flashcomm2_oproj_tp_size})"
        )
    if vllm_config.kv_transfer_config is None:
        logger.warning_once(
            "It is recommended to enable FLASHCOMM2 in P-scenario deployments, enable it in hybrid deployment may lead to decode performance degradation."
        )
    if vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_consumer:
        raise AssertionError(
            "FLASHCOMM2 primarily targets P-scenario deployments, with additional support for hybrid deployment scenarios. It is not applicable in D-scenario environments."
        )

    return flashcomm2_oproj_tp_size


def get_flashcomm2_reorgnized_batch_ids(global_tp_size) -> list[list[int]]:
    # Reorganize batch_ids so that, after the all2all and reduce-scatter operation, each batch_id corresponds to the rank_id within the DP domain.
    # For example, when DP = [0, 1, 2, ..., 15] and flashcomm2_oproj_tensor_parallel_size = 2,
    # the reorganized batch_ids will be [[batch0, batch8], [batch1, batch9], ..., [batch7, batch15]].
    flashcomm2_otp_size = get_ascend_config(
    ).flashcomm2_oproj_tensor_parallel_size
    num_oproj_tensor_parallel_groups: int = (global_tp_size //
                                             flashcomm2_otp_size)

    reorgnized_batch_ids = []
    for i in range(num_oproj_tensor_parallel_groups):
        ranks = []
        for j in range(flashcomm2_otp_size):
            rank_idx = i + j * num_oproj_tensor_parallel_groups
            ranks.append(rank_idx)
        reorgnized_batch_ids.append(ranks)

    return reorgnized_batch_ids


def create_hccl_pg_options(group_name: str):
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    from vllm_ascend.utils.communication_utils import get_hccl_config_for_pg_options
    hccl_config = get_hccl_config_for_pg_options(group_name)
    if hccl_config is not None:
        options.hccl_config = hccl_config
    return options


_ENABLE_SP = None


#TODO: Temporarily use enable_sp to enable the dsa_cp feature of ds32. and subsequent updates will introduce new interfaces. --zzhx1
@lru_cache(maxsize=1)
def enable_dsa_cp() -> bool:
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    is_ds_v32 = hasattr(
        vllm_config.model_config, "hf_text_config") and hasattr(
            vllm_config.model_config.hf_text_config, "index_topk")
    if is_ds_v32 and enable_sp():
        return True
    return False


@lru_cache(maxsize=1)
def enable_dsa_cp_with_layer_shard() -> bool:
    if not enable_dsa_cp():
        return False
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    is_prefill_instance = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_producer
    return is_prefill_instance
