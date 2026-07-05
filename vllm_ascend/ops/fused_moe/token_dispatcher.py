# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from abc import ABC, abstractmethod
from typing import Generic

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all, gather_from_sequence_parallel_region
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEAllToAllCombineMetadata,
    MoEMC2CombineMetadata,
    MoEQuantParams,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    TMoECombineMetadata,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_custom_op,
    get_ascend_device_type,
    is_hierarchical_communication_enabled,
    should_skip_allreduce_across_dp_group,
)

EXPERT_TOKEN_NUMS_TYPE_CUMSUM = 0
EXPERT_TOKEN_NUMS_TYPE_COUNT = 1


def _get_expert_token_nums_type(token_dispatch_input: MoETokenDispatchInput) -> int:
    # grouped_matmul_swiglu_quant_v2 consumes per-expert counts; existing
    # MC2 grouped-matmul paths consume prefix sums.
    if token_dispatch_input.quant.use_w4a8_per_channel_gmm_swiglu:
        return EXPERT_TOKEN_NUMS_TYPE_COUNT
    return EXPERT_TOKEN_NUMS_TYPE_CUMSUM


def _num_tokens_per_tp_rank(vllm_config) -> int:
    scheduler_config = vllm_config.scheduler_config
    compilation_config = vllm_config.compilation_config
    speculative_config = vllm_config.speculative_config
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    uniform_decode_query_len = 1 if not speculative_config else 1 + speculative_config.num_speculative_tokens
    decode_max_num_seqs = getattr(scheduler_config, "decode_max_num_seqs", 0)
    max_num_reqs = max(scheduler_config.max_num_seqs, decode_max_num_seqs)
    if compilation_config.cudagraph_capture_sizes:
        max_num_tokens = compilation_config.max_cudagraph_capture_size
    else:
        max_num_tokens = min(max_num_reqs * uniform_decode_query_len, 512)
    return (max_num_tokens + tp_size - 1) // tp_size


class MoETokenDispatcher(ABC, Generic[TMoECombineMetadata]):
    def __init__(self, **kwargs) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_rank(self):
        return get_ep_group().rank_in_group

    @property
    def ep_size(self):
        return get_ep_group().world_size

    @abstractmethod
    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[TMoECombineMetadata]:
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: TMoECombineMetadata,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Combine function not implemented.")


class TokenDispatcherWithMC2(MoETokenDispatcher[MoEMC2CombineMetadata]):
    def __init__(self, moe_config=None, **kwargs):
        super().__init__(**kwargs)
        device_group = get_mc2_group().device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        self.ep_rank_id = get_mc2_group().rank_in_group
        self.ep_world_size = get_mc2_group().world_size
        vllm_config = get_current_vllm_config()
        self.enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        self.need_extra_args = get_ascend_device_type() in [AscendDeviceType.A3, AscendDeviceType.A5]
        self.a5_need_extra_args = get_ascend_device_type() == AscendDeviceType.A5
        # NOTE: When in A2, setting the environment variables HCCL_INTRA_PCIE_ENABLE=1 and
        # HCCL_INTRA_ROCE_ENABLE=0 can reduce cross-machine communication traffic and significantly
        # improve communication performance.
        # When enable hierarchical communication, param `expert_scales` need to be passed in.
        self.need_expert_scale = is_hierarchical_communication_enabled()

        # Here we need to calculate the global_bs = max_bs_per_rank * ep_world_size to execute
        # dispatch & combine operators with different input num_tokens per rank.
        num_tokens_per_tp_rank = _num_tokens_per_tp_rank(vllm_config)
        _max_global_bs = num_tokens_per_tp_rank * self.ep_world_size

        # When allreduce across DP is not skipped, tokens are uniform across ranks:
        # use global_bs=0 (uniform mode) and pass mc2_mask.
        # When allreduce is skipped, tokens may differ per rank:
        # use the real global_bs and do NOT pass mc2_mask.
        self.global_bs = _max_global_bs if should_skip_allreduce_across_dp_group(vllm_config) else 0

        # NOTE: When enable_mc2_hierarchy_comm is true, we need pass in `comm_alg` to mc2 op.
        self.need_comm_alg = get_ascend_config().enable_mc2_hierarchy_comm

        if not self.enable_dispatch_v2 and self.need_comm_alg:
            raise RuntimeError(
                "PTA and CANN version is too old to support mc2 hierarchy comm, please upgrade your version."
            )

        self.moe_expert_num = 0

    def refresh_hccl_group(self) -> None:
        """Refresh MC2 communicator metadata after HCCL groups are recreated."""
        device_group = get_mc2_group().device_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def get_dispatch_mc2_kwargs(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        comm_quant_mode = token_dispatch_input.quant.comm_quant_mode

        assert expert_map is not None, "expert_map is required for MC2 token dispatch."
        # NOTE: quant_mode differs by quant feature:
        # - Legacy int communication quantization uses quant_mode=2.
        # - A5 MXFP communication uses quant_mode=4.
        if comm_quant_mode is not None:
            quant_mode = comm_quant_mode
        elif token_dispatch_input.quant.dispatch_with_quant:
            quant_mode = 4 if self.a5_need_extra_args and token_dispatch_input.quant.is_mxfp else 2
        else:
            quant_mode = 0
        self.moe_expert_num = len(expert_map) + global_redundant_expert_num
        expert_token_nums_type = _get_expert_token_nums_type(token_dispatch_input)
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
            "expert_token_nums_type": expert_token_nums_type,
        }
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = token_dispatch_input.routing.mc2_mask

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.need_extra_args:
            stage1_kwargs.update(
                {
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        # Only dispatch-enabled MXFP paths pass y_dtype through MC2.
        if (
            self.a5_need_extra_args
            and (token_dispatch_input.quant.is_mxfp or token_dispatch_input.quant.is_fp8)
            and token_dispatch_input.quant.dispatch_with_quant
        ):
            y_dtype = torch.float8_e4m3fn
            if (
                token_dispatch_input.quant.mxfp is not None
                and token_dispatch_input.quant.mxfp.act_quant_type is not None
            ):
                y_dtype = token_dispatch_input.quant.mxfp.act_quant_type
            stage1_kwargs.update({"tp_world_size": 1, "tp_rank_id": 0, "y_dtype": y_dtype})
        if self.need_expert_scale or self.a5_need_extra_args:
            stage1_kwargs.update(
                {
                    "expert_scales": topk_weights.to(torch.float32),
                }
            )
        if self.need_comm_alg:
            stage1_kwargs.update({"comm_alg": "hierarchy"})

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        kwargs_mc2 = self.get_dispatch_mc2_kwargs(token_dispatch_input)
        output = (
            torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)
            if self.enable_dispatch_v2
            else torch_npu.npu_moe_distribute_dispatch(**kwargs_mc2)
        )
        # comm_stream.wait_stream(torch.npu.current_stream())
        (
            expand_x,
            dynamic_scale,
            assist_info_for_combine,
            expert_token_nums,
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = output[0:7]

        group_list_type = kwargs_mc2["expert_token_nums_type"]
        return MoETokenDispatchOutput(
            hidden_states=expand_x,
            dynamic_scale=dynamic_scale,
            group_list=expert_token_nums,
            group_list_type=group_list_type,
            combine_metadata=MoEMC2CombineMetadata(
                topk_ids=token_dispatch_input.topk_ids,
                topk_weights=token_dispatch_input.topk_weights,
                expert_map=token_dispatch_input.routing.expert_map,
                ep_recv_counts=ep_recv_counts,
                tp_recv_counts=tp_recv_counts,
                assist_info_for_combine=assist_info_for_combine,
                expand_scales=expand_scales,
                quant=token_dispatch_input.quant,
                mc2_mask=token_dispatch_input.routing.mc2_mask if self.global_bs == 0 else None,
            ),
        )

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor, combine_metadata: MoEMC2CombineMetadata):
        expert_map = combine_metadata.expert_map
        topk_ids = combine_metadata.topk_ids
        topk_weights = combine_metadata.topk_weights
        ep_recv_counts = combine_metadata.ep_recv_counts
        tp_recv_counts = combine_metadata.tp_recv_counts
        assist_info_for_combine = combine_metadata.assist_info_for_combine
        expand_scales = combine_metadata.expand_scales
        quant_type = combine_metadata.quant.quant_type
        comm_quant_mode = combine_metadata.quant.comm_quant_mode

        assert expert_map is not None
        # NOTE: quant_mode differs by quant features:
        # - A5 MXFP communication uses quant_mode=4 only for MXFP8 currently.
        if comm_quant_mode is not None:
            quant_mode = comm_quant_mode
        elif quant_type == QuantType.MXFP8:
            quant_mode = 4
        else:
            quant_mode = 0
        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
        }
        if self.global_bs == 0:
            kwargs_mc2["x_active_mask"] = combine_metadata.mc2_mask

        if combine_metadata.quant.dispatch_with_quant:
            tp_recv_counts = torch.empty(1, dtype=torch.int32, device=hidden_states.device)

        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
            "expand_scales": expand_scales,
            "comm_quant_mode": quant_mode,
        }

        if self.enable_dispatch_v2:
            stage3_kwargs["assist_info_for_combine"] = assist_info_for_combine
        else:
            stage3_kwargs["expand_idx"] = assist_info_for_combine

        if self.need_extra_args:
            stage3_kwargs.update(
                {
                    "tp_send_counts": tp_recv_counts,
                    "group_tp": self.moe_all_to_all_group_name,
                    "tp_world_size": 1,
                    "tp_rank_id": 0,
                }
            )
        if self.need_comm_alg:
            stage3_kwargs.update({"comm_alg": "hierarchy"})

        kwargs_mc2.update(stage3_kwargs)
        return kwargs_mc2

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states, combine_metadata)
        combined_output = (
            torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2)
            if self.enable_dispatch_v2
            else torch_npu.npu_moe_distribute_combine(**kwargs_mc2)
        )

        return combined_output


class TokenDispatcherWithZB(TokenDispatcherWithMC2):
    """MC2 token dispatcher with zero-buffer SHMEM dispatch/combine."""

    def __init__(self, moe_config=None, **kwargs):
        super().__init__(moe_config=moe_config, **kwargs)
        self._moe_config = moe_config
        vllm_config = get_current_vllm_config()
        self._zb_max_tokens_per_rank = _num_tokens_per_tp_rank(vllm_config)
        self._validate_zb_compat()
        self._ensure_zb_runtime_init()
        # SHMEM process runtime is process-wide; per-dispatcher tensor/aux buffers
        # are populated lazily on first token_dispatch.
        self._zb_bundle = None
        self._zb_aux = None
        self._zb_hidden = None
        self._zb_max_recv_tokens = None
        self._zb_moe_expert_num = None
        self._zb_use_quant = None
        self._zb_dispatch_quant_mode = None

    def _validate_zb_compat(self) -> None:
        if not self.need_extra_args:
            raise RuntimeError(
                "additional_config.enable_mc2_zb=true requires A3/A5 hardware; current "
                f"device type does not match (need_extra_args={self.need_extra_args})."
            )
        if self.need_comm_alg:
            raise RuntimeError(
                "additional_config.enable_mc2_zb=true is incompatible with enable_mc2_hierarchy_comm; "
                "disable one of them."
            )
        from vllm_ascend.ops.fused_moe.zb_runtime import validate_zb_serving_parallel_config

        validate_zb_serving_parallel_config(get_current_vllm_config().parallel_config)
        enable_custom_op()
        ascend_ops = getattr(torch.ops, "_C_ascend", None)
        if ascend_ops is None or not hasattr(ascend_ops, "zb_moe_distribute_dispatch"):
            raise RuntimeError(
                "additional_config.enable_mc2_zb=true but zero-buffer ops are not registered. "
                "Install Ascend SHMEM at /usr/local/Ascend/shmem/latest and rebuild vllm_ascend."
            )

    def _resolve_zb_moe_expert_num(self) -> int:
        if self._moe_config is not None:
            moe_expert_num = int(getattr(self._moe_config, "num_experts", 0) or 0)
            if moe_expert_num > 0:
                return moe_expert_num
        from vllm_ascend.ops.fused_moe.zb_runtime import resolve_zb_moe_expert_num

        return resolve_zb_moe_expert_num(get_current_vllm_config())

    def _ensure_zb_runtime_init(self) -> None:
        """Early init: aclshmemx_init_attr once per process (aligned with MC2 HCCL init)."""
        from vllm_ascend.ops.fused_moe.zb_runtime import (
            ensure_zb_process_initialized,
            estimate_zb_early_local_mem_size,
            get_zb_process_runtime,
            resolve_zb_experts_per_token,
            resolve_zb_moe_expert_num,
            resolve_zb_shmem_uri,
        )

        existing = get_zb_process_runtime()
        if existing is not None:
            self._zb_runtime = existing
            self._zb_early_local_mem_size = existing.local_mem_size
            return

        vllm_config = get_current_vllm_config()
        hidden_size = int(vllm_config.model_config.get_hidden_size())
        moe_expert_num = self._resolve_zb_moe_expert_num()
        if moe_expert_num <= 0:
            moe_expert_num = resolve_zb_moe_expert_num(vllm_config)
        if moe_expert_num <= 0:
            raise RuntimeError(
                "additional_config.enable_mc2_zb=true but moe_expert_num is unknown at "
                "dispatcher init; pass moe_config with num_experts to TokenDispatcherWithZB."
            )

        experts_per_token = int(getattr(self._moe_config, "experts_per_token", 0) or 0)
        if experts_per_token <= 0:
            experts_per_token = resolve_zb_experts_per_token(vllm_config)
        local_mem_size, _ = estimate_zb_early_local_mem_size(
            max_tokens_per_rank=self._zb_max_tokens_per_rank,
            ep_world_size=self.ep_world_size,
            hidden_size=hidden_size,
            moe_expert_num=moe_expert_num,
            experts_per_token=experts_per_token,
            use_quant=False,
        )
        runtime = ensure_zb_process_initialized(
            rank=self.ep_rank_id,
            world_size=self.ep_world_size,
            local_mem_size=local_mem_size,
            server_ip_port=resolve_zb_shmem_uri(),
        )
        self._zb_runtime = runtime
        self._zb_early_local_mem_size = local_mem_size

    def _ensure_zb_buffers(
        self,
        hidden: int,
        moe_expert_num: int,
        use_quant: bool,
        device: torch.device,
    ) -> None:
        """Late init: ext_info, SHMEM tensors, and aux buffers on first dispatch."""
        if self._zb_bundle is not None:
            if (
                self._zb_hidden != hidden
                or self._zb_moe_expert_num != moe_expert_num
                or self._zb_use_quant != use_quant
            ):
                raise RuntimeError(
                    "ZB SHMEM dispatcher cannot change layout after first dispatch; "
                    f"expected hidden={self._zb_hidden} moe_expert_num={self._zb_moe_expert_num} "
                    f"use_quant={self._zb_use_quant} but got hidden={hidden} "
                    f"moe_expert_num={moe_expert_num} use_quant={use_quant}."
                )
            return

        from vllm_ascend.ops.fused_moe.zb_runtime import (
            compute_low_latency_max_recv_tokens,
            estimate_local_mem_size,
            get_zb_process_runtime,
        )

        if moe_expert_num % self.ep_world_size != 0:
            raise RuntimeError(
                "ZB SHMEM dispatcher requires moe_expert_num divisible "
                f"by ep_world_size, got moe_expert_num={moe_expert_num} "
                f"ep_world_size={self.ep_world_size}."
            )
        num_local_experts = moe_expert_num // self.ep_world_size
        experts_per_token = int(getattr(self._moe_config, "experts_per_token", 0) or 0)

        max_recv_tokens = compute_low_latency_max_recv_tokens(
            num_tokens_per_rank=self._zb_max_tokens_per_rank,
            ep_world_size=self.ep_world_size,
            num_local_experts=num_local_experts,
            experts_per_token=experts_per_token,
        )
        local_mem_size = estimate_local_mem_size(
            max_recv_tokens,
            hidden,
            use_quant=use_quant,
            moe_expert_num=moe_expert_num,
            ep_world_size=self.ep_world_size,
        )
        runtime = self._zb_runtime or get_zb_process_runtime()
        if runtime is None:
            raise RuntimeError(
                "ZB SHMEM buffers requested before process runtime init; call _ensure_zb_runtime_init() first."
            )

        early_local_mem_size = self._zb_early_local_mem_size
        if early_local_mem_size is None:
            early_local_mem_size = runtime.local_mem_size
        if early_local_mem_size is not None and local_mem_size > early_local_mem_size:
            raise RuntimeError(
                "ZB SHMEM early pool is too small for the actual dispatch layout; "
                f"early_local_mem_size={early_local_mem_size} "
                f"required_local_mem_size={local_mem_size} hidden={hidden} "
                f"moe_expert_num={moe_expert_num} use_quant={use_quant}. "
                "Increase VLLM_ASCEND_ZB_LOCAL_MEM_SIZE or fix early sizing."
            )

        runtime.alloc_ext_info()
        bundle = runtime.allocate_low_latency_tensors(
            max_recv_tokens=max_recv_tokens,
            hidden_size=hidden,
            device=device,
            use_quant=use_quant,
        )

        # Auxiliary buffers (not SHMEM-backed; sized to upper bounds so we can
        # reuse them across all iterations).
        assist_size = max_recv_tokens * 16
        aux = {
            "assist_info": torch.empty((assist_size,), dtype=torch.int32, device=device),
            "expert_token_nums": torch.empty((num_local_experts,), dtype=torch.int64, device=device),
            "ep_recv_count": torch.empty((moe_expert_num * self.ep_world_size,), dtype=torch.int32, device=device),
            "tp_recv_count": torch.empty((1,), dtype=torch.int32, device=device),
            "dynamic_scales": (
                bundle.dynamic_scales_out
                if bundle.dynamic_scales_out is not None
                else torch.empty((max_recv_tokens,), dtype=torch.float32, device=device)
            ),
        }

        self._zb_runtime = runtime
        self._zb_bundle = bundle
        self._zb_aux = aux
        self._zb_hidden = hidden
        self._zb_max_recv_tokens = max_recv_tokens
        self._zb_moe_expert_num = moe_expert_num
        self._zb_use_quant = use_quant

    def _ensure_zb_initialized(
        self,
        hidden: int,
        moe_expert_num: int,
        use_quant: bool,
        device: torch.device,
    ) -> None:
        if self._zb_runtime is None:
            self._ensure_zb_runtime_init()
        self._ensure_zb_buffers(
            hidden=hidden,
            moe_expert_num=moe_expert_num,
            use_quant=use_quant,
            device=device,
        )

    def _compute_comm_quant_mode(self, quant: MoEQuantParams) -> int:
        if quant.comm_quant_mode is not None:
            return quant.comm_quant_mode
        if quant.dispatch_with_quant:
            return 4 if self.a5_need_extra_args and quant.is_mxfp else 2
        return 0

    def _compute_dispatch_quant_mode(self, token_dispatch_input: MoETokenDispatchInput) -> int:
        return self._compute_comm_quant_mode(token_dispatch_input.quant)

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        from vllm_ascend.ops.fused_moe.zb_runtime import (
            zb_moe_distribute_dispatch,
        )

        hidden_states = token_dispatch_input.hidden_states
        topk_ids = token_dispatch_input.topk_ids
        routing = token_dispatch_input.routing
        quant = token_dispatch_input.quant
        expert_map = routing.expert_map
        assert expert_map is not None, "expert_map is required for MC2 token dispatch."

        moe_expert_num = len(expert_map) + routing.global_redundant_expert_num
        self.moe_expert_num = moe_expert_num

        quant_mode = self._compute_dispatch_quant_mode(token_dispatch_input)
        use_quant = quant_mode != 0
        expert_token_nums_type = _get_expert_token_nums_type(token_dispatch_input)

        self._ensure_zb_initialized(
            hidden=hidden_states.shape[-1],
            moe_expert_num=moe_expert_num,
            use_quant=use_quant,
            device=hidden_states.device,
        )
        self._zb_dispatch_quant_mode = quant_mode

        assert self._zb_runtime is not None and self._zb_bundle is not None and self._zb_aux is not None, (
            "TokenDispatcherWithZB failed to initialize SHMEM buffers."
        )
        bundle = self._zb_bundle
        aux = self._zb_aux
        runtime = self._zb_runtime

        zb_moe_distribute_dispatch(
            x=hidden_states,
            expert_ids=topk_ids,
            expand_x_out=bundle.expand_x_out,
            dynamic_scales_out=aux["dynamic_scales"],
            assist_info_for_combine_out=aux["assist_info"],
            expert_token_nums_out=aux["expert_token_nums"],
            ep_recv_count_out=aux["ep_recv_count"],
            tp_recv_count_out=aux["tp_recv_count"],
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.ep_rank_id,
            moe_expert_num=moe_expert_num,
            ext_info=runtime.ext_info,
            scales=None,
            x_active_mask=routing.mc2_mask if self.global_bs == 0 else None,
            tp_world_size=1,
            tp_rank_id=0,
            quant_mode=quant_mode,
            global_bs=self.global_bs,
            expert_token_nums_type=expert_token_nums_type,
        )

        # GMM calls dispose_tensor() on the dispatch output when dynamic_scale is
        # set; that must not touch the persistent SHMEM expand_x_out buffer.
        expand_x_for_gmm = bundle.expand_x_out.detach()

        # Always wire gmm2 into the full combine_x buffer (same shape as expand_x_out).
        # Valid token count is carried by group_list / expert_token_nums only.
        # Do not call .item() here: it syncs the NPU stream and breaks ACLGraph capture.
        gmm2_out = bundle.combine_x

        return MoETokenDispatchOutput(
            hidden_states=expand_x_for_gmm,
            dynamic_scale=aux["dynamic_scales"] if use_quant else None,
            group_list=aux["expert_token_nums"],
            group_list_type=expert_token_nums_type,
            gmm2_out=gmm2_out,
            combine_metadata=MoEMC2CombineMetadata(
                topk_ids=topk_ids,
                topk_weights=token_dispatch_input.topk_weights,
                expert_map=expert_map,
                ep_recv_counts=aux["ep_recv_count"],
                tp_recv_counts=aux["tp_recv_count"],
                assist_info_for_combine=aux["assist_info"],
                expand_scales=None,
                quant=quant,
                mc2_mask=routing.mc2_mask if self.global_bs == 0 else None,
            ),
        )

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        combine_metadata: MoEMC2CombineMetadata,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_ascend.ops.fused_moe.zb_runtime import (
            zb_moe_distribute_combine,
        )

        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."
        assert self._zb_runtime is not None and self._zb_bundle is not None, (
            "TokenDispatcherWithZB.token_combine called before token_dispatch initialized SHMEM runtime."
        )
        bundle = self._zb_bundle
        runtime = self._zb_runtime

        quant = combine_metadata.quant
        comm_quant_mode = self._compute_comm_quant_mode(quant)

        # GMM2 writes directly into SHMEM ``combine_x`` via ``zb_moe_grouped_matmul_gmm2_out``.
        num_expert_rows = hidden_states.size(0)
        expand_x = bundle.combine_x
        if num_expert_rows > expand_x.size(0):
            raise RuntimeError(
                f"ZB SHMEM combine: GMM output rows {num_expert_rows} exceed combine_x capacity {expand_x.size(0)}."
            )
        ori_x = None

        expand_scales = None
        if comm_quant_mode == 2 and self._zb_aux is not None:
            expand_scales = self._zb_aux.get("dynamic_scales")

        num_combined_tokens = combine_metadata.topk_ids.shape[0]
        combined_x = torch.empty(
            (num_combined_tokens, hidden_states.shape[-1]),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

        zb_moe_distribute_combine(
            expand_x=expand_x,
            expert_ids=combine_metadata.topk_ids,
            assist_info_for_combine=combine_metadata.assist_info_for_combine,
            ep_send_count=combine_metadata.ep_recv_counts,
            expert_scales=combine_metadata.topk_weights.to(torch.float32),
            combined_x=combined_x,
            tp_send_count=combine_metadata.tp_recv_counts,
            ori_x=ori_x,
            expand_scales=expand_scales,
            x_active_mask=combine_metadata.mc2_mask if self.global_bs == 0 else None,
            ep_world_size=self.ep_world_size,
            ep_rank_id=self.ep_rank_id,
            moe_expert_num=self.moe_expert_num,
            ext_info=runtime.ext_info,
            tp_world_size=1,
            tp_rank_id=0,
            global_bs=self.global_bs,
            comm_quant_mode=comm_quant_mode,
        )
        return combined_x


class TokenDispatcherWithAllGather(MoETokenDispatcher[MoEAllGatherCombineMetadata]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_num_tokens = kwargs.get("max_num_tokens")
        num_experts_local = kwargs.get("num_local_experts", 0)
        self.num_experts_local = (
            num_experts_local.item() if torch.is_tensor(num_experts_local) else int(num_experts_local)
        )

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        # TODO: After AllGather MXFP4 communication quantization thorough verification, remove this judgment.
        #  MXFP4 keeps dispatch unquantized in AllGather path, and quantizes again inside the MLP path.
        with_quant = (
            token_dispatch_input.quant.dispatch_with_quant
            and token_dispatch_input.quant.quant_type != QuantType.MXFP4
            and token_dispatch_input.quant.quant_type != QuantType.W8A8FP8
        )
        is_mxfp = token_dispatch_input.quant.is_mxfp
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        dynamic_scale = token_dispatch_input.routing.pertoken_scale
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        restore_shape = hidden_states.shape
        # Fuse the first dynamic quant of moe_mlp into initrouting when
        # dispatch_with_quant is on but got a None dynamic_scale.
        if with_quant and dynamic_scale is None:
            quant_mode = 3 if is_mxfp else 1
        else:
            quant_mode = -1

        num_tokens = hidden_states.shape[:-1].numel()
        apply_router_weight_on_input = token_dispatch_input.routing.apply_router_weight_on_input
        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert topk == 1, "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            mask = expert_map[topk_ids] != -1
            topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local
        sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,
            scale=dynamic_scale,
            active_num=num_tokens * self.top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=quant_mode,
        )
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode

        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            dynamic_scale=dynamic_scale if with_quant else None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        final_hidden_states = DeviceOperator.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=combine_metadata.expanded_row_idx,
            probs=combine_metadata.topk_weights,
        )
        if len(combine_metadata.restore_shape) == 3:
            final_hidden_states = final_hidden_states.view(combine_metadata.restore_shape)

        # these values are no longer used, so they need to be set to None for memory release.
        return final_hidden_states


class TokenDispatcherWithAll2AllV(MoETokenDispatcher[MoEAllToAllCombineMetadata]):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_local_experts = kwargs.get("num_local_experts", 0)

        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = self.ep_rank * self.num_local_experts

        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        assert len(self.local_expert_indices) == self.num_local_experts, "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1, (
                "local_expert_indices must be continuous"
            )

        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=self.ep_group)
        backend = self.ep_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ):
        use_mxfp_quant = token_dispatch_input.quant.is_mxfp
        with_quant = token_dispatch_input.quant.dispatch_with_quant
        scale_type = token_dispatch_input.quant.get_scale_type
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids

        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)

        dynamic_scale_after_all2all = None
        if with_quant:
            dst_type = token_dispatch_input.quant.get_dst_type
            permutated_local_input_tokens, dynamic_scale = DeviceOperator.npu_dynamic_quant(
                permutated_local_input_tokens, act_quant_type=dst_type, use_mxfp_quant=use_mxfp_quant
            )
            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group
            )
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits, self.ep_group
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Postprocess
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = (
            self._dispatch_postprocess(
                global_input_tokens,
                dynamic_scale_after_all2all,
                global_input_tokens_local_experts_indices,
                with_quant,
                scale_type,
            )
        )

        return MoETokenDispatchOutput(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            combine_metadata=MoEAllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                topk_weights=topk_weights,
                reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
                reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
                hidden_shape=hidden_shape,
                hidden_shape_before_permute=hidden_shape_before_permute,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        # 1. Preprocess using metadata
        hidden_states = self._combine_preprocess(hidden_states, combine_metadata)

        # 2. AllToAll
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            combine_metadata.input_splits,
            combine_metadata.output_splits,
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # 3. Postprocess using metadata
        output = self._combine_postprocess(permutated_local_input_tokens, combine_metadata)

        return output

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        (
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        ) = self._preprocess(topk_ids)
        hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=num_out_tokens,
        )

        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        )

    def _preprocess(self, topk_ids: torch.Tensor):
        num_local_tokens_per_expert = torch.histc(topk_ids, bins=self.num_experts, min=0, max=self.num_experts)

        ep_size = self.ep_size
        num_out_tokens = topk_ids.numel()

        input_splits = (
            num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"), non_blocking=True)
            .numpy()
        )

        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.ep_group
        ).reshape(ep_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ]
        if num_global_tokens_per_local_expert is None:
            raise ValueError("num_global_tokens_per_local_expert must be set before sum.")

        output_splits = (
            num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
        )
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError("num_global_tokens_per_local_expert must be set before operations.")
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel()
            )
        else:
            torch.npu.synchronize()

        return (
            num_tokens_per_local_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        )

    def _dispatch_postprocess(
        self,
        global_input_tokens,
        dynamic_scale_after_all2all,
        global_input_tokens_local_experts_indices,
        with_quant,
        scale_type,
    ):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, dynamic_scale_after_all2all, None

        assert global_input_tokens_local_experts_indices is not None, (
            "global_input_tokens_local_experts_indices must be provided"
        )

        experts_indices_2d_copy = global_input_tokens_local_experts_indices.reshape(
            global_input_tokens_local_experts_indices.shape[0], 1
        )

        if scale_type == torch.float8_e8m0fnu:
            dynamic_scale_for_routing = dynamic_scale_after_all2all.view(torch.float8_e8m0fnu)
            global_input_tokens, reversed_global_input_permutation_mapping, _, routed_scale = (
                torch_npu.npu_moe_init_routing_v2(
                    global_input_tokens,
                    experts_indices_2d_copy,
                    scale=dynamic_scale_for_routing,
                    active_num=experts_indices_2d_copy.shape[0],
                    expert_num=self.num_local_experts,
                    expert_tokens_num_type=1,
                    expert_tokens_num_flag=True,
                    active_expert_range=[0, self.num_local_experts],
                )
            )
            dynamic_scale_after_all2all = routed_scale.view(torch.uint8)
            experts_indices_2d_copy.untyped_storage().resize_(0)
            return global_input_tokens, dynamic_scale_after_all2all, reversed_global_input_permutation_mapping
        elif with_quant:
            dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                dynamic_scale_after_all2all.unsqueeze(-1), global_input_tokens_local_experts_indices
            )
            dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(-1)

        # Non-quantized case
        global_input_tokens, reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, global_input_tokens_local_experts_indices
        )
        return global_input_tokens, dynamic_scale_after_all2all, reversed_global_input_permutation_mapping

    def _combine_preprocess(
        self, hidden_states: torch.Tensor, combine_metadata: MoEAllToAllCombineMetadata
    ) -> torch.Tensor:
        # Unpermutation 2: expert output to AlltoAll input
        rev_global = combine_metadata.reversed_global_input_permutation_mapping
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1 and rev_global is not None:
            hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, rev_global)
        return hidden_states

    def _combine_postprocess(
        self,
        permutated_local_input_tokens: torch.Tensor,
        combine_metadata: MoEAllToAllCombineMetadata,
    ) -> torch.Tensor:
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=combine_metadata.reversed_local_input_permutation_mapping.to(torch.int32),
            probs=combine_metadata.topk_weights,
            restore_shape=combine_metadata.hidden_shape_before_permute,
        )
        output = output.view(combine_metadata.hidden_shape)
        return output
