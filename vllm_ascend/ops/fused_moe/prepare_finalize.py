# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_dp_group, get_pcp_group, get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.utils import enable_sp, prefill_context_parallel_enable


class QuantType(Enum):
    NONE = 0
    W8A8 = 1
    W4A8 = 2


class PrepareAndFinalize(ABC):
    """
    Abstract base class for MoE (Mixture-of-Experts) tensor preparation and finalization
    in distributed environments. Subclasses implement specific communication strategies
    (e.g., AllGather, All2All, MC2) to handle tensor padding, slicing,
    broadcasting, and reduction across TP/DP/EP groups.

    Attributes:
        moe_config (FusedMoEConfig): Configuration object containing TP/DP/EP group info,
                                     sizes, ranks, and communication settings.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

    @abstractmethod
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """
        Prepare tensors before MoE computation. May involve:
          - Padding to align communication boundaries
          - Slicing across tensor-parallel ranks
          - Broadcasting across data-parallel ranks

        Args:
            hidden_states (torch.Tensor): Input features, shape [num_tokens, hidden_size]
            router_logits (torch.Tensor): Router outputs, shape [num_tokens, num_experts]
            enable_shared_expert_dp (bool): Skip DP communication for shared experts
            replace_allreduce (bool): Bypass default all-reduce behavior
            quant_type: none, w8a8 or w4a8

        Returns:
            Tuple of:
                - processed hidden_states (may be padded/sliced/broadcasted)
                - processed router_logits (may be recomputed or broadcasted)
                - optional communication mask (e.g., mc2_mask for sparse ops)
                - optional context metadata (e.g., saved split_hidden_states for finalization)
        """
        raise NotImplementedError("Prepare not implemented.")

    def finalize(self,
                 hidden_states: torch.Tensor,
                 reduce_results: bool,
                 context_metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Finalize MoE output. May involve:
          - Gathering sliced tensors across TP ranks
          - Reducing or scattering across DP ranks
          - Unpadding to original token count
          - Applying all-reduce across TP/EP if requested

        Args:
            hidden_states (torch.Tensor): MoE layer output, possibly padded or sliced
            reduce_results (bool): Whether to apply all-reduce across TP/EP groups

        Returns:
            torch.Tensor: Final output with shape [original_num_tokens, hidden_size]
        """
        raise NotImplementedError("Finalize function not implemented.")


class PrepareAndFinalizeWithAll2All(PrepareAndFinalize):
    """
    MoE communication strategy using All-to-All style slicing.
    Similar to MC2 but does not use mc2_mask; instead pads to TP size for uniform slicing.
    Will be used when num_tokens exceed mc2's limitation (512 tokens/rank).
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """Restore original TP configuration (same as MC2)."""
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """
        Preparation steps:
          1. Pad hidden_states and router_logits to next multiple of TP size.
          2. If TP > 1, split along token dim and select current TP rank's slice.
          3. Save splits for later all-gather in finalize.

        Skips if `enable_shared_expert_dp` or `replace_allreduce` is True.

        Returns:
            Tuple of (hidden_states, router_logits, None, context_metadata) — no mask used in All2All.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp
        split_hidden_states = None

        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            self.num_tokens, _ = hidden_states.shape
            pad_size = self.tp_size - self.num_tokens  # Pad to TP size (cyclic)

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits,
                                                  (0, 0, 0, pad_size))

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states,
                                                         self.tp_size,
                                                         dim=0)
                split_router_logits = torch.tensor_split(router_logits,
                                                         self.tp_size,
                                                         dim=0)

                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        context_metadata = {"split_hidden_states": split_hidden_states}

        return hidden_states, router_logits, None, context_metadata

    def finalize(self,
                 hidden_states: torch.Tensor,
                 reduce_results: bool,
                 context_metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Finalization steps:
          1. If TP > 1, all-gather slices to reconstruct full tensor.
          2. Unpad to original token count.
          3. Return [original_num_tokens, hidden_size] tensor.

        Skips if `enable_shared_expert_dp` or `replace_allreduce` is True.
        """
        assert context_metadata is not None

        split_hidden_states = context_metadata["split_hidden_states"]
        if not (self.enable_shared_expert_dp or self.replace_allreduce):
            if self.tp_size > 1:
                dist.all_gather(list(split_hidden_states), hidden_states,
                                self.moe_config.tp_group.device_group)
                hidden_states = torch.cat(split_hidden_states, dim=0)

            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[:self.num_tokens]

        return hidden_states


class PrepareAndFinalizeWithMC2(PrepareAndFinalizeWithAll2All):
    """
    MoE communication strategy using MC2, which is based on All2All. Hence, it inherits
    All2All and share the same finalize method.
    Designed for Ascend or environments requiring explicit padding and slicing control.
    Relies on `mc2_mask` and `padded_num_tokens` from forward_context for alignment.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """
        Restore original TP configuration.
        vLLM flattens TP and DP into a single dimension; this method recovers
        the true TP world size and rank for correct tensor slicing.
        """
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """
        Preparation steps:
          1. Fetch `mc2_mask` and target padding length from forward context.
          2. Pad `hidden_states` and `router_logits` to target length if needed.
          3. If TP > 1, split tensors along token dimension and select current TP rank's slice.
          4. Split and return corresponding `mc2_mask`.

        Skips padding/slicing if `enable_shared_expert_dp` or `replace_allreduce` is True.

        Returns:
            Tuple of (hidden_states, router_logits, mc2_mask, context_metadata), possibly sliced/padded.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp
        split_hidden_states = None
        forward_context = get_forward_context()
        mc2_mask = forward_context.mc2_mask
        if self.tp_size > 1:
            # Also slice mc2_mask
            split_mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)
            mc2_mask = split_mc2_mask[self.tp_rank]

        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            target_pad_length = forward_context.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            # Pad if necessary (unless shared expert DP is enabled)
            if pad_size > 0 and not self.enable_shared_expert_dp:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits,
                                                  (0, 0, 0, pad_size))

            # Slice across TP ranks
            if self.tp_size > 1 and not self.enable_shared_expert_dp:
                split_hidden_states = torch.tensor_split(hidden_states,
                                                         self.tp_size,
                                                         dim=0)
                split_router_logits = torch.tensor_split(router_logits,
                                                         self.tp_size,
                                                         dim=0)
                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        context_metadata = {"split_hidden_states": split_hidden_states}

        return hidden_states, router_logits, mc2_mask, context_metadata


class PrepareAndFinalizeWithAllGather(PrepareAndFinalize):
    """
    MoE communication strategy using All-Gather + Reduce-Scatter on EP group.
    There are two sets of prepare and finalize:
    1. _prepare_with_dp_group/_finalize_with_dp_group: When sequence parallelism is not enabled,
    we gather inputs across DP ranks before MoE, scatter outputs after.
    The communication and calculation process is as follows (AG, AR and RS
    are abbreviations for All-Gather, All-Reduce and Reduce-Scatter, respectively):

    Attn → TP AR → DP AG → MoE → DP RS → TP AR

    2. _prepare_with_ep_group/_finalize_with_ep_group: When sequence parallelism is enabled,
    the above process becomes:

    TP AG → Attn → TP RS → TP AG → DP AG → MoE → DP RS → TP RS

    This strategy further combines TP AG + DP AG into EP All-Gather and TP RS + DP RS
    into EP Reduce-Scatter to improve communication performance. The optimized process is as follows:

    TP AG → Attn → TP RS → EP AG → MoE → EP RS
    """

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """
        Preparation steps:
          AllGather hidden_states and router_logits to form global tensors.

        Returns:
            Tuple of (global_hidden_states, global_router_logits, None)
        """
        if enable_sp():
            return self._prepare_with_ep_group(hidden_states, router_logits,
                                               quant_type)

        return self._prepare_with_dp_group(hidden_states, router_logits,
                                           enable_shared_expert_dp,
                                           replace_allreduce)

    def _prepare_with_ep_group(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        quant_type=QuantType.NONE
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        pertoken_scale = None
        if quant_type == QuantType.W8A8:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
                hidden_states)
            pertoken_scale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                pertoken_scale, True, True)
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states, True, True)
        router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            router_logits, True, True)

        if pertoken_scale is not None:
            return (hidden_states, pertoken_scale), router_logits, None, None

        return hidden_states, router_logits, None, None

    def _prepare_with_dp_group(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """
        Preparation steps:
          1. Fetch max token count across DP group from forward context.
          2. Pad local tensors to that size.
          3. All-gather across DP group to form global input tensor.

        Returns:
            Tuple of (global_hidden_states, global_router_logits, None, None)
        """
        self.enable_shared_expert_dp = enable_shared_expert_dp
        if self.moe_config.dp_size > 1:
            forward_context = get_forward_context()
            max_tokens_across_dp = forward_context.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits,
                                                  (0, 0, 0, pad_size))

            # All-gather across DP group
            hidden_states = self.moe_config.dp_group.all_gather(
                hidden_states, 0)
            router_logits = self.moe_config.dp_group.all_gather(
                router_logits, 0)

        if prefill_context_parallel_enable() and self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return hidden_states, router_logits, None, None

    def finalize(self,
                 hidden_states: torch.Tensor,
                 reduce_results: bool,
                 context_metadata: Optional[dict] = None) -> torch.Tensor:
        """
        Finalization steps:
          Reduce Scatter hidden states.

        Returns:
            Tensor with shape [local_num_tokens, hidden_size]
        """
        if enable_sp():
            return self._finalize_with_ep_group(hidden_states)

        return self._finalize_with_dp_group(hidden_states, reduce_results)

    def _finalize_with_ep_group(self,
                                hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Argument `reduce_results` is not needed in this func. Given sequence parallelism is enabled:
        1. Reduce_results is False usually happens when models have shared experts and need to
        allreduce hidden states after results of shared experts and routed experts are added in FusedMoe.
        We do reduce scatter for hidden states here, then skip allreudce in FusedMoe and add it to the
        result of shared experts.
        2 Reduce_results is True usually happens when model has no shared experts. We still do reduce scatter
        here, then skip allreudce in FusedMoe.
        """
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(
            hidden_states, True)

        return hidden_states

    def _finalize_with_dp_group(self, hidden_states: torch.Tensor,
                                reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If DP > 1 and not shared expert, reduce-scatter output across DP group.
          2. Slice to original local token count.
          3. If `reduce_results=True` and TP/EP > 1, apply tensor_model_parallel_all_reduce.

        Returns:
            Tensor with shape [original_local_num_tokens, hidden_size]
        """
        if self.moe_config.dp_size > 1 and not self.enable_shared_expert_dp:
            hidden_states = get_dp_group().reduce_scatter(hidden_states, 0)
            hidden_states = hidden_states[:self.num_tokens]

        if prefill_context_parallel_enable() and self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(hidden_states,
                                                           dim=0)
        if reduce_results and (self.moe_config.tp_size > 1
                               or self.moe_config.ep_size > 1):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states
