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

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.lora.fused_moe import prepare_lora_indices
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg


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
        self.lora_context = None
        self.dynamic_mx_quant_scale_alg = get_dynamic_mx_quant_scale_alg()

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    @abstractmethod
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Prepare tensors before MoE computation. May involve:
          - Padding to align communication boundaries
          - Slicing across tensor-parallel ranks
          - Broadcasting across data-parallel ranks

        Args:
            hidden_states (torch.Tensor): Input features, shape [num_tokens, hidden_size]
            router_logits (torch.Tensor): Router outputs, shape [num_tokens, num_experts]
            replace_allreduce (bool): Bypass default all-reduce behavior
            quant_type: none, w8a8, w4a8, mxfp8, or mxfp4

        Returns:
            MoEPrepareOutput:
                - processed hidden_states (may be padded/sliced/broadcasted)
                - processed router_logits (may be recomputed or broadcasted)
                - optional communication mask (e.g., mc2_mask for sparse ops)
                - optional padded hidden state shape for finalization
                - optional per-token scale for quantized path
        """
        raise NotImplementedError("Prepare not implemented.")

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
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
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Pad hidden_states and router_logits to next multiple of TP size.
          2. If TP > 1, split along token dim and select current TP rank's slice.
          3. Save splits for later all-gather in finalize.

        Skips if `replace_allreduce` is True.

        Returns:
            MoEPrepareOutput where `mc2_mask` is None for All2All path.
        """
        self.replace_allreduce = replace_allreduce

        padded_hidden_states_shape = hidden_states.shape
        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            pad_size = self.tp_size - self.num_tokens  # Pad to TP size (cyclic)
            if self.lora_context is not None:
                prepare_lora_indices(
                    self.lora_context,
                    num_tokens=self.num_tokens,
                    pad_size=pad_size,
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                )

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)

                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    def pad_and_split_input_ids(
        self,
        input_ids,
    ):
        if not self.replace_allreduce:
            pad_size = self.tp_size - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            if self.tp_size > 1:
                input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)
                input_ids = input_ids[self.tp_rank]
        return input_ids

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalization steps:
          1. If TP > 1, all-gather slices to reconstruct full tensor.
          2. Unpad to original token count.
          3. Return [original_num_tokens, hidden_size] tensor.

        Skips if `replace_allreduce` is True.
        """

        if not self.replace_allreduce:
            if self.tp_size > 1:
                assert padded_hidden_states_shape is not None
                # Cannot reuse `split_hidden_states` from prepare phase as it
                # may share memory with original hidden_states. Since shared
                # experts may use the original tensor, reusing it would cause
                # in-place modification during all_gather, corrupting the data.
                gathered_hidden_states = torch.empty(
                    padded_hidden_states_shape, device=hidden_states.device, dtype=hidden_states.dtype
                )
                split_hidden_states = torch.tensor_split(gathered_hidden_states, self.tp_size, dim=0)
                dist.all_gather(list(split_hidden_states), hidden_states, self.moe_config.tp_group.device_group)
                hidden_states = gathered_hidden_states

            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[: self.num_tokens]

        return hidden_states


class PrepareAndFinalizeWithMC2(PrepareAndFinalizeWithAll2All):
    """
    MoE communication strategy using MC2, based on All2All with additional
    DP-wide padding and unpadding for sequence-parallel inputs.
    Designed for Ascend or environments requiring explicit padding and slicing control.
    Relies on `mc2_mask` and `padded_num_tokens` from forward_context for alignment.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()
        # (unpadded rows, trailing shape, dtype, padded rows) -> persistent padded buffer
        self._pad_cache: dict[tuple[int, tuple[int, ...], torch.dtype, int], torch.Tensor] = {}

    def _restore_tp_across_dp(self):
        """
        Restore original TP configuration.
        vLLM flattens TP and DP into a single dimension; this method recovers
        the true TP world size and rank for correct tensor slicing.
        """
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def _pad_to(self, t: torch.Tensor, target: int) -> torch.Tensor:
        """Pad `t` up to `target` rows along dim 0 using a persistent buffer.

        `nn.functional.pad` allocates a new tensor and zero-fills it on every call, which
        for MC2 means two extra kernels (PadV3 + MemSet) per MoE layer per forward. The
        padded length is `padded_num_tokens`, a function of the TP width rather than of the
        data, so it is constant for a given captured shape and the buffer can be allocated
        and zeroed once.

        `nn.functional.pad` guarantees the padded rows are zero, and that guarantee must be
        preserved: `mc2_mask` is only handed to the dispatch op when `global_bs == 0`, so on
        the other path the op does read the padded rows.

        The cache key therefore includes the unpadded row count as well as the target. Several
        batch sizes map to the same `padded_num_tokens` (with tp_size 8, batches 1..8 all pad
        to 8), and a buffer shared between them would keep the larger batch's rows where the
        smaller one expects zeros. Keying on both means rows `[t.shape[0]:target]` are never
        written for a given buffer, so they stay zero from allocation onwards. One buffer per
        captured shape costs a few KB.

        Reusing a buffer across layers is safe because this method has no multistream path, so
        each layer's dispatch consumes the padded tensor before the next layer's `prepare`
        rewrites it. A persistent allocation also suits ACL graph capture, which wants static
        addresses.
        """
        key = (t.shape[0], tuple(t.shape[1:]), t.dtype, target)
        buf = self._pad_cache.get(key)
        if buf is None or buf.device != t.device:
            buf = torch.zeros((target, *t.shape[1:]), dtype=t.dtype, device=t.device)
            self._pad_cache[key] = buf
        buf[: t.shape[0]].copy_(t)
        return buf

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Fetch `mc2_mask` and target padding length from forward context.
          2. Pad `hidden_states` and `router_logits` to target length if needed.
          3. If TP > 1, split tensors along token dimension and select current TP rank's slice.
          4. Split and return corresponding `mc2_mask`.

        With `replace_allreduce`, inputs are already TP-sharded. Pad only the
        local shard to the DP-wide MC2 length, preserving its original mask.

        Returns:
            MoEPrepareOutput, possibly sliced/padded.
        """
        self.replace_allreduce = replace_allreduce
        mc2_mask = _EXTRA_CTX.mc2_mask
        if self.replace_allreduce:
            # SP shards use the local token count, not the largest DP batch.
            # Select valid bits before adding padding for uniform MC2 batches.
            self.num_tokens = hidden_states.shape[0]
            start = self.tp_rank * self.num_tokens
            mc2_mask = mc2_mask[start : start + self.num_tokens]
            pad_size = _EXTRA_CTX.padded_num_tokens // self.tp_size - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                mc2_mask = nn.functional.pad(mc2_mask, (0, pad_size), value=False)
        elif self.tp_size > 1:
            # Also slice mc2_mask
            split_mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)
            mc2_mask = split_mc2_mask[self.tp_rank]

        padded_hidden_states_shape = hidden_states.shape
        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            if pad_size > 0:
                hidden_states = self._pad_to(hidden_states, target_pad_length)
                router_logits = self._pad_to(router_logits, target_pad_length)
                padded_hidden_states_shape = hidden_states.shape

            # Slice across TP ranks
            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)
                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=mc2_mask,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.replace_allreduce:
            # Return the original SP shard to the residual/shared-expert path.
            return hidden_states[: self.num_tokens]
        return super().finalize(hidden_states, reduce_results, padded_hidden_states_shape)

    def pad_and_split_input_ids(
        self,
        input_ids,
    ):
        if self.replace_allreduce:
            # MoE-only SP retains full token IDs, while model-level SP may
            # already shard them. Align to the local hidden states first.
            if input_ids.numel() != self.num_tokens:
                input_ids = sequence_parallel_chunk(input_ids.reshape(-1, 1)).reshape(-1)
            pad_size = _EXTRA_CTX.padded_num_tokens // self.tp_size - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))
        else:
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            if self.tp_size > 1:
                input_ids = torch.tensor_split(input_ids, self.tp_size, dim=0)
                input_ids = input_ids[self.tp_rank]
        return input_ids


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

    def _use_ep_sequence_parallel(self) -> bool:
        """Whether MoE itself must use the EP sequence-parallel path.

        The MoE configuration owns sequence-parallel tokens.  Selecting the
        EP path from any other flag would gather tokens a second time before
        routing.
        """
        return self.moe_config.is_sequence_parallel

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          AllGather hidden_states and router_logits to form global tensors.

        Returns:
            MoEPrepareOutput with global tensors.
        """
        if self._use_ep_sequence_parallel():
            return self._prepare_with_ep_group(hidden_states, router_logits, quant_type)

        return self._prepare_with_dp_group(hidden_states, router_logits, replace_allreduce)

    def _prepare_with_ep_group(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, quant_type=QuantType.NONE
    ) -> MoEPrepareOutput:
        pertoken_scale = None
        if quant_type == QuantType.W8A8:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        elif quant_type in (QuantType.W8A8MXFP, QuantType.W4A8MXFP):
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch.float8_e4m3fn,
                scale_alg=self.dynamic_mx_quant_scale_alg,
            )
        elif quant_type == QuantType.W4A4MXFP:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch_npu.float4_e2m1fn_x2,
                round_mode="round",
            )

        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states)
        router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(router_logits)

        self.num_tokens = hidden_states.shape[0]

        if pertoken_scale is not None:
            pertoken_scale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(pertoken_scale)

        if self.moe_config.pcp_size > 1:
            max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp

            self.num_tokens_pcp = hidden_states.shape[0]
            pad_size = max_tokens_across_pcp - self.num_tokens_pcp
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                if pertoken_scale is not None:
                    pertoken_scale = (
                        nn.functional.pad(pertoken_scale, (0, pad_size))
                        if pertoken_scale.dim() == 1
                        else nn.functional.pad(pertoken_scale, (0, 0, 0, pad_size))
                    )

            hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
            router_logits = get_pcp_group().all_gather(router_logits, dim=0)
            if pertoken_scale is not None:
                pertoken_scale = get_pcp_group().all_gather(pertoken_scale, dim=0)

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=pertoken_scale,
        )

    def _prepare_with_dp_group(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        """
        Preparation steps:
          1. Fetch max token count across DP group from forward context.
          2. Pad local tensors to that size.
          3. All-gather across DP group to form global input tensor.

        Returns:
            MoEPrepareOutput with global tensors.
        """
        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))

            # All-gather across DP group
            hidden_states = self.moe_config.dp_group.all_gather(hidden_states, 0)
            router_logits = self.moe_config.dp_group.all_gather(router_logits, 0)

        if self.moe_config.pcp_size > 1:
            max_tokens_across_pcp = _EXTRA_CTX.max_tokens_across_pcp

            self.num_tokens_pcp = hidden_states.shape[0]
            pad_size = max_tokens_across_pcp - self.num_tokens_pcp
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))

            hidden_states = get_pcp_group().all_gather(
                hidden_states,
                dim=0,
            )
            router_logits = get_pcp_group().all_gather(
                router_logits,
                dim=0,
            )

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )

    def all_gather_input_id_with_dp_group(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                input_ids = nn.functional.pad(input_ids, (0, pad_size))

            input_ids = self.moe_config.dp_group.all_gather(input_ids, 0)
        return input_ids

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Finalization steps:
          Reduce Scatter hidden states.

        Returns:
            Tensor with shape [local_num_tokens, hidden_size]
        """
        if self._use_ep_sequence_parallel():
            return self._finalize_with_ep_group(hidden_states)

        return self._finalize_with_dp_group(hidden_states, reduce_results)

    def _finalize_with_ep_group(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Argument `reduce_results` is not needed in this func. Given sequence parallelism is enabled:
        1. Reduce_results is False usually happens when models have shared experts and need to
        allreduce hidden states after results of shared experts and routed experts are added in FusedMoe.
        We do reduce scatter for hidden states here, then skip allreudce in FusedMoe and add it to the
        result of shared experts.
        2 Reduce_results is True usually happens when model has no shared experts. We still do reduce scatter
        here, then skip allreudce in FusedMoe.
        """
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
            hidden_states = hidden_states[: self.num_tokens_pcp]

        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states)

        return hidden_states

    def _finalize_with_dp_group(self, hidden_states: torch.Tensor, reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If DP > 1 and not shared expert, reduce-scatter output across DP group.
          2. Slice to original local token count.
          3. If `reduce_results=True` and TP/EP > 1, apply tensor_model_parallel_all_reduce.

        Returns:
            Tensor with shape [original_local_num_tokens, hidden_size]
        """
        if self.moe_config.pcp_size > 1:
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)
            hidden_states = hidden_states[: self.num_tokens_pcp]

        if self.moe_config.dp_size > 1:
            hidden_states = get_dp_group().reduce_scatter(hidden_states, 0)
            hidden_states = hidden_states[: self.num_tokens]
        return hidden_states
