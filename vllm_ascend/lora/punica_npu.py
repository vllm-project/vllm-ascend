# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


# The platforms that are compatible with the PyTorch-native implementation can
# inherit this class
class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int, device: torch.device | str, **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        refresh_all_lora_classes()
        self.lora_config = kwargs.get("lora_config")
        if get_ascend_device_type() == AscendDeviceType._310P or (
            self.lora_config is not None and self.lora_config.max_lora_rank >= 128
        ):
            from vllm.lora.ops.torch_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        else:
            from vllm_ascend.lora.lora_ops import (
                bgmv_expand,
                bgmv_expand_slice,
                bgmv_shrink,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_shrink = bgmv_shrink
        self.sgmv_expand = sgmv_expand
        self.sgmv_expand_slice = sgmv_expand_slice
        self.sgmv_shrink = sgmv_shrink

    def _shrink_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def _shrink_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        self.bgmv_shrink(x, w_t_all, y, self.token_lora_indices, scale)

    def _expand_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            add_inputs,
        )

    def _expand_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        self.bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_inputs)

    def _expand_slice_prefill(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        # No LoRA request, so return directly
        if self.no_lora:
            return
        self.sgmv_expand_slice(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _expand_slice_decode(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        self.bgmv_expand_slice(x, w_t_all, y, self.token_lora_indices, y_offset, y_slice_size, add_inputs)

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool = True,
    ):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all`
        computation, which is suitable for the
        GEMM of lora'b.
        """

        expand_slice_fun: Callable = self._expand_slice_prefill if self.is_prefill else self._expand_slice_decode
        expand_slice_fun(y, x, w_t_all, y_offset, y_slice_size, add_inputs)

    def _apply_shrink(self, y: torch.Tensor, x: torch.Tensor, w_t_all: torch.Tensor, scale: float):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        shrink_fun: Callable = self._shrink_prefill if self.is_prefill else self._shrink_decode
        shrink_fun(y, x, w_t_all, scale)
        y = y.view_as(y_org)

    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        # TODO fuse these kernels
        for slice_idx in range(len(lora_a_stacked)):
            self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx], scale)

    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (Tuple[int, ...]): Every slice's size
            offset_start (int): The starting position of y, defaults to 0
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = offset_start
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_embedding(
        self, y: torch.Tensor, x: torch.Tensor, lora_b_stacked: torch.Tensor, add_inputs: bool = True, **kwargs
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        # Embedding layer only need expand op
        expand_fun: Callable = self._expand_prefill if self.is_prefill else self._expand_decode
        x = x.to(torch.float32)
        expand_fun(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            buffer = tuple(
                torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device) for _ in range(len(output_slices))
            )
        self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs)

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)

        indices = self.sampler_indices

        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)

        y = y.view_as(y_org)

    # ------------------------------------------------------------------
    # MoE-LoRA primitives (v1 reference implementation)
    # ------------------------------------------------------------------
    # NPU does not yet have a fused MoE-LoRA kernel equivalent to the
    # upstream Triton `fused_moe_lora`. v1 ships a torch-based reference
    # implementation: correct-by-construction, suitable for functional
    # verification on real adapters. A fused AscendC kernel is planned
    # for v2.

    def moe_lora_align_block_size(
        self,
        topk_ids: torch.Tensor,
        num_tokens: int,
        block_size: int,
        num_experts: int,
        max_loras: int,
        adapter_enabled: torch.Tensor,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
        naive_block_assignment: bool = False,
    ):
        """Aligns tokens and experts for MoE-LoRA execution.

        v1 NPU implementation always returns the naive layout (one row per
        (orig_token, k) replica) — the equivalent of upstream GPU's
        `naive_block_assignment=True` branch. The caller (AscendFusedMoEWithLoRA)
        does not currently consume `sorted_token_ids` / `num_tokens_post_padded`,
        so we leave them as None to avoid extra allocations.

        Returns:
            tuple(token_lora_mapping, sorted_token_ids, expert_ids,
                  num_tokens_post_padded)
            * token_lora_mapping: 1D LongTensor[num_tokens] of LoRA slot id per
              original token. -1 means "no LoRA".
            * sorted_token_ids: always None in v1.
            * expert_ids: flat 1D LongTensor[num_tokens * top_k]; the expert id
              each (token, k) replica is routed to. Translated by expert_map if
              provided.
            * num_tokens_post_padded: always None in v1.
        """
        del block_size, num_experts, max_loras, pad_sorted_ids
        del naive_block_assignment  # always naive in v1
        token_lora_mapping = self.token_lora_indices[:num_tokens]
        expert_ids = topk_ids.reshape(-1)
        if expert_map is not None:
            expert_ids = expert_map[expert_ids]
        return token_lora_mapping, None, expert_ids, None

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor | None,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor | None,
        max_lora_rank: int,
        top_k_num: int,
        shrink_config,
        expand_config,
        adapter_enabled: torch.Tensor,
        mul_routed_weight: bool = False,
        fully_sharded: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
    ):
        """In-place adds the MoE-LoRA delta to `y` using combined-index bgmv.

        Trick: bgmv expects 3D weights ``[num_loras, out, in]`` and a 1D
        ``lora_indices`` of length N. Our per-expert LoRA weights are 4D
        ``[max_loras+1, local_E, rank/slice, in]``. We fold ``(lora_id,
        expert_id)`` into a single virtual index ``e * (max_loras+1) + l``
        and reshape the weights into ``[(max_loras+1)*local_E, ...]`` — a
        zero-copy view. This collapses the v1 Python double loop into two
        bgmv calls per slice, getting all AscendC-level perf for free.

        Disabled / unmapped rows (``token_lora_mapping == -1`` or
        ``adapter_enabled[l] == 0``) are routed to a sentinel slot (lora id 0,
        expert 0) and their shrink buffer is zeroed afterwards, so expand
        contributes nothing. This avoids any host sync.
        """
        del sorted_token_ids, num_tokens_post_padded
        del max_lora_rank, top_k_num, shrink_config, expand_config
        del fully_sharded  # not needed on the permuted-domain Ascend path

        if mul_routed_weight:
            raise NotImplementedError(
                "mul_routed_weight=True is not supported on the Ascend MoE-LoRA "
                "path (LoRA is applied to permuted activations before combine)."
            )
        if token_lora_mapping is None:
            token_lora_mapping = self.token_lora_indices[: x.size(0)]

        # Early-out: prefill metadata explicitly tagged no LoRA in this batch.
        if self.no_lora:
            return

        # max_loras+1 to include the sentinel "no-LoRA" slot at the end.
        max_loras_plus_one = int(adapter_enabled.shape[0])
        local_E = lora_a_stacked[0].shape[1]

        # ---- build effective lora id per permuted row, no host sync ----
        # adapter_enabled[token_lora_mapping] needs clamp(min=0) so the -1
        # sentinel doesn't out-of-bounds index. The clamped value will be
        # masked out below via valid_mask.
        tlm_clamped = token_lora_mapping.clamp(min=0)
        valid_mask = (token_lora_mapping >= 0) & (
            adapter_enabled[tlm_clamped] != 0
        )
        # Sentinel routing: invalid rows go to (lora_id=0, expert_id=0). The
        # actual values pulled from the weight tensors will be overwritten by
        # zeros in the shrink buffer before expand.
        safe_lora = torch.where(
            valid_mask, tlm_clamped, torch.zeros_like(tlm_clamped)
        )
        # expert_ids may also be -1 if expert_map filtered out tokens on other
        # EP ranks; clamp similarly. EP is currently asserted off, but the
        # defensive clamp keeps the kernel safe.
        safe_expert = expert_ids.clamp(min=0).to(torch.long)
        # virtual_idx[i] = expert * (max_loras+1) + lora_id
        # Layout matches torch.reshape of A from
        # [max_loras+1, local_E, rank, in] (stride: local_E*..., ...)
        # FIX: actually A is [max_loras+1, local_E, ...]. Reshape into
        # [(max_loras+1)*local_E, ...] has stride [local_E*..., ...] only if
        # we reshape via permute. Below we permute to [local_E, max_loras+1, ...]
        # then reshape, so virtual_idx = expert * (max_loras+1) + lora_id
        # picks A_view[expert*(max_loras+1) + lora_id] = original A[lora_id, expert].
        virtual_idx = safe_expert * max_loras_plus_one + safe_lora.to(torch.long)

        # ---- iterate slices (gate / up for _w13_slices==2, just one for w2) ----
        for slice_idx in range(len(lora_a_stacked)):
            A = lora_a_stacked[slice_idx]  # [max_loras+1, local_E, rank, in]
            B = lora_b_stacked[slice_idx]  # [max_loras+1, local_E, slice, rank]
            rank = A.shape[2]
            slice_out = B.shape[2]
            col_start = offset + slice_idx * slice_out

            # Permute (max_loras+1, local_E, ...) -> (local_E, max_loras+1, ...)
            # then reshape so virtual_idx layout matches.
            # .contiguous() is required because bgmv indexes flat dim 0.
            A_flat = A.permute(1, 0, 2, 3).reshape(
                local_E * max_loras_plus_one, rank, A.shape[-1]
            ).contiguous()
            B_flat = B.permute(1, 0, 2, 3).reshape(
                local_E * max_loras_plus_one, slice_out, rank
            ).contiguous()

            # bgmv_shrink writes (not accumulates) into the rank buffer.
            buffer = torch.zeros(
                (x.size(0), rank), dtype=torch.float32, device=x.device
            )
            self.bgmv_shrink(x, A_flat, buffer, virtual_idx, 1.0)
            # Mask out invalid rows so expand contributes 0 to y.
            buffer.mul_(valid_mask.unsqueeze(-1).to(buffer.dtype))
            # bgmv_expand_slice accumulates into y[:, col_start:col_start+slice].
            self.bgmv_expand_slice(
                buffer, B_flat, y, virtual_idx, col_start, slice_out, True
            )
        return
