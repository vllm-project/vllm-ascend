# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.utils.torch_utils import async_tensor_h2d

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
        # The stable mapping buffer is also used by DSA's padded o-projection.
        # Initialize the tail once and only clear rows that become stale after
        # a batch shrinks, instead of allocating torch.full/copy tensors in
        # every transformer layer.
        self._token_lora_indices.fill_(-1)
        self._dsa_previous_token_count = 0
        self._dsa_lora_indices_cpu: tuple[int, ...] = ()
        self._dsa_lora_routing_cache: dict[tuple[int, int, int, bool, int], object] = {}
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

    def update_metadata(
        self,
        mapping,
        lora_index_to_id,
        max_loras,
        vocab_size,
        **kwargs,
    ) -> None:
        index_mapping = tuple(mapping.index_mapping)
        lora_id_to_index = {lora_id: index for index, lora_id in enumerate(lora_index_to_id) if lora_id is not None}
        self._dsa_lora_indices_cpu = tuple(
            lora_id_to_index.get(lora_id, -1) if lora_id > 0 else -1 for lora_id in index_mapping
        )
        routing_cache = getattr(self, "_dsa_lora_routing_cache", None)
        if routing_cache is None:
            self._dsa_lora_routing_cache = {}
        else:
            routing_cache.clear()

        # Preserve PunicaWrapperBase's common metadata handling. The overridden
        # prefill hook below consumes the host mapping prepared above, avoiding
        # compute_meta's device unique/item synchronization.
        super().update_metadata(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            **kwargs,
        )

        current_token_count = len(index_mapping)
        previous_token_count = getattr(self, "_dsa_previous_token_count", 0)
        if current_token_count < previous_token_count and hasattr(self, "_token_lora_indices"):
            self._token_lora_indices[current_token_count:previous_token_count].fill_(-1)
        self._dsa_previous_token_count = current_token_count

        loaded_lora_ids = {lora_id for lora_id in lora_index_to_id if lora_id is not None}
        self.num_active_moe_loras = len(
            set(lora_id for lora_id in index_mapping if lora_id > 0 and lora_id in loaded_lora_ids)
        )
        # PunicaWrapperBase computes this only for prefill. Decode must also
        # choose between the active-LoRA and base-only quantized MoE paths.
        self.no_lora = not any(lora_id > 0 for lora_id in index_mapping)

    def _update_prefill_metadata(self, token_lora_tensor: torch.Tensor) -> None:
        del token_lora_tensor
        self._update_prefill_metadata_from_cpu(self._dsa_lora_indices_cpu)

    def _update_prefill_metadata_from_cpu(self, token_lora_indices: tuple[int, ...]) -> None:
        """Build SGMV metadata without NPU unique/item synchronization."""

        starts: list[int] = []
        lengths: list[int] = []
        slots: list[int] = []
        for row, slot in enumerate(token_lora_indices):
            if not slots or slot != slots[-1]:
                starts.append(row)
                lengths.append(1)
                slots.append(slot)
            else:
                lengths[-1] += 1

        self.batch_size = len(slots)
        self.max_length = max(lengths, default=0)
        self.token_nums = len(token_lora_indices)
        self.no_lora = not any(slot >= 0 for slot in token_lora_indices)
        if not slots:
            return

        metadata = async_tensor_h2d(
            [starts, lengths, slots],
            dtype=torch.long,
            device=self.device,
        )
        self._seq_start_locs[: self.batch_size].copy_(metadata[0])
        self._seq_lengths[: self.batch_size].copy_(metadata[1])
        self._lora_indices_per_batch[: self.batch_size].copy_(metadata[2])

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
        self.bgmv_shrink(x, w_t_all, y, self._get_token_lora_indices(x), scale)

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
        self.bgmv_expand(x, w_t_all, y, self._get_token_lora_indices(x), add_inputs)

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
        self.bgmv_expand_slice(
            x,
            w_t_all,
            y,
            self._get_token_lora_indices(x),
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _get_token_lora_indices(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_token_lora_indices(x.size(0))

    def get_token_lora_indices(self, num_tokens: int) -> torch.Tensor:
        """Return device-side adapter slots for the requested token rows."""
        return torch.narrow(self._token_lora_indices, 0, 0, num_tokens)

    def get_dsa_lora_routing(
        self,
        offset: int,
        num_tokens: int,
        *,
        num_rows: int | None = None,
        prefer_grouped_matmul: bool = False,
        group_multiplier: int = 1,
    ):
        """Return cached DSA routing for one decode/prefill token segment."""

        from vllm_ascend.lora.dsa import DSALoRARouting

        if num_rows is None:
            num_rows = num_tokens
        if offset < 0 or num_tokens < 0 or num_rows < num_tokens:
            raise ValueError(
                f"Invalid DSA LoRA routing range: offset={offset}, num_tokens={num_tokens}, num_rows={num_rows}."
            )
        if offset + num_tokens > len(self._dsa_lora_indices_cpu):
            raise ValueError(
                "DSA LoRA routing exceeds the current token mapping: "
                f"offset={offset}, num_tokens={num_tokens}, "
                f"mapping_size={len(self._dsa_lora_indices_cpu)}."
            )
        if num_rows != num_tokens and offset != 0:
            raise ValueError("Only a zero-offset DSA LoRA segment may contain padded rows.")
        if group_multiplier < 1:
            raise ValueError(f"DSA LoRA group multiplier must be positive, got {group_multiplier}.")

        cache_key = (offset, num_tokens, num_rows, prefer_grouped_matmul, group_multiplier)
        cached = self._dsa_lora_routing_cache.get(cache_key)
        if cached is not None:
            return cached

        logical_slots = self._dsa_lora_indices_cpu[offset : offset + num_tokens]
        group_lengths: list[int] = []
        segment_slots: list[int] = []
        segment_starts: list[int] = []
        for row, slot in enumerate(logical_slots):
            if not segment_slots or slot != logical_slots[row - 1]:
                segment_starts.append(row)
                group_lengths.append(1)
                segment_slots.append(max(slot, 0))
            else:
                group_lengths[-1] += 1

        routing_indices = torch.narrow(
            self._token_lora_indices,
            0,
            offset if num_rows == num_tokens else 0,
            num_rows,
        )
        group_list = None
        segment_lora_indices = None
        if group_lengths:
            segment_metadata = async_tensor_h2d(
                [group_lengths, segment_slots],
                dtype=torch.long,
                device=self.device,
            )
            group_list = segment_metadata[0]
            segment_lora_indices = segment_metadata[1]

        has_lora = any(slot >= 0 for slot in logical_slots)
        has_base = any(slot < 0 for slot in logical_slots)
        active_mask = None
        if has_base:
            active_mask = async_tensor_h2d(
                [slot >= 0 for slot in logical_slots],
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(1)

        expanded_group_list = None
        grouped_group_list = None
        segment_group_lora_indices = None
        segment_group_lora_indices_cpu: tuple[int, ...] = ()
        grouped_row_indices = None
        if group_lengths:
            expanded_group_list = async_tensor_h2d(
                [length * group_multiplier for length in group_lengths],
                dtype=torch.long,
                device=self.device,
            )
            if group_multiplier > 1:
                grouped_lengths = [length for length in group_lengths for _ in range(group_multiplier)]
                grouped_slots = [
                    slot * group_multiplier + group for slot in segment_slots for group in range(group_multiplier)
                ]
                grouped_rows = [
                    token * group_multiplier + group
                    for start, length in zip(segment_starts, group_lengths)
                    for group in range(group_multiplier)
                    for token in range(start, start + length)
                ]
                grouped_metadata = async_tensor_h2d(
                    [grouped_lengths, grouped_slots],
                    dtype=torch.long,
                    device=self.device,
                )
                grouped_group_list = grouped_metadata[0]
                segment_group_lora_indices = grouped_metadata[1]
                segment_group_lora_indices_cpu = tuple(grouped_slots)
                grouped_row_indices = async_tensor_h2d(
                    grouped_rows,
                    dtype=torch.long,
                    device=self.device,
                )

        full_slots = (*logical_slots, *([-1] * (num_rows - num_tokens)))
        expanded_slots = [slot for slot in full_slots for _ in range(group_multiplier)]
        combined_slots = [
            slot * group_multiplier + group if slot >= 0 else -1
            for slot in full_slots
            for group in range(group_multiplier)
        ]
        expanded_metadata = async_tensor_h2d(
            [expanded_slots, combined_slots],
            dtype=torch.long,
            device=self.device,
        )
        expanded_active_mask = None
        if has_base:
            expanded_active_mask = async_tensor_h2d(
                [slot >= 0 for slot in logical_slots for _ in range(group_multiplier)],
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(1)

        routing = DSALoRARouting(
            token_lora_indices=routing_indices,
            num_tokens=num_tokens,
            prefer_grouped_matmul=prefer_grouped_matmul,
            has_lora=has_lora,
            has_base=has_base,
            segment_lora_indices_cpu=tuple(segment_slots),
            group_list=group_list,
            segment_lora_indices=segment_lora_indices,
            active_mask=active_mask,
            expanded_group_list=expanded_group_list,
            grouped_group_list=grouped_group_list,
            segment_group_lora_indices=segment_group_lora_indices,
            segment_group_lora_indices_cpu=segment_group_lora_indices_cpu,
            grouped_row_indices=grouped_row_indices,
            expanded_active_mask=expanded_active_mask,
            expanded_token_lora_indices=expanded_metadata[0],
            expanded_combined_indices=expanded_metadata[1],
        )
        self._dsa_lora_routing_cache[cache_key] = routing
        return routing

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
                    x[i].unsqueeze(0) @ lora_a_stacked[
                    indices[i], layer_idx, :, :] @ lora_b_stacked[
                    indices[i], layer_idx, :, :]
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

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        *,
        topk_weights: torch.Tensor | None = None,
        sorted_token_ids: torch.Tensor | None = None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor | None = None,
        max_lora_rank: int = 0,
        top_k_num: int = 1,
        shrink_config=None,
        expand_config=None,
        adapter_enabled: torch.Tensor,
        mul_routed_weight: bool = False,
        fully_sharded: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
    ) -> None:
        """
        Ascend-native fused MoE LoRA (v2): static-shape per-row gather via the
        same bgmv_shrink/bgmv_expand AscendC kernels (csrc/kernels/bgmv_*.cpp)
        used by the dense Linear LoRA layers, instead of grouping rows by a
        data-dependent ``torch.unique`` over active LoRA ids. The previous
        ``torch.unique``/``nonzero`` version produced output whose *shape*
        depended on tensor values, which ACL Graph capture cannot record
        (it failed with an `aclnnUnique2` error as soon as `enforce_eager`
        was turned off) -- every tensor below has a shape that depends only
        on input shapes, never on values, so this stays graph-capturable.

        Rows are already one-token-per-row (top_k_num=1). Each row needs the
        LoRA slot for (lora_id, expert_id), so we fold both into a single
        gather index into a ``[max_loras * num_experts, ...]`` view of the
        existing per-(lora, expert) weight stacks:
            combined_idx[row] = lora_id[row] * num_experts + expert_id[row]
        or -1 when the row has no active adapter, mirroring the -1 sentinel
        ``PunicaWrapperBase.token_lora_indices`` already uses. bgmv_shrink/
        bgmv_expand skip any row whose index is negative (leaving the
        zero-initialized shrink buffer / unmodified ``y`` in place), so
        inactive rows get a zero delta for free -- no Python-level branching
        needed.
        """
        del sorted_token_ids, num_tokens_post_padded, max_lora_rank
        del shrink_config, expand_config
        assert top_k_num == 1, "Ascend MoE LoRA v1 expects pre-expanded rows (top_k_num=1)."
        if token_lora_mapping is None:
            token_lora_mapping = self.token_lora_indices

        x2d = x.view(-1, x.shape[-1])
        y2d = y.view(-1, y.shape[-1])
        expert_idx = expert_ids.view(-1).to(torch.long)
        num_experts = lora_a_stacked[0].shape[1]

        lora_idx_safe = token_lora_mapping.clamp(min=0)
        enabled = (token_lora_mapping >= 0) & adapter_enabled[lora_idx_safe].bool()
        combined_idx = torch.where(
            enabled,
            lora_idx_safe * num_experts + expert_idx,
            torch.full_like(token_lora_mapping, -1),
        ).contiguous()

        cur_offset = offset
        for slice_idx in range(len(lora_a_stacked)):
            # lora_a_stacked[s]/lora_b_stacked[s]: [max_loras, num_experts, rank, *].
            # Flattening the leading two dims turns "gather by (lora, expert)"
            # into "the plain per-row gather" to reuse bgmv_shrink/bgmv_expand.
            a = lora_a_stacked[slice_idx]
            b = lora_b_stacked[slice_idx]
            local_rank = a.shape[-2]
            full_rank = b.shape[-1]
            out_size = b.shape[-2]
            a_flat = a.view(-1, local_rank, a.shape[-1])

            # bgmv_shrink writes fp32 (its Y_T); bgmv_expand reads fp32
            # (its X_T), so the shrink buffer is fp32.
            shrink_out = torch.zeros(
                (x2d.shape[0], local_rank),
                dtype=torch.float32,
                device=x2d.device,
            )

            self.bgmv_shrink(x2d, a_flat, shrink_out, combined_idx, 1.0)

            if fully_sharded:
                if local_rank == full_rank:
                    shrink_out = tensor_model_parallel_all_reduce(shrink_out)
                else:
                    shrink_out = tensor_model_parallel_all_gather(shrink_out)

            if shrink_out.shape[-1] != full_rank:
                raise ValueError(
                    "MoE LoRA rank mismatch after TP communication: "
                    f"A projection has rank {shrink_out.shape[-1]}, "
                    f"but LoRA B expects rank {full_rank}."
                )
            b_flat = b.view(-1, out_size, full_rank)

            delta = shrink_out
            if mul_routed_weight and topk_weights is not None:
                delta = shrink_out * topk_weights.view(-1, 1)

            self.bgmv_expand_slice(delta, b_flat, y2d, combined_idx, cur_offset, out_size, add_inputs=True)
            cur_offset += out_size

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

        indices = torch.narrow(self._sampler_indices, 0, 0, x.size(0))

        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)

        y = y.view_as(y_org)
