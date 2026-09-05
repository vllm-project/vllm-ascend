# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.utils import refresh_all_lora_classes


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
        self._max_batches = max_batches
        # Set when the fused Ascend LoRA kernels are unavailable. The
        # PyTorch-native fallback gathers a full [rows, rank, in] copy of the
        # adapter weights per call, so it is unusable for anything but a tiny
        # batch; the dense single-adapter route below replaces it whenever the
        # batch routes to at most one adapter. See `_dense_route_slot`.
        self._prefers_dense_single_adapter = False
        # Populated by `update_metadata` when the dense route is taken:
        # the stacked-weight slot every adapted row uses, and a float32 row
        # mask that zeroes the LoRA contribution of base-only rows.
        self._dense_lora_slot: int | None = None
        self._dense_row_mask: torch.Tensor | None = None
        self._dense_sampler_mask: torch.Tensor | None = None
        # Graph replay retains tensor addresses even when the next base-only
        # mapping clears the active mask references. Own the backing buffers
        # for the lifetime of the wrapper and refresh their contents in place.
        self._dense_row_mask_buffer = torch.empty(
            (self._token_lora_indices.numel(), 1), dtype=torch.float32, device=device
        )
        self._dense_sampler_mask_buffer = torch.empty(
            (self._sampler_indices.numel(), 1), dtype=torch.float32, device=device
        )
        if not get_current_hardware_profile().supports(HardwareCapability.LORA_CUSTOM_OPS) or (
            self.lora_config is not None and self.lora_config.max_lora_rank >= 128
        ):
            self._prefers_dense_single_adapter = True
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

    @staticmethod
    def _build_dense_mask(indices: torch.Tensor, slot: int, num_valid: int) -> torch.Tensor:
        """A ``[len(indices), 1]`` float mask selecting the rows routed to ``slot``.

        Entries past ``num_valid`` are left over from an earlier step, so they
        are forced to base weights rather than trusted.
        """
        mask = indices == slot
        mask[num_valid:] = False
        return mask.to(torch.float32).unsqueeze(1)

    def _dense_route_slot(self, mapping, lora_index_to_id) -> int | None:
        """Return the stacked-weight slot for a dense single-adapter forward.

        ``None`` means the ordinary segmented (sgmv/bgmv) route applies.

        Two situations make the segmented route unusable:

        * the fused Ascend kernels are not in play (``max_lora_rank >= 128``
          or hardware without them), in which case the PyTorch fallback
          materialises the adapter weights once per token row;
        * the mapping has more ``unique_consecutive`` runs than
          ``max_batches``, which overflows the sgmv metadata buffers. That is
          not exotic: per-token routing such as UNO's gated LoRA (base seed row
          + adapted noise rows) produces two runs per request.

        Both are answered by the same dense formulation as long as at most one
        adapter is active, which is exactly when it is expressible as two plain
        GEMMs plus a row mask.
        """
        index_mapping = mapping.index_mapping
        active_ids = {lora_id for lora_id in index_mapping if lora_id > 0}
        if len(active_ids) != 1:
            return None

        if not self._prefers_dense_single_adapter:
            # Only the overflow case is left to decide, and counting runs is an
            # O(num_tokens) host loop, so it is skipped when the answer is
            # already known.
            num_runs = 1
            previous = index_mapping[0]
            for lora_id in index_mapping:
                if lora_id != previous:
                    num_runs += 1
                    previous = lora_id
            if num_runs <= self._max_batches:
                return None

        active_id = next(iter(active_ids))
        for slot, lora_id in enumerate(lora_index_to_id):
            if lora_id == active_id:
                return slot
        return None

    def update_metadata(
        self,
        mapping,
        lora_index_to_id,
        max_loras,
        vocab_size,
        **kwargs,
    ) -> None:
        # PunicaWrapperBase computes this only for prefill. Decode must also
        # choose between the active-LoRA and base-only quantized MoE paths.
        no_lora = not any(lora_id > 0 for lora_id in mapping.index_mapping)
        if no_lora:
            # Nothing routes to an adapter, so every `add_*` entry point below
            # returns early and the segment metadata is never read. Building it
            # anyway costs two blocking `.item()` reads inside `compute_meta`,
            # which lands on the critical path of every base-only forward --
            # including UNO's verify forward, one per decode step.
            self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
            self.is_prefill = bool(mapping.is_prefill)
            self._dense_lora_slot = None
            self._dense_row_mask = None
            self._dense_sampler_mask = None
            self.no_lora = True
            return

        dense_slot = self._dense_route_slot(mapping, lora_index_to_id)
        if dense_slot is None:
            super().update_metadata(
                mapping,
                lora_index_to_id,
                max_loras,
                vocab_size,
                **kwargs,
            )
            self._dense_lora_slot = None
            self._dense_row_mask = None
            self._dense_sampler_mask = None
            self.no_lora = False
        else:
            # `_update_prefill_metadata` is what overflows on a many-run
            # mapping, and the dense route does not read its output, so build
            # only the base (per-token) metadata here.
            self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
            self.is_prefill = bool(mapping.is_prefill)
            self._dense_lora_slot = dense_slot
            num_tokens = len(mapping.index_mapping)
            # The mapping covers the scheduled tokens, but a graph-padded
            # forward hands the layers more rows than that, so both masks are
            # built at full buffer width with the padding tail routed to base
            # weights. A short mask would broadcast-fail inside the layer.
            self._dense_row_mask_buffer.copy_(self._build_dense_mask(self._token_lora_indices, dense_slot, num_tokens))
            self._dense_sampler_mask_buffer.copy_(
                self._build_dense_mask(self._sampler_indices, dense_slot, len(mapping.prompt_mapping))
            )
            self._dense_row_mask = self._dense_row_mask_buffer
            self._dense_sampler_mask = self._dense_sampler_mask_buffer
            self.no_lora = False

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
        # The prefill helpers above return early on `no_lora`; the decode ones
        # historically relied on the fused Ascend kernels skipping rows whose
        # index is the `-1` sentinel. The PyTorch reference ops chosen for
        # `max_lora_rank >= 128` do not: `weights[-1]` wraps round to the last
        # adapter, so a base-only decode would silently come out adapted.
        if self.no_lora:
            return
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
        # See `_shrink_decode`: the `-1` sentinel is only honoured by the fused
        # Ascend kernels.
        if self.no_lora:
            return
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
        # See `_shrink_decode`: the `-1` sentinel is only honoured by the fused
        # Ascend kernels.
        if self.no_lora:
            return
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
        return torch.narrow(self._token_lora_indices, 0, 0, x.size(0))

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
        if self._dense_lora_slot is not None:
            self._dense_shrink(y, x, lora_a_stacked, scale)
            return
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
        if self._dense_lora_slot is not None:
            self._dense_expand(y, x, lora_b_stacked, output_slices, offset_start, add_inputs)
            y = y.view_as(y_org)
            return
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

    @staticmethod
    def _select_dense_weight(stacked: torch.Tensor, slot: int) -> torch.Tensor:
        """Return the 2-D weight for one adapter slot.

        Stacked LoRA weights are ``[num_loras, 1, out, in]`` (the middle axis is
        a vestigial layer dimension) or already ``[num_loras, out, in]``. Any
        other layout belongs to a layer family the dense route does not model,
        so it fails here rather than silently reshaping.
        """
        weight = stacked[slot]
        if weight.dim() == 3:
            if weight.shape[0] != 1:
                raise NotImplementedError(
                    f"Unexpected stacked LoRA weight layout {tuple(stacked.shape)} for the dense route."
                )
            weight = weight[0]
        if weight.dim() != 2:
            raise NotImplementedError(
                f"Unexpected stacked LoRA weight layout {tuple(stacked.shape)} for the dense route."
            )
        return weight

    def _dense_shrink(
        self,
        y: tuple[torch.Tensor, ...],
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
    ) -> None:
        """``y[i] = (x @ A_i.T) * scale``, zeroed on rows that stay base-only.

        Zeroing the rank-sized intermediate rather than masking the input keeps
        the mask cost proportional to the rank, and makes the paired
        ``_dense_expand`` a plain accumulation for every row.
        """
        slot = self._dense_lora_slot
        assert slot is not None
        rows = x.size(0)
        x_f32 = x.to(torch.float32)
        row_mask = None if self._dense_row_mask is None else self._dense_row_mask[:rows]
        for slice_idx, a_stacked in enumerate(lora_a_stacked):
            weight = self._select_dense_weight(a_stacked, slot)
            buffer = y[slice_idx].view(-1, y[slice_idx].shape[-1])
            rank = weight.shape[0]
            shrunk = torch.mm(x_f32, weight.to(torch.float32).t())
            shrunk *= scale
            if row_mask is not None:
                shrunk *= row_mask
            buffer[:, :rank] = shrunk

    def _dense_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int,
        add_inputs: bool,
    ) -> None:
        """``y[:, offset:offset+slice] (+)= x[i] @ B_i.T`` for one adapter."""
        slot = self._dense_lora_slot
        assert slot is not None
        offset_left = offset_start
        for slice_idx, b_stacked in enumerate(lora_b_stacked):
            weight = self._select_dense_weight(b_stacked, slot)
            slice_size = output_slices[slice_idx]
            if weight.shape[0] != slice_size:
                # The reference ops write exactly `slice_size` columns; a
                # narrower B would leave the tail of the slice untouched, which
                # is wrong for `add_inputs=False`.
                raise NotImplementedError(
                    f"LoRA B slice is {weight.shape[0]} wide but the output slice is {slice_size}."
                )
            shrunk = x[slice_idx].view(-1, x[slice_idx].shape[-1])
            rank = weight.shape[1]
            expanded = torch.mm(shrunk[:, :rank].to(y.dtype), weight.to(y.dtype).t())
            if add_inputs:
                y[:, offset_left : offset_left + slice_size] += expanded
            else:
                y[:, offset_left : offset_left + slice_size] = expanded
            offset_left += slice_size

    def _dense_expand_single(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        row_mask: torch.Tensor | None,
        add_inputs: bool,
    ) -> None:
        """``y (+)= (x @ B.T)`` for one adapter, zeroed on unrouted rows.

        Used by the embedding and logits wrappers, which have a single
        unsliced B and apply the mask on the output rather than on a rank-sized
        intermediate.
        """
        slot = self._dense_lora_slot
        assert slot is not None
        weight = self._select_dense_weight(lora_b_stacked, slot)
        y_flat = y.view(-1, y.shape[-1])
        x_flat = x.view(-1, x.shape[-1])
        rows = x_flat.size(0)
        expanded = torch.mm(x_flat.to(y_flat.dtype), weight.to(y_flat.dtype).t())
        if row_mask is not None:
            expanded *= row_mask[:rows].to(expanded.dtype)
        common = min(expanded.shape[1], y_flat.shape[1])
        if add_inputs:
            y_flat[:, :common] += expanded[:, :common]
        else:
            y_flat[:, :common] = expanded[:, :common]

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
        x = x.to(torch.float32)
        if self._dense_lora_slot is not None:
            # The dense route deliberately does not build the sgmv segment
            # metadata `_expand_prefill` reads, so the embedding delta has to be
            # expressed densely too.
            self._dense_expand_single(y, x, lora_b_stacked, self._dense_row_mask, add_inputs)
            return
        expand_fun: Callable = self._expand_prefill if self.is_prefill else self._expand_decode
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
        del shrink_config, expand_config, fully_sharded
        assert top_k_num == 1, "Ascend MoE LoRA v1 expects pre-expanded rows (top_k_num=1)."
        if self.no_lora:
            # Every row would carry the `-1` sentinel, which the PyTorch
            # reference ops turn into "the last adapter" rather than "no
            # adapter". See `_shrink_decode`.
            return
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

        # bgmv_shrink writes fp32 (its Y_T); bgmv_expand reads fp32 (its X_T),
        # so the shrink buffer is fp32.
        rank = lora_a_stacked[0].shape[-2]
        shrink_out = torch.zeros((x2d.shape[0], rank), dtype=torch.float32, device=x2d.device)

        cur_offset = offset
        for slice_idx in range(len(lora_a_stacked)):
            # lora_a_stacked[s]/lora_b_stacked[s]: [max_loras, num_experts, rank, *].
            # Flattening the leading two dims turns "gather by (lora, expert)"
            # into "the plain per-row gather" to reuse bgmv_shrink/bgmv_expand.
            a = lora_a_stacked[slice_idx]
            b = lora_b_stacked[slice_idx]
            out_size = b.shape[-2]
            a_flat = a.view(-1, rank, a.shape[-1])
            b_flat = b.view(-1, out_size, rank)

            self.bgmv_shrink(x2d, a_flat, shrink_out, combined_idx, 1.0)

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
        if self.no_lora:
            return
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)

        if self._dense_lora_slot is not None:
            # Same reason as `add_lora_embedding`: the segment metadata the
            # bgmv fallback would index does not exist on the dense route, and
            # its -1 sentinel is not honoured by the PyTorch ops.
            slot = self._dense_lora_slot
            weight_a = self._select_dense_weight(lora_a_stacked, slot)
            buffer[:, : weight_a.shape[0]] = torch.mm(x.to(torch.float32), weight_a.to(torch.float32).t()) * scale
            self._dense_expand_single(y, buffer, lora_b_stacked, self._dense_sampler_mask, True)
            y = y.view_as(y_org)
            return

        indices = torch.narrow(self._sampler_indices, 0, 0, x.size(0))

        self.bgmv_shrink(x, lora_a_stacked, buffer, indices, scale)
        self.bgmv_expand(buffer, lora_b_stacked, y, indices, add_inputs=True)

        y = y.view_as(y_org)
