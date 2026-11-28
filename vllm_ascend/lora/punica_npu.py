# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.lora.lora_ops import _LORA_WRAPPER_IDS, _LORA_WRAPPERS, lora_linear
from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def _expand_with_dtype_guard(
    y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    y_offset: int,
    y_slice_size: int,
    add_inputs: bool,
    run_expand: Callable[[torch.Tensor, torch.Tensor, int, bool], object],
) -> None:
    """Invoke a LoRA expand kernel with strict weight-to-output dtype parity."""
    x_fp32 = x if x.dtype == torch.float32 else x.to(torch.float32)
    if y.dtype == weight.dtype:
        run_expand(y, x_fp32, y_offset, add_inputs)
        return
    # Ascend expand kernels launch in y.dtype and read weight using the same
    # element type; stage into a weight-matched buffer on fp32 or cross-half/bf16.
    op_dtype = weight.dtype if weight.dtype in (torch.half, torch.bfloat16) else torch.bfloat16
    delta = torch.zeros((y.size(0), y_slice_size), dtype=op_dtype, device=y.device)
    run_expand(delta, x_fp32, 0, True)
    target = y[:, y_offset : y_offset + y_slice_size]
    if add_inputs:
        target.add_(delta.to(y.dtype))
    else:
        target.copy_(delta.to(y.dtype))


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
        self._max_num_batched_tokens = max_num_batched_tokens
        self._lora_shrink_buffers: dict[tuple[int, int], torch.Tensor] = {}
        self._lora_triton_workspaces: dict[int, torch.Tensor] = {}
        self.lora_config = kwargs.get("lora_config")
        ascend_device_type = get_ascend_device_type()
        self.ascend_device_type = ascend_device_type
        if not get_current_hardware_profile().supports(HardwareCapability.LORA_CUSTOM_OPS) or (
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
        self._single_lora_slot = (
            ascend_device_type in {AscendDeviceType.A2, AscendDeviceType.A3}
            and self.lora_config is not None
            and self.lora_config.max_loras == 1
            and not self.lora_config.fully_sharded_loras
        )
        self._single_lora_mask = None
        self.active_lora_slots: set[int] | None = None
        self._wrapper_id = next(_LORA_WRAPPER_IDS)
        _LORA_WRAPPERS[self._wrapper_id] = self
        if self._single_lora_slot:
            assert self.lora_config is not None
            lora_dtype = self.lora_config.lora_dtype
            if not isinstance(lora_dtype, torch.dtype):
                raise ValueError(f"LoRA dtype must be resolved before creating the Punica wrapper, got {lora_dtype!r}")
            self._single_lora_mask = torch.empty(
                (max_num_batched_tokens, 1),
                dtype=lora_dtype,
                device=device,
            )

    def _update_base_metadata(
        self,
        mapping,
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
    ) -> None:
        super()._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
        if self._single_lora_mask is None:
            return

        token_count = self.indices_len[0]
        assert token_count is not None
        self._single_lora_mask[:token_count].copy_(self._token_lora_indices[:token_count].eq(0).unsqueeze(1))

    def update_metadata(
        self,
        mapping,
        lora_index_to_id,
        max_loras,
        vocab_size,
        **kwargs,
    ) -> None:
        super().update_metadata(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            **kwargs,
        )
        # PunicaWrapperBase computes this only for prefill. Decode must also
        # choose between the active-LoRA and base-only quantized MoE paths.
        # mapping.index_mapping holds request LoRA IDs (0 means base), not resolved slots.
        request_ids = {lora_id for lora_id in mapping.index_mapping if lora_id > 0}
        self.no_lora = not request_ids
        self.active_lora_slots = {slot for slot, lora_id in enumerate(lora_index_to_id) if lora_id in request_ids}

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
        return torch.narrow(self._token_lora_indices, 0, 0, x.size(0))

    def _in_graph_mode(self) -> bool:
        try:
            return torch.npu.is_current_stream_capturing()
        except Exception:
            return False

    def _use_sgmv(self) -> bool:
        if not self.is_prefill:
            return False
        return not (self.ascend_device_type == AscendDeviceType._310P and self._in_graph_mode())

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

        expand_slice_fun: Callable = self._expand_slice_prefill if self._use_sgmv() else self._expand_slice_decode
        if self._use_sgmv() and self.no_lora:
            return
        _expand_with_dtype_guard(
            y,
            x,
            w_t_all,
            y_offset,
            y_slice_size,
            add_inputs,
            lambda out, inp, offset, add_in: expand_slice_fun(out, inp, w_t_all, offset, y_slice_size, add_in),
        )

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
        if w_t_all.is_floating_point() and x.dtype != w_t_all.dtype:
            x = x.to(w_t_all.dtype)
        shrink_fun: Callable = self._shrink_prefill if self._use_sgmv() else self._shrink_decode
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
        if self._use_sgmv() and self.no_lora:
            return
        expand_fun: Callable = self._expand_prefill if self._use_sgmv() else self._expand_decode
        _expand_with_dtype_guard(
            y,
            x,
            lora_b_stacked,
            0,
            y.size(1),
            add_inputs,
            lambda out, inp, _offset, add_in: expand_fun(out, inp, lora_b_stacked, add_in),
        )

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

        if self.no_lora:
            return

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        if buffer is not None:
            self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
            self.add_expand(y, buffer, lora_b_stacked, output_slices, add_inputs=True, **kwargs)
            return

        lora_linear(
            self._wrapper_id,
            y,
            x,
            list(lora_a_stacked),
            list(lora_b_stacked),
            scale,
            list(output_slices),
            kwargs.get("packed_lora_a"),
            kwargs.get("packed_lora_b"),
            kwargs.get("add_inputs", True),
        )

    def _lora_linear_kernel(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        scale: float,
        output_slices: tuple[int, ...] | list[int],
    ) -> None:
        lora_a = tuple(lora_a_stacked)
        lora_b = tuple(lora_b_stacked)
        slices = tuple(output_slices)

        r = lora_b[0].size(-1)
        buffer = self._get_shrink_buffer(len(slices), x.size(0), r, x.device)
        self.add_shrink(buffer, x, lora_a, scale)
        self.add_expand(y, buffer, lora_b, slices, add_inputs=True)

    def _get_shrink_buffer(
        self,
        n_slices: int,
        num_tokens: int,
        rank: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        key = (n_slices, rank)
        buf = self._lora_shrink_buffers.get(key)
        if buf is None or buf.shape[1] < num_tokens:
            alloc_rows = max(num_tokens, self._max_num_batched_tokens)
            buf = torch.zeros(
                (n_slices, alloc_rows, rank),
                dtype=torch.float32,
                device=device,
            )
            self._lora_shrink_buffers[key] = buf
        return tuple(buf[i][:num_tokens] for i in range(n_slices))

    def _lora_linear_matmul(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...] | list[torch.Tensor],
        scale: float,
        output_slices: tuple[int, ...] | list[int],
        packed_lora_a: torch.Tensor | None,
        packed_lora_b: torch.Tensor | None,
        add_inputs: bool,
    ) -> None:
        x = x.view(-1, x.shape[-1])
        y = y.view(-1, y.shape[-1])
        assert self._single_lora_mask is not None
        adapter_mask = self._single_lora_mask[: x.size(0)]

        if len(lora_b_stacked) == 1:
            a_mat = lora_a_stacked[0][0, 0]
            b_mat = lora_b_stacked[0][0, 0, : output_slices[0]].transpose(0, 1)
        else:
            assert packed_lora_a is not None and packed_lora_b is not None
            a_mat = packed_lora_a[0, 0]
            b_mat = packed_lora_b[0, 0]

        x_in = x if x.dtype == a_mat.dtype else x.to(a_mat.dtype)
        shrink = torch.matmul(x_in, a_mat.transpose(0, 1)).mul_(adapter_mask)
        if scale != 1.0:
            shrink.mul_(scale)
        delta = torch.matmul(shrink, b_mat)
        if add_inputs:
            y.add_(delta)
        else:
            y.copy_(delta)

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
        LoRA slot for ``(lora_id, expert_id)``, so we fold both into a single
        gather index into a ``[max_loras * num_experts, ...]`` view of the
        per-(lora, expert) weight stacks:
            combined_idx[row] = lora_id[row] * num_experts + expert_id[row]
        or -1 when the row has no active adapter, mirroring the -1 sentinel
        ``PunicaWrapperBase.token_lora_indices`` already uses. When
        ``enable_moe_shared_loras=True`` collapses ``w13`` ``lora_A`` or
        ``w2`` ``lora_B`` to a single shared slot (``shape[1] == 1``), those
        shared factors index solely by ``lora_id[row]`` instead of multiplying
        by ``num_experts``. bgmv_shrink/bgmv_expand skip any row whose index
        is negative (leaving the zero-initialized shrink buffer / unmodified
        ``y`` in place), so inactive rows get a zero delta for free -- no
        Python-level branching needed.
        """
        del sorted_token_ids, num_tokens_post_padded, max_lora_rank
        del shrink_config, expand_config, fully_sharded
        assert top_k_num == 1, "Ascend MoE LoRA v1 expects pre-expanded rows (top_k_num=1)."
        if not lora_a_stacked or not lora_b_stacked:
            return
        assert len(lora_a_stacked) == len(lora_b_stacked)
        if token_lora_mapping is None:
            token_lora_mapping = self.token_lora_indices

        x2d = x.view(-1, x.shape[-1])
        y2d = y.view(-1, y.shape[-1])
        expert_idx = expert_ids.view(-1).to(torch.long)
        num_experts = max(max(a.shape[1], b.shape[1]) for a, b in zip(lora_a_stacked, lora_b_stacked, strict=True))
        for a, b in zip(lora_a_stacked, lora_b_stacked, strict=True):
            assert a.shape[1] in (1, num_experts) and b.shape[1] in (1, num_experts), (
                f"Unexpected MoE LoRA expert dimension: a.shape[1]={a.shape[1]}, "
                f"b.shape[1]={b.shape[1]}, num_experts={num_experts}."
            )

        num_slots = adapter_enabled.numel()
        if num_slots == 0:
            return
        lora_mapping = token_lora_mapping.to(torch.long)
        lora_idx_safe = lora_mapping.clamp(min=0, max=num_slots - 1)
        valid_slot = (lora_mapping >= 0) & (lora_mapping < num_slots)
        valid_expert = (expert_idx >= 0) & (expert_idx < num_experts)
        enabled = valid_slot & valid_expert & adapter_enabled[lora_idx_safe].bool()
        inactive = torch.full_like(lora_mapping, -1)
        per_expert_idx = torch.where(enabled, lora_idx_safe * num_experts + expert_idx, inactive).contiguous()
        has_shared_factor = num_experts > 1 and any(
            a.shape[1] == 1 or b.shape[1] == 1 for a, b in zip(lora_a_stacked, lora_b_stacked, strict=True)
        )
        shared_idx = torch.where(enabled, lora_idx_safe, inactive).contiguous() if has_shared_factor else per_expert_idx

        # bgmv_shrink writes fp32 (its Y_T); bgmv_expand reads fp32 (its X_T),
        # so the shrink buffer is fp32.
        rank = lora_a_stacked[0].shape[-2]
        shrink_out = torch.zeros((x2d.shape[0], rank), dtype=torch.float32, device=x2d.device)

        cur_offset = offset
        for a, b in zip(lora_a_stacked, lora_b_stacked, strict=True):
            # lora_a_stacked[s]/lora_b_stacked[s]: [max_loras, num_experts_or_1, rank/*, *].
            # Flattening the leading two dims turns "gather by (lora, expert)"
            # into "the plain per-row gather" to reuse bgmv_shrink/bgmv_expand.
            out_size = b.shape[-2]
            a_flat = a.view(-1, a.shape[-2], a.shape[-1])
            b_flat = b.view(-1, out_size, b.shape[-1])
            shrink_idx = shared_idx if a.shape[1] == 1 else per_expert_idx
            expand_idx = shared_idx if b.shape[1] == 1 else per_expert_idx

            x_in = x2d if not a_flat.is_floating_point() or x2d.dtype == a_flat.dtype else x2d.to(a_flat.dtype)
            self.bgmv_shrink(x_in, a_flat, shrink_out, shrink_idx, 1.0)

            delta = shrink_out
            if mul_routed_weight and topk_weights is not None:
                delta = shrink_out * topk_weights.view(-1, 1)

            def _run_expand_slice(
                out: torch.Tensor,
                inp: torch.Tensor,
                slice_off: int,
                add_in: bool,
                *,
                b_w: torch.Tensor = b_flat,
                e_idx: torch.Tensor = expand_idx,
                slice_len: int = out_size,
            ) -> None:
                self.bgmv_expand_slice(
                    inp,
                    b_w,
                    out,
                    e_idx,
                    slice_off,
                    slice_len,
                    add_inputs=add_in,
                )

            _expand_with_dtype_guard(
                y2d,
                delta,
                b_flat,
                cur_offset,
                out_size,
                True,
                _run_expand_slice,
            )
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

        indices = torch.narrow(self._sampler_indices, 0, 0, x.size(0))

        if y.size(1) < r:
            # Ascend bgmv_expand requires hidden_out >= hidden_in (LoRA rank).
            # The sequence-classification LoRA head (#53555) can have
            # num_labels < rank, so use the matmul fallback.
            self._add_lora_logits_matmul(y, x, lora_a_stacked, lora_b_stacked, scale, indices)
            y = y.view_as(y_org)
            return

        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=torch.float32, device=x.device)

        x_in = x
        if lora_a_stacked.is_floating_point() and x.dtype != lora_a_stacked.dtype:
            x_in = x.to(lora_a_stacked.dtype)
        self.bgmv_shrink(x_in, lora_a_stacked, buffer, indices, scale)
        _expand_with_dtype_guard(
            y,
            buffer,
            lora_b_stacked,
            0,
            y.size(1),
            True,
            lambda out, inp, _offset, add_in: self.bgmv_expand(inp, lora_b_stacked, out, indices, add_inputs=add_in),
        )

        y = y.view_as(y_org)

    def _add_lora_logits_matmul(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale: float,
        indices: torch.Tensor,
    ) -> None:
        """Per-adapter matmul fallback for add_lora_logits (see call site)."""
        active = indices >= 0
        idx = indices.clamp(min=0)
        lora_a = lora_a_stacked[idx]
        lora_b = lora_b_stacked[idx]
        if lora_a.ndim == 4:
            lora_a = lora_a[:, 0]
        if lora_b.ndim == 4:
            lora_b = lora_b[:, 0]

        delta = torch.bmm(x.to(torch.float32).unsqueeze(1), lora_a.to(torch.float32).transpose(1, 2)).squeeze(1)
        delta = torch.bmm(delta.unsqueeze(1), lora_b.to(torch.float32).transpose(1, 2)).squeeze(1)
        delta.mul_(scale)
        delta.masked_fill_(~active.unsqueeze(1), 0)
        y.add_(delta.to(y.dtype))

    def apply_lora_full_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        weight_stacked: torch.Tensor,
        bias_stacked: torch.Tensor,
        module_enabled: torch.Tensor,
    ) -> None:
        """Apply request-routed full linear weights to selected rows.

        Mirrors vLLM's CPU implementation (the stacked weight carries the full
        linear weight, so rank == out_features). The Ascend BGMV shrink kernel
        is intentionally avoided here: it requires half/bf16 inputs and
        ``hidden_in > hidden_out``, neither of which holds for a classification
        head, and it writes in the input dtype rather than fp32.
        """
        indices = self.sampler_indices
        selected = weight_stacked[indices.clamp_min(0)]
        if selected.dim() == 4:
            selected = selected.squeeze(1)
        adapter_y = torch.einsum("bi,boi->bo", x.to(torch.float32), selected.to(torch.float32))
        result = self._select_full_linear_output(y, adapter_y, bias_stacked, module_enabled)
        y.copy_(result)
