# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.utils.torch_utils import PIN_MEMORY

from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

# Switch dense LoRA from per-token BGMV to Group GEMM only when enough tokens
# amortize weight gathering and the Group GEMM launch overhead.
GMM_TOKEN_THRESHOLD = 1024


def _dispatch_lora_shrink(
    y: list[torch.Tensor],
    x: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    token_lora_indices: torch.Tensor,
    scale: float,
    use_gmm: torch.Tensor,
    no_lora: torch.Tensor,
) -> None:
    """Call the external Group GEMM/BGMV shrink dispatcher."""

    torch.ops._C_ascend.add_lora_shrink(
        y,
        x,
        lora_a_stacked,
        lora_indices,
        seq_lengths,
        token_lora_indices,
        scale,
        use_gmm,
        no_lora,
    )


def _dispatch_lora_expand(
    y: torch.Tensor,
    x: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    lora_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
    token_lora_indices: torch.Tensor,
    output_slices: list[int],
    offset_start: int,
    add_inputs: bool,
    use_gmm: torch.Tensor,
    no_lora: torch.Tensor,
) -> None:
    """Call the external Group GEMM/BGMV expand dispatcher."""

    torch.ops._C_ascend.add_lora_expand(
        y,
        x,
        lora_b_stacked,
        lora_indices,
        seq_lengths,
        token_lora_indices,
        output_slices,
        offset_start,
        add_inputs,
        use_gmm,
        no_lora,
    )


@dataclass(frozen=True)
class _HostSGMVMetadata:
    """Run-length encoded LoRA routing."""

    seq_start_locs: tuple[int, ...]
    seq_lengths: tuple[int, ...]
    lora_indices: tuple[int, ...]
    token_nums: int
    max_seq_length: int
    no_lora: bool

    @property
    def batches(self) -> int:
        return len(self.seq_lengths)


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
        self._init_prefill_sgmv_metadata_buffers(max_batches)
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
        self._init_group_gemm_dispatch_flags()

    def _init_group_gemm_dispatch_flags(self) -> None:
        # CPU tensors are live inputs to the opaque custom op, so torch.compile
        # cannot bake one warm-up batch's GMM/BGMV choice into the graph.
        self._use_gmm_shrink_cpu = torch.tensor(False, dtype=torch.bool)
        self._use_gmm_expand_cpu = torch.tensor(False, dtype=torch.bool)
        self._no_lora_cpu = torch.tensor(True, dtype=torch.bool)

    def _init_prefill_sgmv_metadata_buffers(self, max_batches: int) -> None:
        # PunicaWrapperBase allocates these as three independent tensors. Use
        # views of one stable allocation so the host RLE can be transferred by
        # one H2D copy without temporary device tensors.
        shape = (3, max_batches)
        self._prefill_sgmv_metadata_buffer = torch.empty(
            shape,
            dtype=torch.long,
            device=self.device,
        )
        self._prefill_sgmv_metadata_cpu = torch.empty(
            shape,
            dtype=torch.long,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        self._prefill_sgmv_metadata_cpu_array = self._prefill_sgmv_metadata_cpu.numpy()
        self._seq_start_locs = self._prefill_sgmv_metadata_buffer[0]
        self._seq_lengths = self._prefill_sgmv_metadata_buffer[1]
        self._lora_indices_per_batch = self._prefill_sgmv_metadata_buffer[2]
        self._host_sgmv_metadata = self._encode_sgmv_metadata(())
        self._sgmv_max_batches = max_batches

    @staticmethod
    def _encode_sgmv_metadata(lora_indices: Iterable[int]) -> _HostSGMVMetadata:
        """Run-length encode token-to-LoRA routing in one host pass."""

        seq_start_locs: list[int] = []
        seq_lengths: list[int] = []
        lora_indices_per_batch: list[int] = []
        token_nums = 0
        no_lora = True
        for token_offset, lora_index in enumerate(lora_indices):
            token_nums = token_offset + 1
            no_lora = no_lora and lora_index < 0
            if lora_indices_per_batch and lora_index == lora_indices_per_batch[-1]:
                seq_lengths[-1] += 1
            else:
                seq_start_locs.append(token_offset)
                seq_lengths.append(1)
                lora_indices_per_batch.append(lora_index)

        return _HostSGMVMetadata(
            seq_start_locs=tuple(seq_start_locs),
            seq_lengths=tuple(seq_lengths),
            lora_indices=tuple(lora_indices_per_batch),
            token_nums=token_nums,
            max_seq_length=max(seq_lengths, default=0),
            no_lora=no_lora,
        )

    @staticmethod
    def _write_sgmv_metadata(host_buffer, metadata: _HostSGMVMetadata) -> None:
        """Write RLE values into a fixed-size host buffer without tensors."""

        host_buffer[0].fill(0)
        host_buffer[1].fill(0)
        host_buffer[2].fill(-1)
        if metadata.batches:
            host_buffer[0, : metadata.batches] = metadata.seq_start_locs
            host_buffer[1, : metadata.batches] = metadata.seq_lengths
            host_buffer[2, : metadata.batches] = metadata.lora_indices

    @staticmethod
    def _validate_sgmv_batches(metadata: _HostSGMVMetadata, max_batches: int, name: str) -> None:
        if metadata.batches > max_batches:
            raise ValueError(
                f"{name} SGMV routing contains more consecutive LoRA groups "
                f"than the configured maximum batch size: {metadata.batches} > {max_batches}."
            )

    def update_metadata(
        self,
        mapping,
        lora_index_to_id,
        max_loras,
        vocab_size,
        **kwargs,
    ) -> None:
        index_mapping = mapping.index_mapping
        lora_id_to_index = {lora_id: index for index, lora_id in enumerate(lora_index_to_id) if lora_id is not None}
        self._host_sgmv_metadata = self._encode_sgmv_metadata(
            lora_id_to_index.get(lora_id, -1) if lora_id > 0 else -1 for lora_id in index_mapping
        )
        super().update_metadata(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            **kwargs,
        )
        # Keep only the active-LoRA count on the host. The concrete slot and
        # base-token mask stay in device tensors so ACLGraph replay can switch
        # requests without fixing batch-local routing values.
        loaded_lora_ids = set(lora_id for lora_id in lora_index_to_id if lora_id is not None)
        self.num_active_moe_loras = len(
            set(lora_id for lora_id in index_mapping if lora_id > 0 and lora_id in loaded_lora_ids)
        )
        # PunicaWrapperBase computes this only for prefill. Decode must also
        # choose between the active-LoRA and base-only quantized MoE paths.
        self.no_lora = not any(lora_id > 0 for lora_id in mapping.index_mapping)
        if not hasattr(self, "_use_gmm_shrink_cpu"):
            self._init_group_gemm_dispatch_flags()
        use_gmm = self._host_sgmv_metadata.token_nums > GMM_TOKEN_THRESHOLD
        self._use_gmm_shrink_cpu.fill_(use_gmm)
        self._use_gmm_expand_cpu.fill_(use_gmm)
        self._no_lora_cpu.fill_(self.no_lora)

    def _update_prefill_metadata(self, token_lora_tensor: torch.Tensor) -> None:
        """Reuse host RLE and avoid device unique/item synchronization."""

        del token_lora_tensor
        metadata = self._host_sgmv_metadata
        self._validate_sgmv_batches(metadata, self._sgmv_max_batches, "Punica prefill")
        self._write_sgmv_metadata(self._prefill_sgmv_metadata_cpu_array, metadata)
        self._prefill_sgmv_metadata_buffer.copy_(
            self._prefill_sgmv_metadata_cpu,
            non_blocking=True,
        )
        self.batch_size = metadata.batches
        self.max_length = metadata.max_seq_length
        self.token_nums = metadata.token_nums
        self.no_lora = metadata.no_lora

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
        y_views = [y[slice_idx].view(-1, y[slice_idx].shape[-1]) for slice_idx in range(len(lora_a_stacked))]
        _, seq_lengths, lora_indices, _, _, _ = self.prefill_metadata
        _dispatch_lora_shrink(
            y_views,
            x,
            list(lora_a_stacked),
            lora_indices,
            seq_lengths,
            self._get_token_lora_indices(x),
            scale,
            self._use_gmm_shrink_cpu,
            self._no_lora_cpu,
        )

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
        x_views = [x[slice_idx].view(-1, x[slice_idx].shape[-1]) for slice_idx in range(len(lora_b_stacked))]
        # aclnnGroupedMatmulV4 forbids a transposed X when group_type=0.
        x_views = [x_view.contiguous() for x_view in x_views]
        _, seq_lengths, lora_indices, _, _, _ = self.prefill_metadata
        _dispatch_lora_expand(
            y,
            x_views,
            list(lora_b_stacked),
            lora_indices,
            seq_lengths,
            self._get_token_lora_indices(x_views[0]),
            list(output_slices),
            offset_start,
            add_inputs,
            self._use_gmm_expand_cpu,
            self._no_lora_cpu,
        )
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
