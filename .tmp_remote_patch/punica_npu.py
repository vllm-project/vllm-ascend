# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
import torch_npu
from torch_npu.npu.utils import get_cann_version
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_ascend.lora.utils import refresh_all_lora_classes
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


_MOE_LORA_PREFILL_VERSION_KEY = (
    "2.10.0+cpu",
    "2.10.0.post2",
    "9.0.0",
    223,
)
_MOE_LORA_PREFILL_OPS = (
    "moe_lora_prefill_route_allgather",
    "moe_lora_prefill_route_alltoall",
    "moe_lora_prefill_gather_by_perm",
    "moe_lora_prefill_scatter_add",
)
_MOE_LORA_PREFILL_SCHEMA_ARGS = {
    "moe_lora_prefill_route_allgather": (
        "Tensor x, Tensor expanded_row_idx, Tensor routed_topk_ids, "
        "Tensor token_lora_indices, Tensor adapter_enabled, "
        "Tensor($0! -> ) local_count, Tensor($1! -> ) core_prefix, "
        "Tensor($2! -> ) group_total, Tensor($3! -> ) group_start, "
        "Tensor($4! -> ) group_count_i64, Tensor($5! -> ) perm_record, "
        "Tensor($6! -> ) error_per_core, Tensor($7! -> ) route_error, "
        "Tensor($8! -> ) grouped_x, int top_k, int num_experts, "
        "int first_expert_idx) -> Tensor[]"
    ),
    "moe_lora_prefill_route_alltoall": (
        "Tensor x, Tensor expert_count, Tensor exchanged_lora_indices, "
        "Tensor adapter_enabled, Tensor($0! -> ) local_count, "
        "Tensor($1! -> ) core_prefix, Tensor($2! -> ) group_total, "
        "Tensor($3! -> ) group_start, Tensor($4! -> ) group_count_i64, "
        "Tensor($5! -> ) perm_record, Tensor($6! -> ) error_per_core, "
        "Tensor($7! -> ) route_error, Tensor($8! -> ) grouped_x) -> Tensor[]"
    ),
    "moe_lora_prefill_gather_by_perm": (
        "Tensor source, Tensor perm_record, Tensor($0! -> ) grouped_x) -> Tensor"
    ),
    "moe_lora_prefill_scatter_add": (
        "Tensor delta, Tensor perm_record, Tensor($0! -> ) y, "
        "int output_offset) -> Tensor"
    ),
}
_MOE_LORA_PREFILL_GMM_SCHEMA = (
    "npu::npu_grouped_matmul(Tensor[] x, Tensor[] weight, *, "
    "Tensor[]? bias=None, Tensor[]? scale=None, Tensor[]? offset=None, "
    "Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None, "
    "Tensor[]? per_token_scale=None, Tensor? group_list=None, "
    "Tensor[]? activation_input=None, Tensor[]? activation_quant_scale=None, "
    "Tensor[]? activation_quant_offset=None, int? split_item=0, "
    "int? group_type=None, int? group_list_type=0, int? act_type=0, "
    "int[]? tuning_config=None, int? output_dtype=None, int? x_dtype=None, "
    "int? weight_dtype=None, int? scale_dtype=None, "
    "int? per_token_scale_dtype=None) -> Tensor[]"
)


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
        bgmv_moe_w13 = None
        moe_lora_prefill_route_allgather = None
        moe_lora_prefill_route_alltoall = None
        moe_lora_prefill_gather_by_perm = None
        moe_lora_prefill_scatter_add = None
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
                bgmv_moe_w13,
                bgmv_shrink,
                moe_lora_prefill_gather_by_perm,
                moe_lora_prefill_route_allgather,
                moe_lora_prefill_route_alltoall,
                moe_lora_prefill_scatter_add,
                sgmv_expand,
                sgmv_expand_slice,
                sgmv_shrink,
            )
        self.bgmv_expand = bgmv_expand
        self.bgmv_expand_slice = bgmv_expand_slice
        self.bgmv_moe_w13 = bgmv_moe_w13
        self.bgmv_shrink = bgmv_shrink
        self.moe_lora_prefill_route_allgather = moe_lora_prefill_route_allgather
        self.moe_lora_prefill_route_alltoall = moe_lora_prefill_route_alltoall
        self.moe_lora_prefill_gather_by_perm = moe_lora_prefill_gather_by_perm
        self.moe_lora_prefill_scatter_add = moe_lora_prefill_scatter_add
        self.sgmv_expand = sgmv_expand
        self.sgmv_expand_slice = sgmv_expand_slice
        self.sgmv_shrink = sgmv_shrink
        # Per-shape scratch storage is safe to reuse because shrink and expand
        # execute on the same NPU stream. Keeping it on the wrapper also gives
        # ACLGraph stable addresses across layer captures.
        self._moe_w13_workspaces: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._moe_lora_routing_workspaces: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._moe_lora_prefill_workspaces: dict[tuple, dict[str, torch.Tensor]] = {}
        self._moe_lora_prefill_weight_views: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._moe_lora_prefill_capability: bool | None = None

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
        self.no_lora = not any(lora_id > 0 for lora_id in mapping.index_mapping)

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

    def _has_moe_lora_prefill_backend(self) -> bool:
        if self._moe_lora_prefill_capability is not None:
            return self._moe_lora_prefill_capability
        capable = False
        try:
            version_key = (
                torch.__version__,
                torch_npu.__version__,
                get_cann_version(),
                torch_npu.npu.get_soc_version(),
            )
            if version_key != _MOE_LORA_PREFILL_VERSION_KEY:
                self._moe_lora_prefill_capability = False
                return False
            if get_ascend_device_type() != AscendDeviceType.A2:
                self._moe_lora_prefill_capability = False
                return False
            if str(torch.ops.npu.npu_grouped_matmul.default._schema) != _MOE_LORA_PREFILL_GMM_SCHEMA:
                self._moe_lora_prefill_capability = False
                return False
            for name in _MOE_LORA_PREFILL_OPS:
                if not hasattr(torch.ops._C_ascend, name):
                    self._moe_lora_prefill_capability = False
                    return False
                schema = str(getattr(torch.ops._C_ascend, name).default._schema)
                if schema.split("(", 1)[1] != _MOE_LORA_PREFILL_SCHEMA_ARGS[name]:
                    self._moe_lora_prefill_capability = False
                    return False
            capable = all(
                op is not None
                for op in (
                    self.moe_lora_prefill_route_allgather,
                    self.moe_lora_prefill_route_alltoall,
                    self.moe_lora_prefill_gather_by_perm,
                    self.moe_lora_prefill_scatter_add,
                )
            )
        except (AttributeError, RuntimeError):
            capable = False
        self._moe_lora_prefill_capability = capable
        return capable

    @staticmethod
    def _moe_lora_prefill_weight_pair_supported(
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        num_adapters: int,
        num_experts: int,
        input_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        return (
            a.dim() == 4
            and b.dim() == 4
            and a.shape[0] == num_adapters
            and b.shape[0] == num_adapters
            and a.shape[1] == num_experts
            and b.shape[1] == num_experts
            and a.shape[-2] == 16
            and b.shape[-1] == 16
            and a.shape[-1] == input_width
            and b.shape[-2] % 16 == 0
            and a.dtype == b.dtype == dtype
            and a.device == b.device == device
            and a.is_contiguous()
            and b.is_contiguous()
            and a.storage_offset() == 0
            and b.storage_offset() == 0
        )

    @staticmethod
    def _moe_lora_prefill_key_certified(
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        w13_lora_a: tuple[torch.Tensor, ...],
        w13_lora_b: tuple[torch.Tensor, ...],
        w2_lora_a: tuple[torch.Tensor, ...],
        w2_lora_b: tuple[torch.Tensor, ...],
        num_adapters: int,
        num_experts: int,
        route_mode: str,
    ) -> bool:
        """Return whether Phase-4 precision/eager/graph gates certified this key."""
        if (
            x.shape[-1] != 4096
            or y.shape[-1] != 4096
            or num_adapters not in (1, 2, 4)
            or num_experts != 8
            or any(a.shape[-1] != 4096 for a in w13_lora_a)
            or any(b.shape[-2] != 2048 for b in w13_lora_b)
            or w2_lora_a[0].shape[-1] != 2048
            or w2_lora_b[0].shape[-2] != 4096
        ):
            return False
        certified_rows = {
            "alltoall": (1024, 2048, 4096),
            "allgather": (2048, 4096),
        }
        return x.shape[0] in certified_rows.get(route_mode, ())

    def _get_moe_lora_prefill_workspace(
        self,
        *,
        x: torch.Tensor,
        num_groups: int,
        max_width: int,
        route_mode: str,
    ) -> dict[str, torch.Tensor]:
        num_rows = x.shape[0]
        group_pitch = (num_groups + 7) // 8 * 8
        device_index = x.device.index
        if device_index is None:
            device_index = torch.npu.current_device()
        vector_cores = torch.npu.get_device_properties(device_index).vector_core_num
        key = (
            x.device,
            x.dtype,
            num_rows,
            num_groups,
            max_width,
            route_mode,
            vector_cores,
        )
        workspace = self._moe_lora_prefill_workspaces.get(key)
        if workspace is None:
            workspace = {
                "local_count": torch.empty(
                    (vector_cores, group_pitch), dtype=torch.int32, device=x.device
                ),
                "core_prefix": torch.empty(
                    (vector_cores, group_pitch), dtype=torch.int32, device=x.device
                ),
                "group_total": torch.empty(num_groups, dtype=torch.int32, device=x.device),
                "group_start": torch.empty(num_groups, dtype=torch.int32, device=x.device),
                "group_count": torch.empty(num_groups, dtype=torch.int64, device=x.device),
                "perm_record": torch.empty((num_rows, 8), dtype=torch.int32, device=x.device),
                "error_per_core": torch.empty(
                    (vector_cores, 8), dtype=torch.int32, device=x.device
                ),
                "route_error": torch.empty(8, dtype=torch.int32, device=x.device),
                "grouped_storage": torch.empty(
                    num_rows * max_width, dtype=x.dtype, device=x.device
                ),
            }
            self._moe_lora_prefill_workspaces[key] = workspace
        return workspace

    def prepare_moe_lora_prefill(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        w13_lora_a: tuple[torch.Tensor, ...],
        w13_lora_b: tuple[torch.Tensor, ...],
        w2_lora_a: tuple[torch.Tensor, ...],
        w2_lora_b: tuple[torch.Tensor, ...],
        adapter_enabled: torch.Tensor,
        route_mode: str,
        group_list_type: int,
        expanded_row_idx: torch.Tensor | None = None,
        routed_topk_ids: torch.Tensor | None = None,
        token_lora_indices: torch.Tensor | None = None,
        top_k: int = 1,
        first_expert_idx: int = 0,
        expert_count: torch.Tensor | None = None,
        exchanged_lora_indices: torch.Tensor | None = None,
        fully_sharded: bool = False,
        mul_routed_weight: bool = False,
    ) -> dict[str, object] | None:
        if not self.is_prefill or not self._has_moe_lora_prefill_backend():
            return None
        if fully_sharded or mul_routed_weight or group_list_type != 1:
            return None
        if x.dim() != 2 or y.dim() != 2 or x.shape[0] < 512 or x.shape[0] > 2**31 - 1:
            return None
        if (
            x.dtype not in (torch.float16, torch.bfloat16)
            or y.dtype != x.dtype
            or y.device != x.device
            or y.shape[0] != x.shape[0]
        ):
            return None
        if (
            not x.is_contiguous()
            or not y.is_contiguous()
            or x.storage_offset() != 0
            or y.storage_offset() != 0
            or x.shape[-1] % 16 != 0
        ):
            return None
        if len(w13_lora_a) != 2 or len(w13_lora_b) != 2:
            return None
        if len(w2_lora_a) != 1 or len(w2_lora_b) != 1:
            return None
        if (
            adapter_enabled.dtype not in (torch.bool, torch.int32, torch.int64)
            or adapter_enabled.device != x.device
            or adapter_enabled.dim() != 1
        ):
            return None
        if not adapter_enabled.is_contiguous() or adapter_enabled.storage_offset() != 0:
            return None
        num_adapters = adapter_enabled.numel()
        num_experts = w13_lora_a[0].shape[1]
        num_groups = num_adapters * num_experts
        if not 2 <= num_groups <= 1024:
            return None
        if not self._moe_lora_prefill_key_certified(
            x=x,
            y=y,
            w13_lora_a=w13_lora_a,
            w13_lora_b=w13_lora_b,
            w2_lora_a=w2_lora_a,
            w2_lora_b=w2_lora_b,
            num_adapters=num_adapters,
            num_experts=num_experts,
            route_mode=route_mode,
        ):
            return None
        max_width = max(x.shape[-1], w2_lora_a[0].shape[-1])
        weight_pairs = tuple(zip(w13_lora_a, w13_lora_b)) + tuple(zip(w2_lora_a, w2_lora_b))
        input_widths = (x.shape[-1], x.shape[-1], w2_lora_a[0].shape[-1])
        if any(
            not self._moe_lora_prefill_weight_pair_supported(
                a,
                b,
                num_adapters=num_adapters,
                num_experts=num_experts,
                input_width=input_width,
                dtype=x.dtype,
                device=x.device,
            )
            for (a, b), input_width in zip(weight_pairs, input_widths)
        ):
            return None
        if sum(b.shape[-2] for b in w13_lora_b) > y.shape[-1]:
            return None
        if w2_lora_b[0].shape[-2] > x.shape[-1]:
            return None

        if route_mode == "allgather":
            if (
                expanded_row_idx is None
                or routed_topk_ids is None
                or token_lora_indices is None
                or expanded_row_idx.device != x.device
                or routed_topk_ids.device != x.device
                or token_lora_indices.device != x.device
                or expanded_row_idx.dtype not in (torch.int32, torch.int64)
                or routed_topk_ids.dtype != expanded_row_idx.dtype
                or token_lora_indices.dtype != torch.int64
                or expanded_row_idx.dim() != 1
                or routed_topk_ids.shape != expanded_row_idx.shape
                or token_lora_indices.dim() != 1
                or top_k <= 0
                or expanded_row_idx.numel() % top_k != 0
                or token_lora_indices.numel() != expanded_row_idx.numel() // top_k
                or not expanded_row_idx.is_contiguous()
                or not routed_topk_ids.is_contiguous()
                or not token_lora_indices.is_contiguous()
                or expanded_row_idx.storage_offset() != 0
                or routed_topk_ids.storage_offset() != 0
                or token_lora_indices.storage_offset() != 0
            ):
                return None
        elif route_mode == "alltoall":
            if (
                expert_count is None
                or exchanged_lora_indices is None
                or expert_count.device != x.device
                or exchanged_lora_indices.device != x.device
                or expert_count.dtype not in (torch.int32, torch.int64)
                or expert_count.dim() != 1
                or expert_count.numel() != num_experts
                or exchanged_lora_indices.dtype != torch.int64
                or exchanged_lora_indices.dim() != 1
                or exchanged_lora_indices.numel() != x.shape[0]
                or not expert_count.is_contiguous()
                or not exchanged_lora_indices.is_contiguous()
                or expert_count.storage_offset() != 0
                or exchanged_lora_indices.storage_offset() != 0
            ):
                return None
        else:
            return None

        workspace = self._get_moe_lora_prefill_workspace(
            x=x, num_groups=num_groups, max_width=max_width, route_mode=route_mode
        )
        grouped_x = workspace["grouped_storage"][: x.numel()].view_as(x)
        route_workspace = (
            workspace["local_count"],
            workspace["core_prefix"],
            workspace["group_total"],
            workspace["group_start"],
            workspace["group_count"],
            workspace["perm_record"],
            workspace["error_per_core"],
            workspace["route_error"],
            grouped_x,
        )
        if route_mode == "allgather":
            self.moe_lora_prefill_route_allgather(
                x,
                expanded_row_idx,
                routed_topk_ids,
                token_lora_indices,
                adapter_enabled,
                route_workspace,
                top_k,
                num_experts,
                first_expert_idx,
            )
        else:
            self.moe_lora_prefill_route_alltoall(
                x,
                expert_count,
                exchanged_lora_indices,
                adapter_enabled,
                route_workspace,
            )
        return {
            "workspace": workspace,
            "num_rows": x.shape[0],
            "num_groups": num_groups,
            "num_experts": num_experts,
            "dtype": x.dtype,
            "device": x.device,
            "route_width": x.shape[-1],
        }

    def _get_moe_lora_prefill_weight_views(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            a.data_ptr(),
            b.data_ptr(),
            a.storage_offset(),
            b.storage_offset(),
            tuple(a.shape),
            tuple(b.shape),
            tuple(a.stride()),
            tuple(b.stride()),
            a.dtype,
            a.device,
        )
        views = self._moe_lora_prefill_weight_views.get(key)
        if views is None:
            groups = a.shape[0] * a.shape[1]
            a_view = a.view(groups, a.shape[-2], a.shape[-1]).transpose(-1, -2)
            b_view = b.view(groups, b.shape[-2], b.shape[-1]).transpose(-1, -2)
            if (
                a_view.data_ptr() != a.data_ptr()
                or b_view.data_ptr() != b.data_ptr()
                or a_view.storage_offset() != 0
                or b_view.storage_offset() != 0
            ):
                raise RuntimeError("MoE LoRA prefill weight transpose unexpectedly materialized")
            views = (a_view, b_view)
            self._moe_lora_prefill_weight_views[key] = views
        return views

    def apply_moe_lora_prefill(
        self,
        *,
        context: dict[str, object],
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_offset: int = 0,
        gather_input: bool = False,
    ) -> None:
        workspace = context["workspace"]
        assert isinstance(workspace, dict)
        grouped_storage = workspace["grouped_storage"]
        grouped_x = grouped_storage[: x.numel()].view_as(x)
        perm_record = workspace["perm_record"]
        group_count = workspace["group_count"]
        if gather_input:
            self.moe_lora_prefill_gather_by_perm(x, perm_record, grouped_x)
        else:
            if x.shape[-1] != context["route_width"]:
                raise RuntimeError("W13 grouped input does not match routed input width")
        cur_offset = output_offset
        for a, b in zip(lora_a_stacked, lora_b_stacked):
            a_view, b_view = self._get_moe_lora_prefill_weight_views(a, b)
            shrink_out = torch_npu.npu_grouped_matmul(
                x=[grouped_x],
                weight=[a_view],
                bias=None,
                group_list=group_count,
                split_item=2,
                group_type=0,
                group_list_type=1,
                output_dtype=None,
            )[0]
            if shrink_out.dtype != x.dtype or shrink_out.shape[-1] != 16:
                raise RuntimeError("built-in MoE LoRA shrink violated the certified T output ABI")
            delta = torch_npu.npu_grouped_matmul(
                x=[shrink_out],
                weight=[b_view],
                bias=None,
                group_list=group_count,
                split_item=2,
                group_type=0,
                group_list_type=1,
                output_dtype=None,
            )[0]
            if delta.dtype != x.dtype:
                raise RuntimeError("built-in MoE LoRA expand violated the certified T output ABI")
            self.moe_lora_prefill_scatter_add(delta, perm_record, y, cur_offset)
            cur_offset += b.shape[-2]

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
        combined_indices: torch.Tensor | None = None,
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

        if combined_indices is None:
            lora_idx_safe = token_lora_mapping.clamp(min=0)
            enabled = (token_lora_mapping >= 0) & adapter_enabled[lora_idx_safe].bool()
            combined_idx = torch.where(
                enabled,
                lora_idx_safe * num_experts + expert_idx,
                torch.full_like(token_lora_mapping, -1),
            ).contiguous()
        else:
            combined_idx = combined_indices.view(-1)

        fused_w13 = (
            getattr(self, "bgmv_moe_w13", None) is not None
            and get_ascend_device_type() == AscendDeviceType.A2
            and (not self.is_prefill or x2d.shape[0] <= 384)
            and len(lora_a_stacked) == 2
            and len(lora_b_stacked) == 2
            and not fully_sharded
            and not mul_routed_weight
            and lora_a_stacked[0].shape[-2] == 16
            and lora_a_stacked[1].shape == lora_a_stacked[0].shape
            and lora_a_stacked[0].shape[-1] <= 4096
            and lora_a_stacked[0].shape[-1] % 16 == 0
            and lora_b_stacked[0].shape[-1] == 16
            and lora_b_stacked[1].shape == lora_b_stacked[0].shape
            and lora_b_stacked[0].shape[-2] % 512 == 0
            and offset + 2 * lora_b_stacked[0].shape[-2] <= y2d.shape[-1]
        )
        if fused_w13:
            a0 = lora_a_stacked[0].view(-1, 16, x2d.shape[-1])
            a1 = lora_a_stacked[1].view(-1, 16, x2d.shape[-1])
            output_slice = lora_b_stacked[0].shape[-2]
            b0 = lora_b_stacked[0].view(-1, output_slice, 16)
            b1 = lora_b_stacked[1].view(-1, output_slice, 16)
            workspace_key = (x2d.device, x2d.shape[0])
            workspaces = getattr(self, "_moe_w13_workspaces", None)
            if workspaces is None:
                workspaces = self._moe_w13_workspaces = {}
            workspace = workspaces.get(workspace_key)
            if workspace is None:
                workspace = torch.empty((2, x2d.shape[0], 16), dtype=torch.float32, device=x2d.device)
                workspaces[workspace_key] = workspace
            self.bgmv_moe_w13(
                x2d,
                a0,
                a1,
                b0,
                b1,
                combined_idx,
                workspace,
                y2d,
                offset,
                1.0,
            )
            return

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
