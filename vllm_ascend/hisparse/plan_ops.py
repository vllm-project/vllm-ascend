# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend950 SparseKvPlan adapter for HiSparse decode.

The operator owns one persistent resident-cache state per index group.  This
module binds graph-capture-friendly state buffers to the Python API exported by
the CANN ops-transformer torch extension.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.hisparse.runtime import HiSparseRuntime

from vllm_ascend.device.device_config import AscendDeviceType, get_ascend_device_type

_SPARSE_KV_PLAN_PACKAGES = (
    "cann_ops_transformer_sparsekvplan",
    "cann_ops_transformer_qliorder",
    "cann_ops_transformer",
)


@lru_cache(maxsize=1)
def _has_sparse_kv_plan_package(package: str) -> bool:
    """Whether an installed ops-transformer distribution may provide Plan."""
    return find_spec(package) is not None


def sparse_kv_plan_op() -> Callable[..., None] | None:
    """Find the SparseKvPlan API in the installed CANN operator package."""
    for package in _SPARSE_KV_PLAN_PACKAGES:
        if not _has_sparse_kv_plan_package(package):
            continue

        module_name = f"{package}.ops"
        expected_missing = (package, module_name)
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name in expected_missing:
                continue
            raise

        operation = getattr(module, "sparse_kv_plan", None)
        if callable(operation):
            return operation
    return None


def ordered_quant_lightning_indexer_op() -> Callable[..., Any] | None:
    """Find the ordered-QLI API in the installed CANN operator package."""
    # Both Plan and ordered-QLI ops ship in the same CANN distributions;
    # sparsekvplan is preferred, the others are legacy/generic fallbacks.
    for package in _SPARSE_KV_PLAN_PACKAGES:
        if not _has_sparse_kv_plan_package(package):
            continue

        module_name = f"{package}.ops.attention.quant_lightning_indexer_ordered"
        expected_missing = (
            package,
            f"{package}.ops",
            f"{package}.ops.attention",
            module_name,
        )
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name in expected_missing:
                continue
            raise

        operation = getattr(module, "quant_lightning_indexer_ordered", None)
        if callable(operation):
            return operation
    return None


def _sparse_kv_plan_dependencies_available() -> bool:
    """Require both the resolver and its ordered-QLI producer."""
    return sparse_kv_plan_op() is not None and ordered_quant_lightning_indexer_op() is not None


class SparseKvPlanWorkspace:
    """Persistent input, output and scratch tensors for one index group.

    HiSparse A5 geometry maps MTP query width 1/2/4 to a 4096/4096/8192 row
    resident buffer. Keeping the buffers here decouples the operator's int32
    state from Core's int16 upstream LRU workspace.
    """

    def __init__(
        self,
        max_rows: int,
        top_k: int,
        group_width: int,
        block_size: int,
        device: torch.device,
        *,
        hot_block_table: torch.Tensor,
    ) -> None:
        if group_width not in (1, 2, 4):
            raise ValueError(f"SparseKvPlan supports 1, 2 or 4 query rows, got {group_width}")
        capacity = 8192 if group_width == 4 else 4096
        if max_rows <= 0 or max_rows % group_width or top_k != 2048:
            raise ValueError("SparseKvPlan requires positive grouped rows and TopK 2048")
        if block_size <= 0 or block_size > capacity or capacity % block_size:
            raise ValueError(f"SparseKvPlan requires a block divisor of {capacity}")
        self.op = sparse_kv_plan_op()
        if self.op is None:
            raise RuntimeError("SparseKvPlan torch extension is not installed")
        self.max_rows = max_rows
        self.top_k = top_k
        self.group_width = group_width
        self.max_requests = max_rows // group_width
        self.capacity = capacity
        self.block_size = block_size
        self.device = device
        # Capture the immutable Hot block table as a contiguous device tensor so
        # graph replay never performs an eager reshape/copy.
        self._hot_block_table = hot_block_table.contiguous()

        def empty_i32(*shape: int) -> torch.Tensor:
            return torch.empty(shape, dtype=torch.int32, device=device)

        self.last_req_ids = torch.full((max_rows,), -1, dtype=torch.int64, device=device)
        self.slot_to_token = torch.full((max_rows, capacity), -1, dtype=torch.int32, device=device)
        self.lru_slots = torch.arange(capacity, dtype=torch.int32, device=device).repeat(max_rows, 1)
        self.current_slots = torch.full((max_rows, top_k), -1, dtype=torch.int32, device=device)
        self.miss_count = empty_i32(max_rows)
        self.miss_tokens = torch.full((max_rows, top_k), -1, dtype=torch.int32, device=device)
        self.miss_slots = torch.full((max_rows, top_k), -1, dtype=torch.int32, device=device)
        self.plan_workspace = torch.full((max_rows * 1024,), 2**31 - 1, dtype=torch.int32, device=device)
        self.row_modes = empty_i32(max_rows)
        self.resident_rows = torch.full((max_rows,), -1, dtype=torch.int32, device=device)
        self.resident_block_table = torch.full((max_rows, capacity // block_size), -1, dtype=torch.int32, device=device)

        # Avoid NPU allocations while a graph is capturing or replaying.
        self._physical_scratch = empty_i32(max_rows, top_k)
        self._block_index_i32 = empty_i32(max_rows, top_k)
        self._block_index_i64 = torch.empty((max_rows, top_k), dtype=torch.int64, device=device)
        self._state_rows_i64 = torch.empty((max_rows,), dtype=torch.int64, device=device)
        self._row_ids = torch.arange(max_rows, dtype=torch.int32, device=device)
        self._active_mask = torch.empty((max_rows,), dtype=torch.bool, device=device)
        self._attention_valid = torch.empty((max_rows, top_k), dtype=torch.bool, device=device)
        self._attention_capacity_valid = torch.empty((max_rows, top_k), dtype=torch.bool, device=device)

    @classmethod
    def from_runtime(cls, runtime: HiSparseRuntime) -> SparseKvPlanWorkspace:
        top_k = runtime.index_group.shared_topk.device_topk_rows.shape[1]
        group_width = runtime.max_union_rows // top_k
        return cls(
            runtime.max_swap_rows,
            top_k,
            group_width,
            runtime.hot.block_size,
            runtime.device,
            hot_block_table=runtime.hot_block_table,
        )

    @staticmethod
    def supports(runtime: HiSparseRuntime) -> bool:
        # SparseKvPlan is an Arch35/A5 kernel. Keep the A3 runtime on the
        # upstream resolver instead of loading an incompatible custom op.
        if get_ascend_device_type() != AscendDeviceType.A5:
            return False
        if not _sparse_kv_plan_dependencies_available():
            return False
        if not getattr(runtime, "_ascend_sparse_kv_plan_eligible", False):
            return False
        top_k = runtime.index_group.shared_topk.device_topk_rows.shape[1]
        if top_k != 2048 or runtime.max_union_rows % top_k:
            return False
        group_width = runtime.max_union_rows // top_k
        if group_width not in (1, 2, 4) or runtime.max_swap_rows % group_width:
            return False
        expected = 8192 if group_width == 4 else 4096
        if runtime.region_stride != expected:
            return False
        # `hot` only exists after bind_hot_cache(); supports() is also called
        # during memory profiling before binding, so defer the page-geometry
        # check until the hot cache is bound.
        hot = getattr(runtime, "hot", None)
        if hot is None:
            return True
        block_size = hot.block_size
        return block_size > 0 and block_size <= expected and expected % block_size == 0

    def execute(
        self,
        topk_indices: torch.Tensor,
        request_ids: torch.Tensor,
        stable_prefix_lens: torch.Tensor,
        visible_seq_lens: torch.Tensor,
        token_to_req: torch.Tensor,
        active_rows: torch.Tensor,
        *,
        max_token: int,
    ) -> None:
        """Run one plan with graph-captured metadata inputs."""
        if max_token <= 0:
            raise ValueError("SparseKvPlan requires a positive max-token bound")
        rows = topk_indices.shape[0]
        requests = rows // self.group_width
        if rows % self.group_width or rows > self.max_rows:
            raise ValueError("SparseKvPlan input is not a complete query-row group prefix")
        if (
            request_ids.shape[0] != requests
            or stable_prefix_lens.shape[0] != rows
            or visible_seq_lens.shape[0] != rows
            or token_to_req.shape[0] != rows
            or active_rows.numel() != 1
        ):
            raise ValueError("SparseKvPlan metadata does not match its TopK rows")
        self.op(
            request_ids=request_ids,
            topk_indices=topk_indices,
            stable_prefix_lens=stable_prefix_lens,
            visible_seq_lens=visible_seq_lens,
            token_to_req=token_to_req,
            active_rows=active_rows.reshape(1),
            last_req_ids=self.last_req_ids[:rows],
            slot_to_token=self.slot_to_token[:rows],
            lru_slots=self.lru_slots[:rows],
            current_slots=self.current_slots[:rows],
            miss_count=self.miss_count[:rows],
            miss_tokens=self.miss_tokens[:rows],
            miss_slots=self.miss_slots[:rows],
            plan_workspace=self.plan_workspace[: rows * 1024],
            row_modes=self.row_modes[:rows],
            resident_rows=self.resident_rows[:rows],
            resident_block_table=self.resident_block_table[:rows],
            capacity=self.capacity,
            max_token=max_token,
            block_size=self.block_size,
        )

    def _map_slots_to_hot_rows(
        self,
        runtime: HiSparseRuntime,
        shared_rows: slice,
        slots: torch.Tensor,
        state_rows: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Map slots in a request region to HiSparse Hot-cache physical rows."""
        table = self._hot_block_table
        table_columns = table.shape[1]
        flat_table = table.reshape(-1)
        state_rows_i64 = self._state_rows_i64[shared_rows]
        block_indices_i32 = self._block_index_i32[shared_rows]
        block_indices_i64 = self._block_index_i64[shared_rows]

        state_rows_i64.copy_(state_rows[shared_rows])
        torch.div(slots, self.block_size, rounding_mode="floor", out=block_indices_i32)
        block_indices_i64.copy_(block_indices_i32)
        block_indices_i64.mul_(table_columns)
        block_indices_i64.add_(state_rows_i64)
        torch.take(flat_table, block_indices_i64, out=output)
        output.mul_(self.block_size)
        torch.sub(slots, output, out=block_indices_i32)
        output.add_(block_indices_i32)
        output.masked_fill_(slots < 0, -1)

    def _map_hot_rows_for_attention(
        self,
        shared_rows: slice,
        hot_rows: torch.Tensor,
        output: torch.Tensor,
        *,
        attention_block_stride: int | None,
        attention_capacity: int,
    ) -> None:
        """Map virtual Hot rows to the native C8 attention storage layout."""
        if attention_block_stride is None or attention_capacity <= 0:
            output.copy_(hot_rows)
            return

        valid = self._attention_valid[shared_rows]
        capacity_valid = self._attention_capacity_valid[shared_rows]
        block_indices = self._block_index_i32[shared_rows]
        torch.ge(hot_rows, 0, out=valid)
        torch.lt(hot_rows, attention_capacity, out=capacity_valid)
        valid &= capacity_valid

        mapped_rows = self._block_index_i64[shared_rows]
        torch.div(hot_rows, self.block_size, rounding_mode="floor", out=mapped_rows)
        mapped_rows.mul_(attention_block_stride)
        torch.remainder(hot_rows, self.block_size, out=block_indices)
        mapped_rows.add_(block_indices)
        output.copy_(mapped_rows)
        valid.logical_not_()
        output.masked_fill_(valid, -1)

    def fill_shared_result(
        self,
        runtime: HiSparseRuntime,
        shared_rows: slice,
        logical_topk_indices: torch.Tensor,
        state_rows: torch.Tensor,
        active_rows: torch.Tensor,
        physical_topk_indices: torch.Tensor,
        valid_topk_counts: torch.Tensor | None,
    ) -> None:
        shared = runtime.index_group.shared_topk
        torch.ge(self._row_ids[shared_rows], active_rows.reshape(()), out=self._active_mask[shared_rows])
        inactive = self._active_mask[shared_rows]
        current = self.current_slots[shared_rows]
        virtual_rows = self._physical_scratch[shared_rows]
        self._map_slots_to_hot_rows(runtime, shared_rows, current, state_rows, virtual_rows)
        virtual_rows.masked_fill_(inactive.unsqueeze(1), -1)
        self._map_hot_rows_for_attention(
            shared_rows,
            virtual_rows,
            physical_topk_indices,
            attention_block_stride=runtime.cache_ops.attention_block_stride,
            attention_capacity=runtime.cache_ops.attention_capacity,
        )

        miss_slots = self.miss_slots[shared_rows]
        device_rows = self._physical_scratch[shared_rows]
        self._map_slots_to_hot_rows(runtime, shared_rows, miss_slots, state_rows, device_rows)
        shared.swap_device_physical_rows[shared_rows].copy_(device_rows)
        shared.swap_host_physical_rows[shared_rows].copy_(self.miss_tokens[shared_rows])
        shared.swap_counts[shared_rows].copy_(self.miss_count[shared_rows])
        shared.swap_counts[shared_rows].masked_fill_(inactive, 0)
        if valid_topk_counts is not None:
            torch.sum(logical_topk_indices >= 0, dim=1, dtype=torch.int32, out=valid_topk_counts)
            valid_topk_counts.masked_fill_(inactive, 0)


def _sparse_plan_swap_in(runtime: HiSparseRuntime, original, **kwargs: Any):
    """Resolve with SparseKvPlan, otherwise preserve upstream decode behavior."""
    plan = getattr(runtime.cache_ops, "plan_workspace", None)
    metadata = getattr(runtime, "_sparse_kv_plan_inputs", None)
    if not runtime.is_group_leader or plan is None or metadata is None:
        runtime._sparse_kv_plan_inputs = None
        return original(runtime, **kwargs)

    # Dummy/profile batches carry no real tokens. Skip planning and keep the
    # upstream decode behavior so warmup and graph capture stay valid.
    if getattr(metadata, "plan_max_token", 0) <= 0:
        runtime._sparse_kv_plan_inputs = None
        return original(runtime, **kwargs)

    from vllm.forward_context import in_piecewise_cudagraph
    from vllm.utils.torch_utils import current_stream

    topk_indices = kwargs["logical_topk_indices"]
    rows = topk_indices.shape[0]
    shared_rows = runtime._step_rows(rows)
    group = runtime.index_group
    physical = group.shared_topk.physical_topk_indices[shared_rows]
    valid_counts = kwargs.get("valid_counts_out")
    if kwargs.get("return_valid_counts", False):
        valid_counts = group.shared_topk.valid_topk_counts[shared_rows] if valid_counts is None else valid_counts
    else:
        valid_counts = None

    compute_stream = current_stream()
    if group.logical_topk_ready is not None and not in_piecewise_cudagraph():
        group.copy_stream.wait_event(group.logical_topk_ready)
    else:
        group.copy_stream.wait_stream(compute_stream)
    with group.copy_stream:
        plan.execute(
            topk_indices,
            metadata.plan_request_ids,
            metadata.plan_stable_prefix_lens,
            metadata.plan_visible_seq_lens,
            metadata.plan_token_to_req,
            metadata.plan_active_rows,
            max_token=metadata.plan_max_token,
        )
        # Mirror the upstream resolver's connector hit/miss accounting so the
        # SparseKvPlan path reports the same swap stats granularity.
        misses = plan.miss_count[:rows].sum(dtype=torch.int64)
        stats = group.swap_stats
        stats[1] += misses
        stats[0] += rows - misses
        plan.fill_shared_result(
            runtime,
            shared_rows,
            topk_indices,
            kwargs["req_id_per_token"],
            metadata.plan_active_rows,
            physical,
            valid_counts,
        )
        runtimes = [runtime]
        if kwargs["resident"].decode_batch:
            runtimes.extend(group.followers)
        for staged_runtime in runtimes:
            staged_runtime._swap_rows(shared_rows)
            if staged_runtime._layer_ready_event is None:
                staged_runtime._layer_ready_event = torch.Event()
            staged_runtime._layer_ready_event.record(group.copy_stream)
            staged_runtime._swap_staged = True

    runtime._sparse_kv_plan_inputs = None
    if valid_counts is not None:
        return physical, valid_counts
    return physical


def install_sparse_kv_plan() -> bool:
    """Install the A5 decode bridge once the ops-transformer APIs are found."""
    from vllm.v1.hisparse.runtime import HiSparseRuntime

    if getattr(HiSparseRuntime.swap_in, "_ascend_sparse_kv_plan", False):
        return True
    if not _sparse_kv_plan_dependencies_available():
        return False

    original = HiSparseRuntime.swap_in

    def swap_in(self: HiSparseRuntime, **kwargs: Any):
        return _sparse_plan_swap_in(self, original, **kwargs)

    swap_in.__dict__["_ascend_sparse_kv_plan"] = True
    HiSparseRuntime.swap_in = swap_in
    return True
