# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DSA row-mode decode 的 eager/graph 共享物理镜像运行时。

``DSAInputBatchState`` 是本轮请求语义真源，``dsa_graph_gate.py`` 只负责图
准入策略；本文件则是二者之后的物理镜像层。它提供
``DSARowModeBufferOwner`` 和 ``DSARowModeRuntimeMixin``，把同一份固定容量
CPU/NPU 存储适配成两类非 owning view：eager 使用 active-prefix 并支持
request-major MTP query，FULL graph 仅接受 single-token captured-prefix，
并把尾部显式标记为 PAD。图路径额外负责 capture dummy、
每层 LIDU 输出 backing storage，以及 replay 前把真实 row 元数据
刷入固定地址 prefix；不同 capture size 只持有同一最大 owner 的 view。

DSASparseBase 仍属于算法核心基类，保留在 dsa_sparse.py 中。decode 新满块
dump 的 src/dst 固定列也由这里统一持有；它们是压紧的 copy-job prefix，
不是 request-row 对齐字段。eager 无 job 时跳过算子，graph 则保留固定节点并
用 ``src=0,dst=-1`` 填充空转尾部。这里不重新解释 stage/budget/tail，也不
进行 DRAM block 分配。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.dsa_sparse.dsa_forward_batch import (
    DSAForwardLayerHookPlan,
    DSAForwardRowModeDecodeBatch,
    DSAFullBlockDumpBatch,
    DSALightningIndexerUpdateBuffers,
)
from vllm_ascend.dsa_sparse.dsa_hot_kv_store_core import BlockType
from vllm_ascend.dsa_sparse.dsa_input_batch_state import DSAInputBatchState
from vllm_ascend.dsa_sparse.dsa_types import (
    DSA_LIDU_OUTPUT_CAPACITY,
    KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID,
    DSADecodeRowMode,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch

logger = init_logger(__name__)


_ROW_META_RESIDENT_POOL = 0
_ROW_META_MODE = 1
_ROW_META_DUMP_SRC_HBM_BLOCK = 2
_ROW_META_DUMP_DST_DRAM_BLOCK = 3
_ROW_META_FIELD_COUNT = 4
_INT32_ITEMSIZE = 4


def _row_mode_decode_lengths_are_valid(
    *,
    indexer_key_lens: np.ndarray,
    candidate_range_ends: np.ndarray,
    budget_lens: np.ndarray,
    sparse_mask: np.ndarray,
) -> bool:
    """Validate semantic lengths without rejecting short DENSE rows.

    ``candidate_range_ends`` describes only the full-block prefix exposed to sparse
    selection. A short DENSE request can legitimately keep its whole context
    in the current tail block, in which case this value is zero. The row is
    still a valid decode row because LIDU/SFA-Offload consume
    ``indexer_key_lens`` and KSC has ``miss_count=0`` for DENSE rows. Only
    SPARSE rows require a non-empty candidate prefix.
    """
    if (np.any(indexer_key_lens <= 0) or np.any(candidate_range_ends < 0)
            or np.any(budget_lens <= 0)):
        return False
    return not np.any(candidate_range_ends[sparse_mask] <= 0)


class DSARowModeBufferOwner:
    """Own row-mode decode CPU staging and NPU buffers for the worker.

    The previous implementation split one physical allocation into three
    bookkeeping dataclasses: byte layout, replay staging, and persistent NPU
    buffers.  That made ownership look more fragmented than it really was.
    This owner follows the native ``CpuGpuBuffer`` model instead: allocate the
    CPU/NPU pair once, expose typed views, and let eager/graph adapters select
    an active or captured prefix without taking ownership of storage.

    Eager decode takes an active-row prefix; graph replay takes a captured-row
    prefix and marks its trailing rows as PAD. Both are non-owning views of the
    same worker-lifetime buffers.
    """

    def __init__(
        self,
        *,
        capacity: int,
        graph_capacity: int,
        device: torch.device,
        hbm_table_width: int,
        max_logical_blocks: int,
        total_layers: int,
    ) -> None:
        self.capacity = int(capacity)
        self.graph_capacity = int(graph_capacity)
        self.device = torch.device(device)
        self.total_layers = int(total_layers)
        hbm_table_width = int(hbm_table_width)

        row_metadata_end = (
            _ROW_META_FIELD_COUNT * self.capacity * _INT32_ITEMSIZE)
        hbm_table_start = row_metadata_end
        hbm_table_end = (
            hbm_table_start
            + self.capacity * hbm_table_width * _INT32_ITEMSIZE)
        total_bytes = hbm_table_end

        pin_memory = self.device.type != "cpu"
        self.row_metadata_buffer = CpuGpuBuffer(
            total_bytes,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.row_metadata_storage = self.row_metadata_buffer.gpu
        self.row_metadata_slab = self.row_metadata_storage[
            :row_metadata_end].view(torch.int32).reshape(
                _ROW_META_FIELD_COUNT, self.capacity)
        self.hbm_block_table = self.row_metadata_storage[
            hbm_table_start:hbm_table_end].view(torch.int32).reshape(
                self.capacity, hbm_table_width)
        self.row_metadata_storage_cpu = self.row_metadata_buffer.cpu
        row_metadata_cpu = self.row_metadata_storage_cpu[
            :row_metadata_end].view(torch.int32).reshape(
                _ROW_META_FIELD_COUNT, self.capacity)
        hbm_block_table_cpu = self.row_metadata_storage_cpu[
            hbm_table_start:hbm_table_end].view(torch.int32).reshape(
                self.capacity, hbm_table_width)
        self.row_metadata_np = row_metadata_cpu.numpy()
        self.hbm_block_table_np = hbm_block_table_cpu.numpy()

        self.dram_block_table_buffer = CpuGpuBuffer(
            self.capacity,
            int(max_logical_blocks),
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dram_block_table_cpu = self.dram_block_table_buffer.cpu
        self.dram_block_table_np = self.dram_block_table_cpu.numpy()
        self.dram_block_table = self.dram_block_table_buffer.gpu
        # LIDU outputs are shared by eager and graph prefix views. They are
        # allocated when this worker-lifetime owner is created, before any
        # layer hook runs, so the first eager decode does not allocate inside
        # model forward and graph capture observes stable addresses.
        self.layer_lidu_topk_index: torch.Tensor | None = None
        self.layer_lidu_topk_slots: torch.Tensor | None = None
        self.layer_lidu_miss_count: torch.Tensor | None = None
        self.layer_lidu_tail_info: torch.Tensor | None = None
        self._layer_lidu_outputs: tuple[
            DSALightningIndexerUpdateBuffers, ...
        ] | None = None
        self.dram_signature: tuple | None = None
        self.eager_batches: dict[
            tuple[int, int, int, int], DSAForwardRowModeDecodeBatch] = {}

    def ensure_lidu_outputs(
        self,
    ) -> tuple[DSALightningIndexerUpdateBuffers, ...]:
        """创建 eager/graph 共用的逐层固定地址 LIDU 输出。"""
        if self._layer_lidu_outputs is None:
            common_shape = (
                self.total_layers,
                self.capacity,
                1,
                DSA_LIDU_OUTPUT_CAPACITY,
            )
            self.layer_lidu_topk_index = torch.empty(
                common_shape, dtype=torch.int32, device=self.device)
            self.layer_lidu_topk_slots = torch.empty(
                common_shape, dtype=torch.int32, device=self.device)
            self.layer_lidu_miss_count = torch.empty(
                (self.total_layers, self.capacity),
                dtype=torch.int32,
                device=self.device,
            )
            self.layer_lidu_tail_info = torch.empty(
                (self.total_layers, self.capacity, 2),
                dtype=torch.int32,
                device=self.device,
            )
            assert self.layer_lidu_topk_index is not None
            assert self.layer_lidu_topk_slots is not None
            assert self.layer_lidu_miss_count is not None
            assert self.layer_lidu_tail_info is not None
            self._layer_lidu_outputs = tuple(
                DSALightningIndexerUpdateBuffers(
                    topk_index=self.layer_lidu_topk_index[layer_id],
                    topk_slots=self.layer_lidu_topk_slots[layer_id],
                    miss_count=self.layer_lidu_miss_count[layer_id],
                    tail_info=self.layer_lidu_tail_info[layer_id],
                )
                for layer_id in range(self.total_layers)
            )
        return self._layer_lidu_outputs

class DSARowModeRuntimeMixin:
    def configure_row_mode_decode_graph_capacity(self, capacity: int) -> None:
        """Bind storage capacity to the native FULL graph key family."""
        capacity = int(capacity)
        if capacity <= 0:
            raise RuntimeError(
                f"DSA graph capacity must be positive, got {capacity}")
        existing = self._row_mode_decode_buffer_owner
        if existing is not None and existing.graph_capacity != capacity:
            raise RuntimeError(
                "DSA graph capacity changed after persistent buffers were "
                f"allocated: existing={existing.graph_capacity}, "
                f"new={capacity}")
        self._row_mode_decode_graph_capacity_hint = capacity

    def get_row_mode_decode_graph_dummy_seq_len(self) -> int:
        """Return a capture-time indexer key length covering replayed requests."""
        max_model_len = int(self._vllm_config.model_config.max_model_len or 0)
        if max_model_len > 0:
            return max_model_len
        budget_tokens = int(self._hbm_sparse_budget_tokens or 0)
        return max(1, budget_tokens + int(self._vllm_blk_size))

    def get_row_mode_decode_graph_dummy_resident_seq_len(self) -> int:
        """Return the capture-time resident MLA cache length for SFA."""
        budget_tokens = int(self._hbm_sparse_budget_tokens or 0)
        return max(1, budget_tokens + int(self._vllm_blk_size))

    def _graph_max_logical_blocks(self) -> int:
        max_model_len = int(self._vllm_config.model_config.max_model_len or 0)
        max_model_len = max(max_model_len,
                            self.get_row_mode_decode_graph_dummy_seq_len())
        return max(1, (max_model_len + self._vllm_blk_size - 1)
                   // self._vllm_blk_size)

    def _row_mode_decode_graph_capacity(self) -> int:
        """Return the worker-level capacity backing all capture-size views."""
        resident_capacity = int(self.resident_token_pool.max_reqs)
        configured_hint = int(
            self._row_mode_decode_graph_capacity_hint or 0)
        if configured_hint > 0:
            capacity = configured_hint
        else:
            capacity = 0
        scheduler_capacity = int(
            self._vllm_config.scheduler_config.max_num_seqs or 0)
        capture_sizes = (
            self._vllm_config.compilation_config.cudagraph_capture_sizes or ())
        eligible_capture_sizes = [
            int(size) for size in capture_sizes
            if int(size) > 0 and (
                scheduler_capacity <= 0 or int(size) <= scheduler_capacity)
        ]
        if capacity <= 0:
            if eligible_capture_sizes:
                # Allocate only what the native FULL dispatcher can capture.
                # This avoids turning max_num_seqs=256 with capture sizes up
                # to 16 into an unnecessarily large per-layer output allocation.
                capacity = max(eligible_capture_sizes)
            else:
                capacity = scheduler_capacity
        if capacity <= 0:
            # Small unit-test owners do not carry a SchedulerConfig.  The
            # resident pool is still a valid upper bound in that environment.
            capacity = resident_capacity
        if capacity > resident_capacity:
            raise RuntimeError(
                "DSA graph capacity exceeds the resident token pool: "
                f"graph_capacity={capacity}, resident_capacity="
                f"{resident_capacity}")
        return capacity

    def _row_mode_decode_buffer_capacity(self) -> int:
        """Capacity of the shared eager/graph row metadata owner."""
        resident_capacity = int(self.resident_token_pool.max_reqs)
        scheduler_capacity = int(
            self._vllm_config.scheduler_config.max_num_seqs or 0)
        if scheduler_capacity <= 0:
            return resident_capacity
        if scheduler_capacity > resident_capacity:
            raise RuntimeError(
                "DSA row-mode buffer capacity exceeds the resident pool: "
                f"buffer_capacity={scheduler_capacity}, "
                f"resident_capacity={resident_capacity}")
        return scheduler_capacity

    def _fill_graph_dummy_hbm_block_table(
        self,
        graph_batch: DSAForwardRowModeDecodeBatch,
        full_block_table_tensor: torch.Tensor | None,
    ) -> None:
        """Fill capture-time HBM block table with legal non-zero ids."""
        row_count = int(graph_batch.batch_hbm_block_table.shape[0])
        block_count = int(graph_batch.batch_hbm_block_table.shape[1])
        device = graph_batch.batch_hbm_block_table.device
        fallback = torch.arange(
            1,
            block_count + 1,
            dtype=graph_batch.batch_hbm_block_table.dtype,
            device=device,
        ).reshape(1, -1).expand(row_count, block_count)

        graph_batch.batch_hbm_block_table.copy_(fallback)
        if not torch.is_tensor(full_block_table_tensor):
            return
        table = full_block_table_tensor.to(
            device=device,
            dtype=graph_batch.batch_hbm_block_table.dtype,
        )
        if table.ndim < 2:
            return
        rows = min(row_count, int(table.shape[0]))
        cols = min(block_count, int(table.shape[1]))
        if rows <= 0 or cols <= 0:
            return
        copied = table[:rows, :cols]
        graph_batch.batch_hbm_block_table[:rows, :cols].copy_(
            torch.where(copied > 0, copied, fallback[:rows, :cols]))

    def _graph_dummy_dram_block_capacity(self) -> int:
        blk_pool_mgr = self.dsa_hot_kv_store
        if blk_pool_mgr is None:
            return 0
        for layer_id in range(int(self.total_num_hidden_layers)):
            try:
                arena = blk_pool_mgr.get_arena(layer_id, BlockType.NOPE_K)
            except Exception:
                continue
            if torch.is_tensor(arena):
                return max(0, int(arena.shape[0]) - 1)
        return 0

    def _fill_graph_dummy_dram_block_tables(
        self,
        graph_batch: DSAForwardRowModeDecodeBatch,
    ) -> None:
        """Fill capture-time DRAM tables with legal non-zero arena block ids."""
        capacity = self._graph_dummy_dram_block_capacity()
        if capacity <= 0:
            raise RuntimeError(
                "DSA row-mode decode graph capture requires a "
                "preallocated DRAM "
                "arena before dummy row-mode decode can run")
        device = graph_batch.batch_dram_block_table.device
        width = int(graph_batch.max_logical_blocks)
        ids = ((torch.arange(width,
                             dtype=graph_batch.batch_dram_block_table.dtype,
                             device=device)
                % int(capacity)) + 1)
        graph_batch.batch_dram_block_table.copy_(
            ids.reshape(1, -1).expand_as(
                graph_batch.batch_dram_block_table))

    def _get_or_create_row_mode_decode_buffer_owner(
        self,
        *,
        tensor_device: torch.device | str,
    ) -> DSARowModeBufferOwner:
        device = torch.device(tensor_device)
        cached = self._row_mode_decode_buffer_owner
        if cached is not None:
            if cached.device != device:
                raise RuntimeError(
                    "DSA graph buffers were initialized on another device: "
                    f"existing={cached.device}, requested={device}")
            return cached

        graph_capacity = self._row_mode_decode_graph_capacity()
        capacity = self._row_mode_decode_buffer_capacity()
        if graph_capacity > capacity:
            raise RuntimeError(
                "DSA graph capacity exceeds row-mode buffer capacity: "
                f"graph_capacity={graph_capacity}, "
                f"buffer_capacity={capacity}")
        budget_tokens = int(self._hbm_sparse_budget_tokens or 0)
        if budget_tokens <= 0:
            raise RuntimeError(
                "DSA decode graph requires a positive sparse budget")
        block_size = int(self._vllm_blk_size)
        resident_graph_limit = (
            self.get_row_mode_decode_graph_dummy_resident_seq_len())
        budget_blocks = max(
            1, (resident_graph_limit + block_size - 1) // block_size)
        max_logical_blocks = self._graph_max_logical_blocks()
        total_layers = int(self.total_num_hidden_layers)
        if total_layers <= 0:
            raise RuntimeError(
                "DSA decode graph attention buffers require model layers")

        buffers = DSARowModeBufferOwner(
            capacity=capacity,
            graph_capacity=graph_capacity,
            device=device,
            hbm_table_width=budget_blocks,
            max_logical_blocks=max_logical_blocks,
            total_layers=total_layers,
        )
        buffers.ensure_lidu_outputs()
        self._row_mode_decode_buffer_owner = buffers
        return buffers

    def _get_or_create_row_mode_decode_graph_batch(
        self,
        row_count: int,
        *,
        tensor_device: torch.device | str,
    ) -> DSAForwardRowModeDecodeBatch:
        row_count = int(row_count)
        if row_count <= 0:
            raise RuntimeError(
                "DSA decode graph requires a positive row count")
        buffers = self._get_or_create_row_mode_decode_buffer_owner(
            tensor_device=tensor_device)
        if row_count > buffers.graph_capacity:
            raise RuntimeError(
                "DSA decode graph row count exceeds persistent buffer "
                f"capacity: row_count={row_count}, capacity="
                f"{buffers.graph_capacity}")
        cached = self._graph_row_mode_decode_batches.get(row_count)
        if cached is not None:
            return cached

        graph_row_metadata_slab = buffers.row_metadata_slab[:, :row_count]
        graph_batch_hbm_block_table = buffers.hbm_block_table[:row_count]
        graph_batch = DSAForwardRowModeDecodeBatch(
            resident_pool_indices_tensor=(
                graph_row_metadata_slab[_ROW_META_RESIDENT_POOL]),
            batch_hbm_block_table=graph_batch_hbm_block_table,
            batch_dram_block_table=(
                buffers.dram_block_table[:row_count]),
            layer_lidu_outputs=tuple(
                DSALightningIndexerUpdateBuffers(
                    topk_index=outputs.topk_index[:row_count],
                    topk_slots=outputs.topk_slots[:row_count],
                    miss_count=outputs.miss_count[:row_count],
                    tail_info=outputs.tail_info[:row_count],
                )
                for outputs in buffers.ensure_lidu_outputs()),
            row_modes_tensor=graph_row_metadata_slab[_ROW_META_MODE],
            full_block_dump_batch=DSAFullBlockDumpBatch(
                src_hbm_block_ids_tensor=(
                    graph_row_metadata_slab[
                        _ROW_META_DUMP_SRC_HBM_BLOCK]),
                dst_dram_block_ids_tensor=(
                    graph_row_metadata_slab[
                        _ROW_META_DUMP_DST_DRAM_BLOCK]),
            ),
        )
        self._graph_row_mode_decode_batches[row_count] = graph_batch
        return graph_batch

    def _get_or_create_row_mode_graph_layer_hook_plan(
        self,
        graph_batch: DSAForwardRowModeDecodeBatch,
    ) -> DSAForwardLayerHookPlan:
        """Bind graph-stable dump views to the common layer-hook contract."""
        row_count = int(graph_batch.row_count)
        cached = self._graph_row_mode_layer_hook_plans.get(row_count)
        if cached is not None:
            return cached
        hook_plan = DSAForwardLayerHookPlan(
            full_block_dump_batch=graph_batch.full_block_dump_batch,
        )
        self._graph_row_mode_layer_hook_plans[row_count] = hook_plan
        return hook_plan

    def _reset_row_mode_decode_graph_batch_for_capture(
        self,
        graph_batch: DSAForwardRowModeDecodeBatch,
        full_block_table_tensor: torch.Tensor | None = None,
    ) -> None:
        """Fill graph buffers with a valid row-mode dummy decode batch."""
        row_count = int(graph_batch.resident_pool_indices_tensor.numel())
        device = graph_batch.resident_pool_indices_tensor.device
        row_ids = torch.arange(row_count, dtype=torch.int32, device=device)
        graph_batch.resident_pool_indices_tensor.copy_(row_ids)
        graph_batch.row_modes_tensor.fill_(int(DSADecodeRowMode.SPARSE))
        graph_batch.full_block_dump_batch.src_hbm_block_ids_tensor.zero_()
        graph_batch.full_block_dump_batch.dst_dram_block_ids_tensor.fill_(
            KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID)
        self._fill_graph_dummy_hbm_block_table(graph_batch,
                                               full_block_table_tensor)
        self._fill_graph_dummy_dram_block_tables(graph_batch)

    def prepare_row_mode_decode_graph_capture_batch(
        self,
        row_count: int,
        *,
        tensor_device: torch.device | str,
        full_block_table_tensor: torch.Tensor | None = None,
    ):
        """Install persistent DSA buffers while FULL graph is being captured."""
        graph_batch = self._get_or_create_row_mode_decode_graph_batch(
            row_count, tensor_device=tensor_device)
        # Allocate pinned replay staging while graphs are being captured, not
        # on the first real request. A recapture rewrites HBM/DRAM graph inputs
        # with dummy values, so only invalidate refresh signatures while
        # retaining the already allocated host buffers.
        buffer_owner = self._get_or_create_row_mode_decode_buffer_owner(
            tensor_device=graph_batch.resident_pool_indices_tensor.device)
        buffer_owner.dram_signature = None
        # DSA 图捕获会用 dummy request/pool row 真正跑一遍 LIDU。cache_slots
        # 是请求生命周期状态，不是普通 graph input；capture 前种入合法
        # first-fill metadata，capture 后清理，避免 dummy 映射污染真实请求。
        self.resident_token_pool.seed_cache_slots_prefix(
            int(row_count), int(self._hbm_sparse_budget_tokens))
        saved_state = (
            self.forward_row_mode_decode_batch,
            self.forward_layer_hook_plan,
            int(row_count),
        )
        self._reset_row_mode_decode_graph_batch_for_capture(
            graph_batch,
            full_block_table_tensor=full_block_table_tensor,
        )

        self.forward_row_mode_decode_batch = graph_batch
        self.forward_layer_hook_plan = (
            self._get_or_create_row_mode_graph_layer_hook_plan(graph_batch))
        return saved_state

    def restore_row_mode_decode_graph_capture_batch(self, saved_state) -> None:
        if saved_state is None:
            return
        (
            self.forward_row_mode_decode_batch,
            self.forward_layer_hook_plan,
            row_count,
        ) = saved_state
        self.resident_token_pool.clear_cache_slots_prefix(int(row_count))

    def _serialize_row_mode_scalar_metadata(
        self,
        input_state: DSAInputBatchState,
        buffer_owner: DSARowModeBufferOwner,
        *,
        active_row_count: int,
        output_row_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Mirror canonical InputBatch columns into the shared CPU slab.

        Eager passes ``active_row_count == output_row_count``. Graph replay
        passes its captured row count as ``output_row_count`` and receives
        explicit PAD rows after the active prefix. Both paths therefore use
        exactly the same field mapping and differ only in physical shape.

        Most fields remain request-row aligned. The two dump columns are an
        operator-specific exception: boundary rows whose payload is already
        present in DRAM carry destination ``-1`` and are filtered here. The
        remaining physical copies form a compact job prefix, so optimized
        eager launches exactly the real jobs while graph replay exposes the
        same storage as a fixed captured prefix padded with destination ``-1``.

        Returns zero-copy views of the semantic columns needed by later
        HBM/DRAM table staging, followed by the number of physical dump jobs.
        Candidate range and budget stay in ``DSAInputBatchState`` because no
        device operator consumes a second copy of them.
        """
        active_row_count = int(active_row_count)
        output_row_count = int(output_row_count)
        if (active_row_count <= 0 or output_row_count < active_row_count
                or output_row_count > int(buffer_owner.capacity)):
            raise RuntimeError(
                "Invalid DSA row-mode serialization shape: "
                f"active={active_row_count}, output={output_row_count}, "
                f"capacity={buffer_owner.capacity}")

        active_slice = slice(0, active_row_count)
        row_metadata = buffer_owner.row_metadata_np
        pool_indices = input_state.resident_pool_indices[active_slice]
        candidate_range_ends = input_state.candidate_range_ends[active_slice]
        budget_lens = input_state.budget_slot_counts[active_slice]
        sparse_mask = input_state.sparse_mask[active_slice]

        if active_row_count < output_row_count:
            pad_slice = slice(active_row_count, output_row_count)
            padding_pool_index = int(
                self.resident_token_pool.padding_pool_index)
            row_metadata[:, pad_slice].fill(0)
            row_metadata[_ROW_META_RESIDENT_POOL,
                         pad_slice] = padding_pool_index
            row_metadata[_ROW_META_MODE,
                         pad_slice] = int(DSADecodeRowMode.PAD)

        np.copyto(
            row_metadata[_ROW_META_RESIDENT_POOL, active_slice],
            pool_indices,
            casting="unsafe",
        )
        np.copyto(
            row_metadata[_ROW_META_MODE, active_slice],
            input_state.row_modes[active_slice],
            casting="unsafe",
        )
        dump_src_column = row_metadata[
            _ROW_META_DUMP_SRC_HBM_BLOCK, :output_row_count]
        dump_dst_column = row_metadata[
            _ROW_META_DUMP_DST_DRAM_BLOCK, :output_row_count]
        dump_src_column.fill(0)
        dump_dst_column.fill(KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID)
        dump_boundary_count = int(input_state.decode_dump_row_count)
        if dump_boundary_count > output_row_count:
            raise RuntimeError(
                "DSA decode dump boundary count exceeds output rows: "
                f"boundaries={dump_boundary_count}, "
                f"output_rows={output_row_count}")
        dump_job_count = 0
        if dump_boundary_count > 0:
            dump_rows = input_state.decode_dump_row_indices[
                :dump_boundary_count]
            dump_destinations = (
                input_state.decode_dump_dst_dram_block_ids[dump_rows])
            dump_job_rows = dump_rows[dump_destinations >= 0]
            dump_job_count = int(dump_job_rows.size)
            np.take(
                input_state.decode_dump_src_hbm_block_ids,
                dump_job_rows,
                out=dump_src_column[:dump_job_count],
            )
            np.take(
                input_state.decode_dump_dst_dram_block_ids,
                dump_job_rows,
                out=dump_dst_column[:dump_job_count],
            )
        return (
            pool_indices,
            candidate_range_ends,
            budget_lens,
            sparse_mask,
            dump_job_count,
        )

    def _stage_row_mode_hbm_block_table(
        self,
        input_batch: NPUInputBatch,
        buffer_owner: DSARowModeBufferOwner,
        *,
        full_attention_group_id: int,
        active_row_count: int,
        output_row_count: int,
        required_hbm_blocks: int,
    ) -> bool:
        """Stage native full-cache block rows into the packed CPU buffer."""
        active_row_count = int(active_row_count)
        output_row_count = int(output_row_count)
        required_hbm_blocks = int(required_hbm_blocks)
        if (required_hbm_blocks <= 0
                or required_hbm_blocks
                > int(buffer_owner.hbm_block_table.shape[1])):
            return False

        staging = buffer_owner.hbm_block_table_np
        staging[:output_row_count].fill(0)
        full_block_table_np = input_batch.block_table[
            int(full_attention_group_id)].get_numpy_array()
        if (not isinstance(full_block_table_np, np.ndarray)
                or full_block_table_np.ndim != 2
                or int(full_block_table_np.shape[0]) < active_row_count):
            return False

        copy_cols = min(
            required_hbm_blocks,
            int(staging.shape[1]),
            int(full_block_table_np.shape[1]),
        )
        if copy_cols > 0:
            np.copyto(
                staging[:active_row_count, :copy_cols],
                full_block_table_np[:active_row_count, :copy_cols],
                casting="unsafe",
            )
        return True

    def _refresh_row_mode_dram_block_table(
        self,
        buffer_owner: DSARowModeBufferOwner,
        *,
        active_row_count: int,
        output_row_count: int,
        logical_block_count: int,
        pool_indices: np.ndarray,
        sparse_mask: np.ndarray,
    ) -> bool:
        """Refresh the versioned DRAM table shared by eager and graph views."""
        active_row_count = int(active_row_count)
        output_row_count = int(output_row_count)
        logical_block_count = int(logical_block_count)
        if (logical_block_count <= 0
                or logical_block_count
                > int(buffer_owner.dram_block_table.shape[1])):
            return False

        any_sparse = bool(np.any(sparse_mask))
        dram_store = self.dsa_hot_kv_store
        if any_sparse and dram_store is None:
            return False

        table_version = (
            None if dram_store is None else
            dram_store.dram_block_table_version)
        if any_sparse:
            pool_signature = tuple(int(value) for value in pool_indices)
            signature = (
                id(dram_store),
                table_version,
                output_row_count,
                logical_block_count,
                pool_signature,
            )
            cacheable = table_version is not None
        else:
            # Dense rows have miss_count=0 and KSC never reads DRAM for them.
            # Keep their physical
            # input deterministic and avoid materializing irrelevant tables.
            signature = ("zero", output_row_count, logical_block_count)
            cacheable = True

        if cacheable and buffer_owner.dram_signature == signature:
            return True

        staging = buffer_owner.dram_block_table_np
        staging[:output_row_count, :logical_block_count].fill(0)
        if any_sparse:
            dram_table = dram_store.get_dram_block_table_tensor(
                num_logical_blocks=logical_block_count,
                device=torch.device("cpu"),
                dtype=torch.int32,
            )
            if (not torch.is_tensor(dram_table)
                    or dram_table.device.type != "cpu"
                    or dram_table.ndim != 2):
                return False
            dram_table_np = dram_table.numpy()
            if int(pool_indices.max()) >= int(dram_table_np.shape[0]):
                return False
            np.copyto(
                staging[:active_row_count, :logical_block_count],
                dram_table_np[
                    pool_indices.astype(np.intp, copy=False),
                    :logical_block_count,
                ],
            )

        buffer_owner.dram_block_table[
            :output_row_count, :logical_block_count].copy_(
                buffer_owner.dram_block_table_cpu[
                    :output_row_count, :logical_block_count],
                non_blocking=True,
            )
        if any_sparse and cacheable:
            # Table growth during materialization may bump the version.
            signature = (
                id(dram_store),
                dram_store.dram_block_table_version,
                output_row_count,
                logical_block_count,
                pool_signature,
            )
        buffer_owner.dram_signature = signature if cacheable else None
        return True

    def prepare_row_mode_decode_eager_batch(
        self,
        input_state: DSAInputBatchState,
    ) -> DSAForwardRowModeDecodeBatch | None:
        """Build the supported eager decode view from worker-lifetime buffers.

        A return value of ``None`` means the current forward is outside the
        supported decode-only request-row contract. Query lengths may differ
        per request for MTP; the layer data plane executes them round by round.
        """
        row_count = int(input_state.row_count)
        active_slice = slice(0, row_count)
        indexer_key_lens = input_state.indexer_key_lens[active_slice]
        decode_mask = (
            (input_state.num_output_tokens[active_slice] > 0)
            & (input_state.query_lens[active_slice] > 0)
            & (indexer_key_lens > 0)
            & (input_state.budget_slot_counts[active_slice] > 0)
            & (input_state.resident_pool_indices[active_slice] >= 0)
        )
        decode_rows = np.flatnonzero(decode_mask).astype(
            np.int64, copy=False)
        decode_row_count = int(decode_rows.size)
        if decode_row_count == 0:
            return DSAForwardRowModeDecodeBatch.empty(
                tensor_device=self.resident_token_pool.device)

        identity_rows_np = np.arange(decode_row_count, dtype=np.int64)
        if (decode_row_count != row_count
                or not np.array_equal(decode_rows, identity_rows_np)):
            return None
        device = self.resident_token_pool.device

        buffer_owner = self._get_or_create_row_mode_decode_buffer_owner(
            tensor_device=device)
        if decode_row_count > buffer_owner.capacity:
            raise RuntimeError(
                "DSA eager decode rows exceed row-mode buffer capacity: "
                f"rows={decode_row_count}, capacity={buffer_owner.capacity}")

        (
            pool_indices,
            candidate_range_ends,
            budget_lens,
            sparse_mask,
            dump_job_count,
        ) = self._serialize_row_mode_scalar_metadata(
            input_state,
            buffer_owner,
            active_row_count=decode_row_count,
            output_row_count=decode_row_count,
        )
        if not _row_mode_decode_lengths_are_valid(
                indexer_key_lens=indexer_key_lens,
                candidate_range_ends=candidate_range_ends,
                budget_lens=budget_lens,
                sparse_mask=sparse_mask):
            raise RuntimeError(
                "DSA eager row-mode decode has invalid indexer/candidate/"
                "budget lengths")
        sparse_budget = input_state.sparse_budget_tokens[active_slice]
        if (np.any(sparse_mask)
                and int(sparse_budget[sparse_mask].max())
                > int(self._hbm_sparse_budget_tokens)):
            raise RuntimeError(
                "DSA sparse-row budget exceeds row-mode topK")

        block_size = int(self._vllm_blk_size)
        max_budget_slots = int(budget_lens.max())
        required_hbm_blocks = max(
            (max_budget_slots + block_size - 1) // block_size,
            (int(self._hbm_sparse_budget_tokens) + block_size - 1)
            // block_size,
        )
        input_batch = input_state.input_batch
        full_group_id = int(input_state.full_attention_group_id)
        if not self._stage_row_mode_hbm_block_table(
                input_batch,
                buffer_owner,
                full_attention_group_id=full_group_id,
                active_row_count=decode_row_count,
                output_row_count=decode_row_count,
                required_hbm_blocks=required_hbm_blocks):
            raise RuntimeError(
                "DSA eager decode requires the native full-cache CPU block "
                "table in InputBatch row order with sufficient capacity")

        max_candidate_end = int(
            input_state.candidate_range_ends[active_slice].max())
        # The custom-op ABI still needs a physical 2-D DRAM table when every
        # row is short DENSE and therefore has no full-block candidate prefix.
        # Keep one zero-filled placeholder column; semantic candidate lengths
        # remain zero and the DENSE LIDU/KSC branch never reads this table.
        max_logical_blocks = max(
            1,
            (max_candidate_end + block_size - 1) // block_size,
        )
        if max_logical_blocks > int(buffer_owner.dram_block_table.shape[1]):
            raise RuntimeError(
                "DSA eager DRAM block table exceeds shared buffer width: "
                f"required={max_logical_blocks}, available="
                f"{int(buffer_owner.dram_block_table.shape[1])}")
        buffer_owner.row_metadata_buffer.copy_to_gpu()

        if not self._refresh_row_mode_dram_block_table(
                buffer_owner,
                active_row_count=decode_row_count,
                output_row_count=decode_row_count,
                logical_block_count=max_logical_blocks,
                pool_indices=pool_indices,
                sparse_mask=sparse_mask):
            raise RuntimeError(
                "DSA eager decode could not refresh the hot-DRAM block table")

        batch_key = (
            decode_row_count,
            required_hbm_blocks,
            max_logical_blocks,
            dump_job_count,
        )
        eager_batch = buffer_owner.eager_batches.get(batch_key)
        if eager_batch is None:
            row_metadata_device = buffer_owner.row_metadata_slab[
                :, :decode_row_count]
            eager_batch = DSAForwardRowModeDecodeBatch(
                resident_pool_indices_tensor=(
                    row_metadata_device[_ROW_META_RESIDENT_POOL]),
                # KSC interprets both tables as contiguous row-major storage.
                # Slicing their column dimension would preserve the backing
                # row stride and create a non-contiguous 2-D view.  Eager and
                # graph therefore expose the same full-width owner prefix;
                # miss_count/src token ids still bound all semantic reads.
                batch_hbm_block_table=(
                    buffer_owner.hbm_block_table[:decode_row_count]),
                batch_dram_block_table=(
                    buffer_owner.dram_block_table[:decode_row_count]),
                layer_lidu_outputs=tuple(
                    DSALightningIndexerUpdateBuffers(
                        topk_index=outputs.topk_index[:decode_row_count],
                        topk_slots=outputs.topk_slots[:decode_row_count],
                        miss_count=outputs.miss_count[:decode_row_count],
                        tail_info=outputs.tail_info[:decode_row_count],
                    )
                    for outputs in buffer_owner.ensure_lidu_outputs()),
                row_modes_tensor=row_metadata_device[_ROW_META_MODE],
                full_block_dump_batch=DSAFullBlockDumpBatch(
                    src_hbm_block_ids_tensor=(
                        row_metadata_device[
                            _ROW_META_DUMP_SRC_HBM_BLOCK,
                            :dump_job_count]),
                    dst_dram_block_ids_tensor=(
                        row_metadata_device[
                            _ROW_META_DUMP_DST_DRAM_BLOCK,
                            :dump_job_count]),
                ),
            )
            buffer_owner.eager_batches[batch_key] = eager_batch
        return eager_batch

    def prepare_row_mode_decode_graph_replay_batch(
        self,
        input_state: DSAInputBatchState,
        input_batch: NPUInputBatch,
        full_attention_group_id: int,
        row_count: int,
    ) -> bool:
        """Serialize the shared InputBatch row state into captured inputs.

        Request stage/budget/tail semantics have already been projected once
        after baseline ``_update_states`` fixed the InputBatch row order, and
        ``_prepare_inputs`` has bound the resulting query layout. This method
        is deliberately a physical adapter only: it bulk-writes those rows to
        the persistent graph slab and never re-reads scheduler dictionaries or
        rebuilds request-level forward objects.
        """
        captured_row_count = int(row_count)
        active_row_count = int(input_state.row_count)
        if (captured_row_count <= 0 or active_row_count <= 0
                or active_row_count > captured_row_count
                or not input_state.matches_input_batch(
                    input_batch, active_row_count)
                or not input_state.query_layout_bound):
            return False
        active_slice = slice(0, active_row_count)
        indexer_key_lens = input_state.indexer_key_lens[active_slice]
        if np.any(input_state.num_scheduled_tokens[active_slice] != 1):
            return False
        if np.any(input_state.num_output_tokens[active_slice] <= 0):
            return False
        if np.any(input_state.query_lens[active_slice] != 1):
            return False
        pool_indices_np = input_state.resident_pool_indices[active_slice]
        if np.any(pool_indices_np < 0):
            return False

        graph_batch = self._get_or_create_row_mode_decode_graph_batch(
            captured_row_count,
            tensor_device=self.resident_token_pool.device,
        )
        buffer_owner = self._get_or_create_row_mode_decode_buffer_owner(
            tensor_device=graph_batch.resident_pool_indices_tensor.device)
        (
            pool_indices_np,
            candidate_range_ends,
            budget_lens,
            sparse_mask,
            _dump_job_count,
        ) = self._serialize_row_mode_scalar_metadata(
            input_state,
            buffer_owner,
            active_row_count=active_row_count,
            output_row_count=captured_row_count,
        )
        if not _row_mode_decode_lengths_are_valid(
                indexer_key_lens=indexer_key_lens,
                candidate_range_ends=candidate_range_ends,
                budget_lens=budget_lens,
                sparse_mask=sparse_mask):
            return False
        if np.any(sparse_mask) and self.dsa_hot_kv_store is None:
            return False

        max_budget_slots = int(budget_lens.max())
        max_candidate_end = int(
            input_state.candidate_range_ends[active_slice].max())
        block_size = int(self._vllm_blk_size)
        actual_logical_blocks = (
            max_candidate_end + block_size - 1) // block_size
        if actual_logical_blocks > int(graph_batch.max_logical_blocks):
            return False
        # Copy only the request's resident budget blocks. Graph storage is
        # fixed wider than this active prefix, so it already satisfies LIDU's
        # resident-capacity table-width check for DENSE and SPARSE rows.
        required_hbm_blocks = (
            max_budget_slots + block_size - 1) // block_size
        if not self._stage_row_mode_hbm_block_table(
                input_batch,
                buffer_owner,
                full_attention_group_id=full_attention_group_id,
                active_row_count=active_row_count,
                output_row_count=captured_row_count,
                required_hbm_blocks=required_hbm_blocks):
            return False

        # The packed layout is field-major so each operator input remains a
        # contiguous 1-D view. A captured-row prefix is therefore not one raw
        # byte prefix; copy the small max-capacity slab once, matching the
        # baseline persistent-buffer pattern without issuing per-field H2Ds.
        buffer_owner.row_metadata_buffer.copy_to_gpu()

        if not self._refresh_row_mode_dram_block_table(
                buffer_owner,
                active_row_count=active_row_count,
                output_row_count=captured_row_count,
                logical_block_count=int(graph_batch.max_logical_blocks),
                pool_indices=pool_indices_np,
                sparse_mask=sparse_mask):
            return False

        self.forward_row_mode_decode_batch = graph_batch
        # DENSE, ENTER and SPARSE all consume this fixed row-mode contract.
        # ENTER's scheduler-side block-table transition is already complete;
        # decode full-block dump is represented by the fixed src/dst columns
        # above and executed by the independent layer-wise dump op.
        self.forward_layer_hook_plan = (
            self._get_or_create_row_mode_graph_layer_hook_plan(graph_batch))
        return True
