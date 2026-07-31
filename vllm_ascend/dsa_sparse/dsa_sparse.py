"""DSA 稀疏卸载的 scheduler/worker 算法核心与主 hook 管理器。

本文件保留 DSA 稀疏卸载的运行时主流程：scheduler/worker 两侧请求状态、
slot 估算与 cache layout 转换、model-forward 元数据汇聚、
``attention_begin/execute_decode_selection_pipeline/attention_finished`` hook，以及
eager/graph
共享 row-mode runtime 的生命周期衔接。``DSASparseBase`` 保留算法公共契约，
``DSASparseV1`` 是当前 LIDU/KSC/SFA-Offload row-mode + 独立满块复制方案的
具体实现。

不再把所有辅助逻辑都堆在这里：
- dsa_input_batch_state.py 保存与原生 InputBatch 对齐的列式行语义。
- dsa_forward_batch.py 只定义一轮 forward 的数据契约。
- dsa_forward_batch_builder.py 构造动态 eager 生命周期计划和物理 dump jobs。
- dsa_graph_gate.py 只负责图模式准入策略。
- dsa_row_mode_runtime.py 负责 eager/graph 共用的固定容量物理镜像。

本模块不复制 vLLM graph dispatcher，也不拥有 Ascend kernel。后续优化应优先
减少类内热路径开销，避免再次引入与 ``DSAInputBatchState`` 同义的请求或
forward 元数据对象；``execute_begin/execute_finished`` 当前是刻意保留的
model-forward 扩展 hook，尚无计算逻辑。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_ascend.dsa_sparse.dsa_ascend_ops_backend import (
    DSAOffloadSelectionOutput,
)
from vllm_ascend.dsa_sparse.dsa_forward_batch import (
    DSAForwardLayerHookPlan,
    DSAForwardRowModeDecodeBatch,
    DSALightningIndexerUpdateBuffers,
)
from vllm_ascend.dsa_sparse.dsa_forward_batch_builder import (
    build_forward_layer_hook_plan_from_input_state,
    reserve_decode_full_block_dumps_from_input_state,
)
from vllm_ascend.dsa_sparse.dsa_hot_kv_store_core import BlockType, DSAHotKVStore
from vllm_ascend.dsa_sparse.dsa_input_batch_state import DSAInputBatchState
from vllm_ascend.dsa_sparse.dsa_layer_cache_zones import (
    DSALayerCacheRegistry,
    LayerCacheZones,
    resolve_layer_cache_zones,
)
from vllm_ascend.dsa_sparse.dsa_resident_pool import DSAResidentTokenPool
from vllm_ascend.dsa_sparse.dsa_row_mode_runtime import (
    DSARowModeBufferOwner,
    DSARowModeRuntimeMixin,
)
from vllm_ascend.dsa_sparse.dsa_spec_utils import (
    is_dsa_indexer_spec,
    is_dsa_mla_resident_spec,
)
from vllm_ascend.dsa_sparse.dsa_types import (
    INVALID_SLOT,
    DSASparseRole,
    ReqStage,
    ReqType,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

logger = init_logger(__name__)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(value / multiple) * multiple)


class DSASparseBase:
    """DSA sparse offload algorithm base shared by scheduler and worker.

    DSASparseBase owns the common algorithm configuration and invariants:
    block size, rounded sparse budget, sparse enable threshold, and the
    per-forward batch placeholders consumed by eager and graph paths.

    Keep this base class focused on the algorithm contract shared by future
    DSA sparse variants.  Execution-mode helpers, such as FULL graph stable
    buffers, should be added by concrete implementations instead of being a
    hard dependency of every DSA algorithm base.
    """

    def __init__(self, vllm_config: VllmConfig, role):
        self._vllm_config = vllm_config
        self._role = role
        self._vllm_blk_size = int(vllm_config.cache_config.block_size or 0)
        if self._vllm_blk_size <= 0:
            raise ValueError(
                "DSA sparse offload requires Ascend to resolve a positive "
                f"block_size before runtime construction, got "
                f"{self._vllm_blk_size}")
        configured_sparse_budget = int(
            vllm_config.cache_config.dsa_hbm_sparse_budget or 0)
        rounded_sparse_budget = _round_up_to_multiple(
            configured_sparse_budget, self._vllm_blk_size)
        if (configured_sparse_budget > 0
                and rounded_sparse_budget != configured_sparse_budget):
            logger.warning(
                "DSA hbm_sparse_budget=%s is not aligned to block_size=%s; "
                "rounding up to %s",
                configured_sparse_budget, self._vllm_blk_size,
                rounded_sparse_budget)
        self._hbm_sparse_budget_tokens = rounded_sparse_budget
        self._sparse_activation_tokens = int(
            vllm_config.cache_config.dsa_sparse_activation_tokens)
        self._prompt_budget_thresholds = tuple(
            int(value) for value in
            vllm_config.cache_config.dsa_prompt_budget_thresholds)
        self._resident_budget_tokens = tuple(
            int(value) for value in
            vllm_config.cache_config.dsa_resident_budget_tokens)
        if any(
                budget % self._vllm_blk_size != 0
                for budget in self._resident_budget_tokens):
            raise ValueError(
                "All DSA resident budgets must be aligned to block_size: "
                f"budgets={self._resident_budget_tokens}, "
                f"block_size={self._vllm_blk_size}")
        if self._sparse_activation_tokens % self._vllm_blk_size != 0:
            raise ValueError(
                "DSA sparse activation must be aligned to block_size: "
                f"activation={self._sparse_activation_tokens}, "
                f"block_size={self._vllm_blk_size}")
        # 保留历史属性名，含义已收敛为“总上下文超过该值后允许 ENTER”。
        # 它不再等于最大 HBM 预算加 tail；每请求预算由 prompt 档位单独冻结。
        self._enable_dsa_prompt_len = self._sparse_activation_tokens
        self.forward_row_mode_decode_batch = DSAForwardRowModeDecodeBatch.empty()
        self._empty_forward_layer_hook_plan = DSAForwardLayerHookPlan.empty()
        self.forward_layer_hook_plan = self._empty_forward_layer_hook_plan
        # FULL graph captures tensor addresses, not Python objects. DSA graph
        # paths therefore swap the normal per-forward row-mode batch with
        # persistent buffers keyed by captured row count and only refresh
        # their contents before replay.
        self._graph_row_mode_decode_batches: dict[
            int, DSAForwardRowModeDecodeBatch] = {}
        self._graph_row_mode_layer_hook_plans: dict[
            int, DSAForwardLayerHookPlan] = {}
        # Physical graph inputs/outputs are allocated once at worker capacity;
        # entries above are only shape-specific prefix views captured by the
        # native FULL-decode graph keys.
        self._row_mode_decode_buffer_owner: DSARowModeBufferOwner | None = None
        self._row_mode_decode_graph_capacity_hint = 0
        self._full_attention_group_id: int | None = None
        self._indexer_group_id: int | None = None

    def _is_sparse_cache_enabled(self) -> bool:
        return bool(self._vllm_config.cache_config.enable_dsa_sparse_cache)

    def _select_target_resident_budget_tokens(self,
                                               prompt_tokens: int) -> int:
        prompt_tokens = max(0, int(prompt_tokens))
        for threshold, budget in zip(self._prompt_budget_thresholds,
                                     self._resident_budget_tokens):
            if prompt_tokens <= threshold:
                return int(budget)
        return int(self._resident_budget_tokens[-1])

    def _get_sparse_budget_tokens(self, candidate_tokens: int,
                                  target_budget_tokens: int) -> int:
        target_budget_tokens = int(target_budget_tokens)
        if target_budget_tokens <= 0 or candidate_tokens <= 0:
            return 0
        if candidate_tokens < target_budget_tokens:
            return 0
        return target_budget_tokens


class DSASparseV1(DSARowModeRuntimeMixin, DSASparseBase):
    """Current LIDU/KSC/SFA-Offload based DSA sparse implementation.

    DSASparseV1 keeps the scheduler/worker hooks in the algorithm class and
    opts into DSARowModeRuntimeMixin because this concrete implementation supports
    shared eager/graph row-mode physical adapters. A future DSA variant may choose a
    different graph strategy without changing DSASparseBase.
    """

    def __init__(self,
                 vllm_config,
                 role,
                 dram_store: DSAHotKVStore | None = None,
                 ops_backend: Any | None = None,
                 resident_device: torch.device | str | None = None):
        super().__init__(vllm_config, role)

        if self._role == DSASparseRole.SCHEDULER:
            return

        self.dsa_hot_kv_store = (
            dram_store if dram_store is not None else DSAHotKVStore())
        if ops_backend is None:
            raise RuntimeError(
                "DSA sparse worker requires an Ascend offload-operator backend")
        self.ops_backend = ops_backend

        self.total_num_hidden_layers = (
            vllm_config.model_config.get_total_num_hidden_layers()
        )
        resident_budget_tokens = max(1, int(self._hbm_sparse_budget_tokens or 0))
        self.resident_token_pool = DSAResidentTokenPool(
            max_reqs=int(vllm_config.scheduler_config.max_num_seqs or 1),
            num_layers=self.total_num_hidden_layers,
            max_tokens=resident_budget_tokens,
            max_model_len=int(vllm_config.model_config.max_model_len),
            block_size=int(self._vllm_blk_size),
            device=resident_device,
        )

        self.layer_cache_registry = DSALayerCacheRegistry(
            num_layers=self.total_num_hidden_layers)
        # 这些张量是所有真实 decode 都会使用的 worker-lifetime 常驻状态，
        # 不能等 KV cache 容量规划完成后才在首次 eager/graph forward 中突发
        # 分配。提前创建可让 determine_available_memory() 把 cache_slots、
        # row metadata 和逐层 LIDU out buffer 的 HBM 占用计入真实基线。
        self._get_or_create_row_mode_decode_buffer_owner(
            tensor_device=self.resident_token_pool.device)

    def bind_input_batch_state(
        self,
        input_batch: NPUInputBatch,
    ) -> DSAInputBatchState:
        """Attach/reuse the DSA sidecar for the current NPU InputBatch."""
        state = input_batch.dsa_state
        max_num_reqs = int(input_batch.max_num_reqs)
        if (not isinstance(state, DSAInputBatchState)
                or state.input_batch is not input_batch
                or state.max_num_reqs != max_num_reqs
                or state.block_size != int(self._vllm_blk_size)):
            state = DSAInputBatchState(
                input_batch=input_batch,
                block_size=int(self._vllm_blk_size),
            )
            input_batch.dsa_state = state
        return state

    def prepare_input_batch_state(
        self,
        scheduler_output: SchedulerOutput,
        requests: dict[str, CachedRequestState],
        input_batch: NPUInputBatch,
        num_scheduled_tokens: np.ndarray,
        kv_cache_config,
    ) -> DSAInputBatchState:
        """Refresh the canonical DSA row projection once per model forward."""
        state = self.bind_input_batch_state(input_batch)
        state.refresh(
            scheduler_output=scheduler_output,
            requests=requests,
            num_scheduled_tokens_by_row=num_scheduled_tokens,
            full_attention_group_id=self._get_full_attention_group_id(
                kv_cache_config),
            indexer_group_id=self._get_indexer_group_id(kv_cache_config),
            resident_token_pool=self.resident_token_pool,
        )
        return state


    def _get_full_attention_group_id(self, kv_cache_config) -> int:
        # DSA residency/load applies to the MLA/full cache group, never the
        # selector-only indexer group.
        cached_group_id = self._full_attention_group_id
        if cached_group_id is not None:
            return int(cached_group_id)
        full_group_ids = [
            i for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if is_dsa_mla_resident_spec(group.kv_cache_spec)
        ]
        if not full_group_ids:
            raise RuntimeError(
                "DSA requires an MLA/full resident KVSpec group for "
                "full-cache residency")
        self._full_attention_group_id = int(full_group_ids[0])
        return self._full_attention_group_id

    def _get_indexer_group_id(self, kv_cache_config) -> int:
        # The indexer cache is a separate dense KV group so its block table and
        # budget do not collapse back into MLA/full-cache sparse semantics.
        cached_group_id = self._indexer_group_id
        if cached_group_id is not None:
            return int(cached_group_id)
        indexer_group_ids = [
            i for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if is_dsa_indexer_spec(group.kv_cache_spec)
        ]
        if not indexer_group_ids:
            raise RuntimeError("DSA requires an IndexerKVSpec group for indexer residency")
        self._indexer_group_id = int(indexer_group_ids[0])
        return self._indexer_group_id

    def _get_group_num_free_blocks(self, block_pool, group_id: int) -> int:
        if hasattr(block_pool, "block_pools"):
            return block_pool.block_pools[group_id].get_num_free_blocks()
        return block_pool.get_num_free_blocks()

    def get_full_attention_group_id(self, kv_cache_config) -> int:
        return self._get_full_attention_group_id(kv_cache_config)

    def should_release_full_cache_after_prefill(self, request) -> bool:
        if request.num_prompt_tokens <= self._enable_dsa_prompt_len:
            return False
        # Once sparse cache is enabled, completed prefill MLA full blocks must
        # enter the DRAM-resident path before sparse decode can begin.
        return request.num_computed_tokens >= request.num_prompt_tokens

    def release_prefill_full_cache_except_tail(
        self,
        kv_cache_manager: KVCacheManager,
        request: Request,
    ) -> bool:
        """Release dense prefill full-cache blocks while keeping an unfull tail.

        DSA sparse decode reuses the prefill tail block as the final block in
        the full/MLA block table. Releasing the entire full-cache group at the
        prefill/decode boundary would also free that tail, forcing the first
        decode step to allocate an empty replacement block and losing the tail
        KV data.
        """
        full_group_id = self._get_full_attention_group_id(
            kv_cache_manager.kv_cache_config)
        full_manager = kv_cache_manager.coordinator.single_type_managers[
            full_group_id]
        request_id = request.request_id
        req_blocks = full_manager.req_to_blocks.get(request_id)
        if not req_blocks:
            return False

        preserve_tail_block = request.num_prompt_tokens % self._vllm_blk_size != 0
        preserved_tail_block = self._release_full_blocks_except_tail(
            full_manager, request_id, preserve_tail_block)
        self._append_preserved_tail_block(
            full_manager, request_id, preserved_tail_block)
        return True

    def build_dsa_meta(
        self,
        input_batch: NPUInputBatch,
        full_block_table_tensor: torch.Tensor | None,
        graph_row_count: int | None = None,
    ) -> bool:
        """Build DSA request metadata once for the current model forward.

        This is the physical-adapter boundary between the canonical
        ``NPUInputBatch.dsa_state`` row projection and the DSA sparse-cache
        runtime.  Eager forwards acquire/bind missing resident rows and build
        active-prefix tensors; graph forwards serialize the same row semantics
        into captured-prefix persistent buffers with explicit PAD rows.

        It intentionally does not compute indexer scores, choose replacement
        tokens, move KV cache data, or update layer resident mappings. Those
        actions happen later in the attention hooks and backend DSA ops.
        """
        input_state = input_batch.dsa_state
        if (input_state is None
                or not input_state.matches_input_batch(
                    input_batch, int(input_batch.num_reqs))):
            raise RuntimeError(
                "DSA InputBatch state was not refreshed before metadata build")
        full_attention_group_id = int(input_state.full_attention_group_id)
        if full_attention_group_id < 0:
            raise RuntimeError(
                "DSA InputBatch state has no MLA/full attention group")
        if graph_row_count is not None:
            # Graph decode has a stricter contract than eager: every active
            # model row is one decode token and all resident/DRAM resources
            # must already own fixed-capacity backing storage. DENSE, ENTER and
            # SPARSE differ only in row metadata; decode full-block dump is
            # represented by fixed src/dst rows and remains graphable.
            # graph_row_count is the captured (possibly upward-padded) count;
            # the serializer reads the actual count and materializes PAD rows.
            reserve_decode_full_block_dumps_from_input_state(
                input_state,
                dram_store=self.dsa_hot_kv_store,
            )
            return bool(
                self.prepare_row_mode_decode_graph_replay_batch(
                    input_state,
                    input_batch,
                    full_attention_group_id,
                    int(graph_row_count),
                ))

        input_state.ensure_resident_resources(
            resident_token_pool=self.resident_token_pool,
            dram_store=self.dsa_hot_kv_store,
        )
        reserve_decode_full_block_dumps_from_input_state(
            input_state,
            dram_store=self.dsa_hot_kv_store,
        )
        row_mode_batch = self.prepare_row_mode_decode_eager_batch(input_state)
        if row_mode_batch is None:
            raise RuntimeError(
                "DSA LIDU/KSC currently supports prefill-only forwards or "
                "decode-only request rows; prefill/decode mixed layouts are "
                "unsupported")
        else:
            lifecycle_plan = build_forward_layer_hook_plan_from_input_state(
                input_state,
                dram_store=self.dsa_hot_kv_store,
                # Optimized row-mode eager has already serialized decode dump
                # src/dst into the shared CpuGpuBuffer. Keep this builder for
                # dynamic prefill dump jobs only; rebuilding decode tensors here
                # would duplicate the native eager/graph staging path.
                include_decode_dump=False,
                tensor_device=self.resident_token_pool.device,
                empty_plan=self._empty_forward_layer_hook_plan,
            )
            # Optimized row-mode serialization owns decode-boundary jobs,
            # while the lifecycle adapter still owns final-prefill jobs.  A
            # prefill-only forward deliberately returns an empty row-mode
            # batch, so replacing the lifecycle batch with that empty view
            # would drop its real KV payload copy jobs.
            #
            # The optimized branch only admits either prefill-only (zero
            # decode rows) or decode-only identity rows.  If both producers
            # become active, the execution contract has expanded to a mixed
            # layout and must be handled explicitly instead of silently
            # choosing one physical copy batch.
            lifecycle_dump_batch = lifecycle_plan.full_block_dump_batch
            decode_dump_batch = row_mode_batch.full_block_dump_batch
            if lifecycle_dump_batch and decode_dump_batch:
                raise RuntimeError(
                    "DSA optimized row-mode path produced both prefill and "
                    "decode full-block dump batches")
            full_block_dump_batch = (
                decode_dump_batch
                if decode_dump_batch else lifecycle_dump_batch)
            has_full_block_dump = bool(full_block_dump_batch)
            if (lifecycle_plan is self._empty_forward_layer_hook_plan
                    and not has_full_block_dump):
                self.forward_layer_hook_plan = (
                    self._empty_forward_layer_hook_plan)
            else:
                self.forward_layer_hook_plan = DSAForwardLayerHookPlan(
                    full_block_dump_batch=full_block_dump_batch,
                )
        self.forward_row_mode_decode_batch = row_mode_batch
        return True

    """
    EngineCore Scheduler侧逻辑
    """
    def request_begin(self, request_id, prompt_token_ids) -> int:
        token_id_count = (
            len(prompt_token_ids) if prompt_token_ids is not None else 0)
        logger.debug(
            "========== DSA TOKENIZED PROMPT =========="
            " req_id=%s prompt_token_id_len=%s block_size=%s "
            "sparse_threshold=%s",
            request_id,
            token_id_count,
            self._vllm_blk_size,
            self._enable_dsa_prompt_len,
        )
        return self._select_target_resident_budget_tokens(token_id_count)

    def request_finished_in_scheduler(self, request_id):
        pass

    """
    Worker侧逻辑
    """
    def execute_decode_selection_pipeline(
        self,
        layer_name: str,
        *,
        forward_batch: DSAForwardRowModeDecodeBatch | None,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        row_indices: torch.Tensor | None = None,
    ) -> DSAOffloadSelectionOutput | None:
        """Run the full-batch LIDU -> KSC decode data plane for one layer.

        All request-row metadata and output tensors are forward-level views
        owned by the shared row-mode buffer. The only layer-specific state is
        the LIDU ``cache_slots`` row map and this layer's HBM/DRAM cache planes.
        DENSE, ENTER/first-fill, SPARSE/steady and graph PAD rows remain in one
        physical batch and are interpreted by ``row_modes`` plus per-row LIDU
        metadata; Python never partitions the batch by stage or budget.
        """
        if forward_batch is None or not forward_batch:
            return None
        layer_id = int(layer_name.split(".")[2])
        cache_zones = self.layer_cache_registry.require(layer_id)
        dram_store = self.dsa_hot_kv_store
        if dram_store is None:
            raise RuntimeError(
                "DSA LIDU/KSC decode requires a hot-DRAM block store")

        outputs = forward_batch.lidu_outputs_for_layer(layer_id)
        req_pool_entries = forward_batch.resident_pool_indices_tensor
        row_modes = forward_batch.row_modes_tensor
        hbm_block_table = forward_batch.batch_hbm_block_table
        dram_block_table = forward_batch.batch_dram_block_table
        if row_indices is not None:
            row_indices = row_indices.to(
                device=req_pool_entries.device,
                dtype=torch.long,
            )
            req_pool_entries = req_pool_entries.index_select(
                0, row_indices)
            row_modes = row_modes.index_select(0, row_indices)
            hbm_block_table = hbm_block_table.index_select(
                0, row_indices)
            dram_block_table = dram_block_table.index_select(
                0, row_indices)
            outputs = DSALightningIndexerUpdateBuffers(
                topk_index=outputs.topk_index.index_select(
                    0, row_indices),
                topk_slots=outputs.topk_slots.index_select(
                    0, row_indices),
                miss_count=outputs.miss_count.index_select(
                    0, row_indices),
                tail_info=outputs.tail_info.index_select(
                    0, row_indices),
            )
        self.ops_backend.lightning_indexer_decode_update(
            query=query,
            key=key,
            weights=weights,
            req_pool_entries=req_pool_entries,
            cache_slots=self.resident_token_pool.get_cache_slots(
                layer_id=layer_id),
            row_modes=row_modes,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            outputs=outputs,
        )
        self.ops_backend.kvcache_scatter_copy(
            nopek_cache_zone=cache_zones.nopek_cache_zone,
            ropek_cache_zone=cache_zones.ropek_cache_zone,
            nopek_dram_arena=dram_store.get_arena(
                layer_id, BlockType.NOPE_K),
            ropek_dram_arena=dram_store.get_arena(
                layer_id, BlockType.ROPE_K),
            hbm_block_table=hbm_block_table,
            dram_block_table=dram_block_table,
            src_token_ids=outputs.topk_index,
            dst_slots=outputs.topk_slots,
            copy_counts=outputs.miss_count,
        )
        return DSAOffloadSelectionOutput(
            sparse_indices=outputs.topk_slots,
            tail_info=outputs.tail_info,
        )

    def sparse_attention_for_offload(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        selection: DSAOffloadSelectionOutput,
        scale_value: float,
        block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
    ) -> torch.Tensor:
        return self.ops_backend.sparse_flash_attention_for_offload(
            query=query,
            key=key,
            sparse_indices=selection.sparse_indices,
            tail_info=selection.tail_info,
            scale_value=scale_value,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=key_rope,
        )

    # Layer-level DSA hook before MLA/SFA.
    # Current responsibilities:
    # 1. bind this layer's cache zones for later dump/load;
    # 2. expose the zones to the post-attention full-block dump hook.
    def attention_begin(self, layer_name, forward_context: ForwardContext):
        layer_id = int(layer_name.split(".")[2])
        layer_cache_zones = self.layer_cache_registry.get(layer_id)
        if layer_cache_zones is None:
            resolved_cache_zones = resolve_layer_cache_zones(layer_name,
                                                             forward_context)
            # Cache zones are worker-lifetime resources. Resolve them only on
            # first sight; later forwards use the registry fast path to avoid
            # repeated Python object traversal and tensor identity checks.
            layer_cache_zones = self.layer_cache_registry.bind_or_validate(
                layer_id, resolved_cache_zones)

    def _dump_layer_full_blocks_to_dram_batch(
        self,
        layer_id: int,
        cache_zones: LayerCacheZones,
        hook_plan: DSAForwardLayerHookPlan,
    ) -> None:
        """Dump this layer's newly completed MLA blocks with one custom op.

        Request/hash ownership was resolved once during model-forward setup.
        Eager supplies a compact active batch; graph capture supplies the same
        contract as a fixed captured-row view padded with destination ``-1``.
        No request, logical-block or copy-run traversal remains layer-wise.
        """
        dump_batch = hook_plan.full_block_dump_batch
        if dump_batch:
            dram_store = self.dsa_hot_kv_store
            if dram_store is None:
                raise RuntimeError(
                    "DSA full-block dump batch has rows but no DRAM store")
            nopek_dram_arena = dram_store.get_arena(
                layer_id, BlockType.NOPE_K)
            ropek_dram_arena = dram_store.get_arena(
                layer_id, BlockType.ROPE_K)
            self.ops_backend.dump_full_kv_cache_blocks(
                nopek_cache_zone=cache_zones.nopek_cache_zone,
                ropek_cache_zone=cache_zones.ropek_cache_zone,
                nopek_dram_arena=nopek_dram_arena,
                ropek_dram_arena=ropek_dram_arena,
                src_hbm_block_ids=(
                    dump_batch.src_hbm_block_ids_tensor),
                dst_dram_block_ids=(
                    dump_batch.dst_dram_block_ids_tensor),
            )

    # Layer-level DSA hook after MLA/SFA.
    # Current responsibilities:
    # 1. dump this layer's prefill/decode newly-full MLA blocks to hot DRAM;
    # Token-level sparse selection/materialization remains in
    # execute_decode_selection_pipeline.
    # A block completed by the current query remains this forward's dense tail;
    # it becomes a LIDU candidate only on the next forward, so post-attention
    # dump preserves the original lifecycle without delaying current
    # LIDU/KSC/SFA-Offload.
    #
    # 当前 DSA 只支持单流有序执行：MLA cache 写入、满块 dump，以及后续
    # forward 的 LIDU/KSC 消费都提交到同一个 current NPU stream。因而依赖由 stream
    # ordering 保证，不需要 CPU readiness 表，也不能把“算子已下发”误当成
    # “设备已完成”。若未来将 dump 放到独立 stream，必须用 NPU event/wait
    # 或真实完成状态建立跨流依赖，并延后相关 HBM block 的复用。
    def attention_finished(self, layer_name: str):
        hook_plan = self.forward_layer_hook_plan
        dump_batch = hook_plan.full_block_dump_batch
        if not dump_batch:
            # Eager decode does not need to launch the no-op operator.
            # Graph capture/replay uses a fixed-size dump batch (dst=-1), so
            # it deliberately does not take this branch and keeps the dump
            # node in the captured topology.
            return

        layer_id = int(layer_name.split(".")[2])
        cache_zones = self.layer_cache_registry.require(layer_id)
        self._dump_layer_full_blocks_to_dram_batch(
            layer_id,
            cache_zones,
            hook_plan,
        )

    def request_finished_in_worker(self, request_id):
        self.resident_token_pool.release(request_id)
        self.dsa_hot_kv_store.release_request(request_id)

    def request_preempted_in_worker(self, request_id):
        if self.resident_token_pool.get_index(request_id) is not None:
            self.resident_token_pool.clear_request(request_id)

    def execute_begin(self, scheduler_output: SchedulerOutput):
        pass

    def execute_finished(self):
        pass

    def _get_sparse_tail_slots_need(
        self,
        request: Request,
        total_tokens: int | None = None,
    ) -> int:
        if total_tokens is None:
            total_tokens = int(request.num_tokens)
        else:
            total_tokens = int(total_tokens)
        if total_tokens <= 0:
            return 0
        full_blocks_before_tail = (total_tokens - 1) // self._vllm_blk_size
        return total_tokens - full_blocks_before_tail * self._vllm_blk_size

    def _should_preserve_sparse_tail_block(
        self,
        request: Request,
        dense_new_tokens: int,
    ) -> bool:
        del dense_new_tokens
        return (
            int(request.num_computed_tokens) % self._vllm_blk_size != 0
        )

    def _release_full_blocks_except_tail(
        self,
        full_manager,
        request_id: ReqType,
        preserve_tail_block: bool,
    ) -> KVCacheBlock | None:
        req_blocks = full_manager.req_to_blocks.get(request_id)
        if not req_blocks:
            return None

        tail_block = req_blocks[-1] if preserve_tail_block else None
        full_blocks_to_release = req_blocks[:-1]
        if not preserve_tail_block:
            full_blocks_to_release = req_blocks
        full_manager.req_to_blocks[request_id] = []
        if full_blocks_to_release:
            full_manager.block_pool.free_blocks(
                list(reversed(full_blocks_to_release)))
        full_manager.num_cached_block.pop(request_id, None)
        return tail_block

    @staticmethod
    def _append_preserved_tail_block(
        full_manager,
        request_id: ReqType,
        preserved_tail_block: KVCacheBlock | None,
    ) -> None:
        if preserved_tail_block is None:
            return
        req_blocks = full_manager.req_to_blocks[request_id]
        req_blocks.append(preserved_tail_block)

    def dsa_alloc_slots_wrap(
        self,
        kv_cache_manager: KVCacheManager,
        request: Request,
        resident_valid_seq_len: int,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
    ) -> KVCacheBlocks | None:
        def allocate_dense() -> KVCacheBlocks | None:
            request.dsa_next_req_stage = (
                ReqStage.PREFILL if request.num_output_tokens == 0
                else ReqStage.DENSE_DECODE)
            request.dsa_resident_valid_seq_len = INVALID_SLOT
            request.dsa_sparse_budget_tokens = 0
            dense_blocks = kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=delay_cache_blocks,
                num_encoder_tokens=num_encoder_tokens,
                full_sequence_must_fit=full_sequence_must_fit,
                reserved_blocks=reserved_blocks,
            )
            if dense_blocks is not None:
                request.dsa_req_stage = ReqStage.coerce(
                    request.dsa_next_req_stage)
            return dense_blocks

        if (request.num_computed_tokens < request.num_prompt_tokens
                or resident_valid_seq_len == INVALID_SLOT):
            return allocate_dense()

        if (num_new_computed_tokens > 0
                or num_external_computed_tokens > 0
                or num_lookahead_tokens > 0
                or delay_cache_blocks
                or num_encoder_tokens > 0
                or full_sequence_must_fit
                or reserved_blocks > 0):
            return allocate_dense()

        else:
            coordinator = kv_cache_manager.coordinator
            block_pool = kv_cache_manager.block_pool
            has_group_pools = hasattr(block_pool, "block_pools")

            # ENTER_SPARSE_DECODE shrinks the full/MLA table to sparse-budget
            # blocks plus an optional unfilled tail block. This covers both the
            # old long-prompt first decode and the short-prompt long-decode
            # transition once the sequence crosses the sparse threshold.
            full_group_id = self._get_full_attention_group_id(
                kv_cache_manager.kv_cache_config)
            indexer_group_id = self._get_indexer_group_id(
                kv_cache_manager.kv_cache_config)
            full_manager = coordinator.single_type_managers[full_group_id]
            indexer_manager = coordinator.single_type_managers[indexer_group_id]
            dense_computed_tokens = (
                request.num_computed_tokens
                + max(0, int(num_new_computed_tokens))
                + max(0, int(num_external_computed_tokens)))
            dense_num_tokens_need_slot = min(
                dense_computed_tokens
                + max(0, int(num_new_tokens))
                + max(0, int(num_lookahead_tokens)),
                kv_cache_manager.max_model_len,
            )
            indexer_blocks_to_allocate = (
                indexer_manager.get_num_blocks_to_allocate(
                    request_id=request.request_id,
                    num_tokens=dense_num_tokens_need_slot,
                    new_computed_blocks=[],
                    total_computed_tokens=dense_computed_tokens,
                    num_tokens_main_model=dense_num_tokens_need_slot,
                ))
            if (has_group_pools
                    and indexer_blocks_to_allocate > self._get_group_num_free_blocks(
                        block_pool, indexer_group_id)):
                return None

            req_stage = ReqStage.coerce(request.dsa_next_req_stage)
            reset_full_cache = req_stage.is_enter_sparse_decode
            preserved_tail_block = None
            sparse_budget_slots = resident_valid_seq_len
            if reset_full_cache:
                if request.num_computed_tokens < request.num_prompt_tokens:
                    return allocate_dense()
                tail_slots_need = self._get_sparse_tail_slots_need(
                    request,
                    dense_num_tokens_need_slot,
                )
                preserve_tail_block = self._should_preserve_sparse_tail_block(
                    request, num_new_tokens)
                existing_full_blocks = full_manager.req_to_blocks.get(
                    request.request_id, [])
                will_preserve_tail = (
                    preserve_tail_block and bool(existing_full_blocks))
                sparse_budget_slots = (
                    max(0, resident_valid_seq_len - tail_slots_need)
                    if will_preserve_tail
                    else resident_valid_seq_len)
                releasable_full_blocks = max(
                    0,
                    len(existing_full_blocks)
                    - (1 if will_preserve_tail else 0),
                )
                sparse_budget_blocks_need = (
                    (sparse_budget_slots + full_manager.block_size - 1)
                    // full_manager.block_size
                    if sparse_budget_slots > 0 else 0)
                if has_group_pools:
                    full_blocks_available_after_release = (
                        self._get_group_num_free_blocks(
                            block_pool, full_group_id)
                        + releasable_full_blocks)
                    if (sparse_budget_blocks_need
                            > full_blocks_available_after_release):
                        return None
                else:
                    blocks_available_after_release = (
                        block_pool.get_num_free_blocks()
                        + releasable_full_blocks)
                    if (sparse_budget_blocks_need
                            + max(0, indexer_blocks_to_allocate)
                            > blocks_available_after_release):
                        return None
                preserved_tail_block = self._release_full_blocks_except_tail(
                    full_manager, request.request_id, preserve_tail_block)

            num_blocks_to_allocate = full_manager.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=sparse_budget_slots,
                new_computed_blocks=[],
                total_computed_tokens=sparse_budget_slots,
                num_tokens_main_model=sparse_budget_slots,
            )
            if has_group_pools:
                has_enough_blocks = (
                    num_blocks_to_allocate <= self._get_group_num_free_blocks(
                        block_pool, full_group_id))
            else:
                has_enough_blocks = (
                    max(0, num_blocks_to_allocate)
                    + max(0, indexer_blocks_to_allocate)
                    <= block_pool.get_num_free_blocks())
            if not has_enough_blocks:
                if reset_full_cache:
                    raise RuntimeError(
                        "DSA sparse allocation capacity precheck passed but "
                        "post-release capacity check failed")
                self._append_preserved_tail_block(
                    full_manager, request.request_id, preserved_tail_block)
                return None
            full_manager.allocate_new_blocks(
                request.request_id,
                sparse_budget_slots,
                sparse_budget_slots,
            )
            self._append_preserved_tail_block(
                full_manager, request.request_id, preserved_tail_block)
            # Indexer cache is the dense selector plane. It must keep a full
            # block table for the original sequence in HBM and must not follow
            # the sparse full/MLA table shrink/replace policy.
            indexer_manager.allocate_new_blocks(
                request.request_id,
                dense_num_tokens_need_slot,
                dense_num_tokens_need_slot,
            )
            request.dsa_req_stage = req_stage
            request.dsa_next_req_stage = req_stage
            request.dsa_resident_valid_seq_len = resident_valid_seq_len
            return KVCacheBlocks(coordinator.get_blocks(request.request_id))


    def plan_decode_resident_slots(
        self,
        request: Request,
        num_new_tokens: int = 1,
    ):
        # This scheduler-side planner is the single stage-advance point for DSA
        # cache layout. It both returns the resident MLA/full-cache slot count
        # for sparse decode and writes the request stage metadata consumed by
        # worker hooks. Keep this state transition out of layer-wise code.
        previous_stage = ReqStage.coerce(request.dsa_req_stage)
        dense_stage = (
            ReqStage.PREFILL if request.num_output_tokens == 0
            else ReqStage.DENSE_DECODE)
        request.dsa_next_req_stage = dense_stage
        request.dsa_resident_valid_seq_len = INVALID_SLOT
        request.dsa_sparse_budget_tokens = 0
        if not self._is_sparse_cache_enabled():
            return INVALID_SLOT
        if request.num_computed_tokens < request.num_prompt_tokens:
            return INVALID_SLOT
        if request.num_output_tokens == 0:  # prefill/chunked_prefill
            return INVALID_SLOT
        if request.num_output_placeholders or request.has_encoder_inputs:
            return INVALID_SLOT
        target_budget_tokens = int(
            request.dsa_target_resident_budget_tokens or 0)
        if target_budget_tokens <= 0:
            target_budget_tokens = self._select_target_resident_budget_tokens(
                request.num_prompt_tokens)
            request.dsa_target_resident_budget_tokens = target_budget_tokens
        total_tokens = min(
            int(self._vllm_config.model_config.max_model_len),
            int(request.num_computed_tokens)
            + max(0, int(num_new_tokens)),
        )
        if total_tokens <= self._enable_dsa_prompt_len:
            return INVALID_SLOT

        block_size = self._vllm_blk_size
        full_blocks_before_tail = (total_tokens - 1) // block_size
        tail_slots_need = total_tokens - full_blocks_before_tail * block_size
        if full_blocks_before_tail <= 0:
            return INVALID_SLOT
        return self._plan_sparse_decode_resident_slots(
            request=request,
            candidate_full_blocks=full_blocks_before_tail,
            tail_slots_need=tail_slots_need,
            previous_stage=previous_stage,
            target_budget_tokens=target_budget_tokens,
            total_tokens=total_tokens,
        )

    def _plan_sparse_decode_resident_slots(
            self,
            request: Request,
            candidate_full_blocks: int,
            tail_slots_need: int,
            previous_stage: ReqStage,
            target_budget_tokens: int,
            total_tokens: int | None = None,
    ) -> int:
        block_size = self._vllm_blk_size
        if total_tokens is None:
            total_tokens = int(request.num_tokens)
        else:
            total_tokens = int(total_tokens)
        candidate_tokens = candidate_full_blocks * block_size
        if candidate_tokens <= 0:
            return INVALID_SLOT

        sparse_budget_tokens = self._get_sparse_budget_tokens(
            candidate_tokens, target_budget_tokens)
        if sparse_budget_tokens <= 0:
            return INVALID_SLOT

        resident_valid_seq_len = sparse_budget_tokens + tail_slots_need
        next_stage = (
            ReqStage.SPARSE_DECODE
            if previous_stage.is_sparse_decode
            else ReqStage.ENTER_SPARSE_DECODE)
        request.dsa_next_req_stage = next_stage
        if next_stage.is_enter_sparse_decode:
            logger.debug(
                "========== DSA DECODE REACHED SPARSE THRESHOLD =========="
                " req_id=%s prompt_tokens=%s output_tokens=%s total_tokens=%s "
                "computed_tokens=%s candidate_full_blocks=%s tail_slots=%s "
                "sparse_budget=%s resident_valid_seq_len=%s block_size=%s "
                "sparse_threshold=%s",
                request.request_id,
                request.num_prompt_tokens,
                request.num_output_tokens,
                total_tokens,
                request.num_computed_tokens,
                candidate_full_blocks,
                tail_slots_need,
                sparse_budget_tokens,
                resident_valid_seq_len,
                block_size,
                self._enable_dsa_prompt_len,
            )
        request.dsa_sparse_budget_tokens = sparse_budget_tokens
        request.dsa_resident_valid_seq_len = resident_valid_seq_len
        return request.dsa_resident_valid_seq_len
