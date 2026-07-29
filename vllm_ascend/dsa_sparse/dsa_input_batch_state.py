"""DSA 在当前 ``NPUInputBatch`` 行序上的单一 forward 语义投影。

vLLM 的 ``InputBatch`` 是 worker 侧当前驻留请求的 CPU 行账本。请求经过
remove/add、condense 和 reorder 后，行号只在本轮最终 batch 内有效。DSA
不能再分别从 scheduler 字典、CachedRequestState 和 graph staging 中重复
推导同一套 stage/budget/tail 语义，否则 eager 与 graph 很容易发生漂移。

``DSAInputBatchState`` 因此只做一件事：在最终 InputBatch 行序稳定后，把
scheduler/request/resource 真源一次性投影为固定容量、row-aligned 的本轮
forward 状态。它是 eager、graph gate 和 graph replay 共同读取的本轮
**语义真源投影**，但不是新的请求生命周期账本：

* stage/resident/budget 的真源仍是 SchedulerOutput；
* token、block id 和 full-block hash 的真源仍是 CachedRequestState；
* resident pool 与 DRAM store 仍拥有资源生命周期；
* graph slab 仍只是固定地址的物理镜像。

eager adapter、graph gate 和 graph serializer 共同只读本对象，避免各自再
解释一次请求语义。它还在同一次 O(B) refresh 中识别 final-prefill 和
single-token decode 的满块边界，保存 row-aligned 的 source/logical-index
列；DRAM 目的块预留仍在后续 model-forward 控制面完成。

每轮整体刷新而不增量跟随 InputBatch 的 swap/condense，是为了让 hole 复用、
preempt/resume 和 continuous batching 下的最终行序始终安全。图准入前的
refresh 不得偷偷 acquire resident/DRAM 资源。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vllm_ascend.dsa_sparse.dsa_types import (
    INVALID_SLOT,
    KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID,
    DSADecodeRowMode,
    ReqStage,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    from vllm_ascend.dsa_sparse.dsa_hot_kv_store_core import DSAHotKVStore
    from vllm_ascend.dsa_sparse.dsa_resident_pool import DSAResidentTokenPool


class DSAInputBatchState:
    """Fixed-capacity DSA columns attached to exactly one InputBatch.

    Native request identity and token-count columns remain owned by
    ``InputBatch``. This component owns only DSA-specific projections and
    borrows the native columns through typed properties, so eager and graph
    consumers cannot observe a stale duplicate after row compaction.
    """

    def __init__(self, input_batch: InputBatch, block_size: int) -> None:
        max_num_reqs = int(input_batch.max_num_reqs)
        block_size = int(block_size)
        if max_num_reqs <= 0:
            raise ValueError(
                f"max_num_reqs must be positive, got {max_num_reqs}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.input_batch = input_batch
        self.max_num_reqs = max_num_reqs
        self.block_size = block_size
        self.row_count = 0
        self.valid = False
        self.query_layout_bound = False
        self.full_attention_group_id = -1
        self.sparse_row_count = 0
        # Variable-length references point at existing request/resource
        # ledgers. They do not copy or own request-lifetime block/hash state.
        self.budget_block_id_rows: list[list[int] | None] = (
            [None] * max_num_reqs)
        self.context_full_block_hash_rows: list[list | None] = (
            [None] * max_num_reqs)

        # Numeric fields are fixed-capacity NumPy rows, matching the baseline
        # InputBatch pattern: mutate values per forward, keep storage stable.
        self.num_scheduled_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_output_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.stages = np.zeros(max_num_reqs, dtype=np.int8)
        # Scheduler resident length retains INVALID_SLOT for dense rows. The
        # normalized attention length is already owned by ModelRunner's
        # resident_valid_seq_lens_cpu and is deliberately not duplicated here.
        self.scheduler_resident_lens = np.full(
            max_num_reqs, INVALID_SLOT, dtype=np.int32)
        self.sparse_budget_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.target_resident_budget_tokens = np.zeros(
            max_num_reqs, dtype=np.int32)
        self.resident_pool_indices = np.full(
            max_num_reqs, INVALID_SLOT, dtype=np.int32)
        self.row_modes = np.full(
            max_num_reqs, int(DSADecodeRowMode.PAD), dtype=np.int32)
        self.sparse_mask = np.zeros(max_num_reqs, dtype=np.bool_)
        self.sparse_row_indices = np.zeros(max_num_reqs, dtype=np.int32)

        self.budget_slot_counts = np.zeros(max_num_reqs, dtype=np.int32)
        self.candidate_range_ends = np.zeros(max_num_reqs, dtype=np.int32)
        # 当前请求在 dense indexer 平面上的有效 key 长度。该列只用于图准入
        # 和 row-mode 物理表宽校验；LIDU 从原生 attention metadata 直接消费
        # 对应的 device tensor，不在这里维护第二份算子输入。
        self.indexer_key_lens = np.zeros(max_num_reqs, dtype=np.int32)
        self.full_block_dump_mask = np.zeros(max_num_reqs, dtype=np.bool_)
        self.last_prefill_chunk_mask = np.zeros(max_num_reqs, dtype=np.bool_)
        # Decode full-block dump is detected in the existing refresh row loop.
        # Keep both a compact row list for O(number_of_dumps) control-plane
        # reservation and row-aligned src/dst columns for eager/graph physical
        # serialization. The generic dump operator uses destination id -1 as
        # no-op; DSA's logical DRAM tables independently reserve block 0.
        # Number of request rows that reach a full-block boundary in this
        # forward. This is a semantic boundary count, not necessarily the
        # number of physical copies: reservation may resolve an already
        # completed hash and publish destination -1. The row-mode serializer
        # filters those hits into a compact physical copy-job prefix.
        self.decode_dump_row_count = 0
        self.decode_dump_reservations_ready = False
        self.decode_dump_row_indices = np.zeros(
            max_num_reqs, dtype=np.int32)
        self.decode_dump_src_hbm_block_ids = np.zeros(
            max_num_reqs, dtype=np.int32)
        self.decode_dump_dst_dram_block_ids = np.full(
            max_num_reqs,
            KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID,
            dtype=np.int32,
        )
        self.decode_dump_logical_block_indices = np.full(
            max_num_reqs, INVALID_SLOT, dtype=np.int32)

        self.query_start_locs = np.zeros(max_num_reqs, dtype=np.int32)
        self.query_lens = np.zeros(max_num_reqs, dtype=np.int32)

    @property
    def request_ids(self) -> list[str]:
        """Native InputBatch request rows; DSA never owns a second copy."""
        return self.input_batch.req_ids

    @property
    def num_prompt_tokens(self) -> np.ndarray:
        return self.input_batch.num_prompt_tokens

    @property
    def num_computed_tokens(self) -> np.ndarray:
        return self.input_batch.num_computed_tokens_cpu

    @property
    def context_lens(self) -> np.ndarray:
        return self.input_batch.num_tokens_no_spec

    def _project_row_cache_semantics(
        self,
        row: int,
        *,
        num_prompt_tokens: int,
        num_output_tokens: int,
        num_computed_tokens: int,
        resident_valid_seq_len: int,
        context_full_block_hashes: list,
        budget_block_ids: list[int],
        sparse_enabled: bool,
        sparse_budget_tokens: int,
    ) -> bool:
        """Write one request's budget/tail/candidate semantics in place.

        These values used to be materialized as a short-lived
        ``ReqForwardPlan`` and immediately copied into the sidecar. Writing
        the authoritative columns directly keeps the formulas explicit while
        avoiding one Python object per request and per model forward.
        """
        block_size = self.block_size
        sparse_budget_tokens = max(0, int(sparse_budget_tokens or 0))
        is_sparse_decode = bool(
            sparse_enabled
            and int(num_output_tokens) > 0
            and int(resident_valid_seq_len) >= 0
            and sparse_budget_tokens > 0)

        total_budget_slots = len(budget_block_ids) * block_size
        budget_slot_count = (
            min(total_budget_slots, sparse_budget_tokens)
            if is_sparse_decode else total_budget_slots)
        budget_slot_count = max(0, int(budget_slot_count))

        # The cache contains exactly ``num_computed_tokens`` before this
        # forward. This remains correct for a request-major MTP query with
        # several verification tokens, unlike deriving the boundary from the
        # number of already accepted output tokens.
        dense_tokens_before_query = int(num_computed_tokens)
        dense_tail_start = (
            dense_tokens_before_query // block_size) * block_size
        if is_sparse_decode:
            dumped_full_token_end = (
                len(context_full_block_hashes) * block_size)
            candidate_range_end = min(
                dumped_full_token_end, dense_tail_start)
        else:
            candidate_range_end = dense_tail_start

        self.sparse_mask[row] = is_sparse_decode
        self.row_modes[row] = int(
            DSADecodeRowMode.SPARSE
            if is_sparse_decode else DSADecodeRowMode.DENSE)
        self.budget_slot_counts[row] = budget_slot_count
        self.candidate_range_ends[row] = candidate_range_end
        return is_sparse_decode

    def clear(self) -> None:
        """Drop forward-scoped references when the runner has no work."""
        old_row_count = self.row_count
        for row in range(old_row_count):
            self.budget_block_id_rows[row] = None
            self.context_full_block_hash_rows[row] = None
        if old_row_count > 0:
            old_rows = slice(0, old_row_count)
            self.row_modes[old_rows] = int(DSADecodeRowMode.PAD)
            self.sparse_mask[old_rows] = False
            self.resident_pool_indices[old_rows] = INVALID_SLOT
            self.scheduler_resident_lens[old_rows] = INVALID_SLOT
            self.decode_dump_src_hbm_block_ids[old_rows] = 0
            self.decode_dump_dst_dram_block_ids[old_rows] = (
                KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID)
            self.decode_dump_logical_block_indices[old_rows] = INVALID_SLOT
        self.row_count = 0
        self.sparse_row_count = 0
        self.decode_dump_row_count = 0
        self.decode_dump_reservations_ready = False
        self.valid = False
        self.query_layout_bound = False
        self.full_attention_group_id = -1

    def refresh(
        self,
        *,
        scheduler_output: SchedulerOutput,
        requests: dict[str, CachedRequestState],
        num_scheduled_tokens_by_row: np.ndarray,
        full_attention_group_id: int,
        indexer_group_id: int,
        resident_token_pool: DSAResidentTokenPool,
    ) -> None:
        """Rebuild the active row prefix after InputBatch row order is final."""
        input_batch = self.input_batch
        row_count = int(input_batch.num_reqs)
        if row_count < 0 or row_count > self.max_num_reqs:
            raise RuntimeError(
                "DSA InputBatch rows exceed sidecar capacity: "
                f"rows={row_count}, capacity={self.max_num_reqs}")

        old_row_count = self.row_count
        self.valid = False
        self.query_layout_bound = False
        self.row_count = row_count
        self.full_attention_group_id = int(full_attention_group_id)
        indexer_group_id = int(indexer_group_id)
        self.sparse_row_count = 0
        self.decode_dump_row_count = 0
        self.decode_dump_reservations_ready = False

        scheduled_rows = np.asarray(num_scheduled_tokens_by_row).reshape(-1)
        if int(scheduled_rows.size) < row_count:
            raise RuntimeError(
                "DSA scheduled-token rows are shorter than InputBatch: "
                f"rows={int(scheduled_rows.size)}, batch_rows={row_count}")
        resident_lens = (
            scheduler_output.req_dsa_resident_valid_seq_len or {})
        sparse_budgets = (
            scheduler_output.req_dsa_sparse_budget_tokens or {})
        target_budgets = (
            scheduler_output.req_dsa_target_resident_budget_tokens or {})
        req_stages = scheduler_output.req_dsa_stage or {}
        for row in range(row_count):
            req_id = input_batch.req_ids[row]
            if req_id is None or req_id not in requests:
                raise RuntimeError(
                    "DSA InputBatch row has no matching worker request: "
                    f"row={row}, req_id={req_id!r}")
            req_state = requests[req_id]
            req_block_ids = req_state.block_ids
            if self.full_attention_group_id >= len(req_block_ids):
                raise RuntimeError(
                    "DSA InputBatch state is missing the MLA/full block group "
                    f"for request {req_id}: group={self.full_attention_group_id}, "
                    f"total_groups={len(req_block_ids)}")
            if indexer_group_id >= len(req_block_ids):
                raise RuntimeError(
                    "DSA InputBatch state is missing the indexer block group "
                    f"for request {req_id}: group={indexer_group_id}, "
                    f"total_groups={len(req_block_ids)}")

            budget_block_ids = req_block_ids[self.full_attention_group_id]
            context_full_blk_hashes = req_state.context_full_blk_hashes or []
            num_prompt = int(self.num_prompt_tokens[row])
            num_computed = int(self.num_computed_tokens[row])
            context_len = int(self.context_lens[row])
            expected_full_blocks = num_computed // self.block_size
            if len(context_full_blk_hashes) < expected_full_blocks:
                raise RuntimeError(
                    "DSA full-block hash metadata is incomplete for request "
                    f"{req_id}: expected_at_least={expected_full_blocks}, "
                    f"actual={len(context_full_blk_hashes)}, "
                    f"num_tokens={context_len}, "
                    f"block_size={self.block_size}.")

            num_scheduled = int(scheduled_rows[row])
            num_output = len(req_state.output_token_ids)
            resident_value = resident_lens.get(req_id, INVALID_SLOT)
            resident_len = int(
                INVALID_SLOT if resident_value is None else resident_value)
            budget_value = sparse_budgets.get(req_id, 0)
            sparse_budget = int(0 if budget_value is None else budget_value)
            target_budget_value = target_budgets.get(req_id, 0)
            target_budget = int(
                0 if target_budget_value is None else target_budget_value)
            if target_budget <= 0:
                raise RuntimeError(
                    "DSA scheduler output is missing the immutable target "
                    f"resident budget for request {req_id!r}")
            if sparse_budget > 0 and sparse_budget != target_budget:
                raise RuntimeError(
                    "DSA sparse budget does not match the request's immutable "
                    "resident-budget tier: "
                    f"request={req_id!r}, sparse_budget={sparse_budget}, "
                    f"target_budget={target_budget}")
            if num_output == 0:
                fallback_stage = ReqStage.PREFILL
            elif sparse_budget > 0 and resident_len != INVALID_SLOT:
                fallback_stage = ReqStage.SPARSE_DECODE
            else:
                fallback_stage = ReqStage.DENSE_DECODE
            stage = ReqStage.coerce(req_stages.get(req_id, fallback_stage))
            sparse_enabled = (
                stage.is_sparse_decode
                and sparse_budget > 0
                and num_output > 0
                and resident_len != INVALID_SLOT)

            # Refresh is a projection step, not a resource-allocation boundary.
            # In particular, graph decode must not manufacture a missing
            # resident row immediately before the gate. Requests acquire that
            # fixed row during an earlier eager prefill/dense lifecycle step;
            # ENTER only changes the row's cache semantics. Eager metadata
            # assembly calls ensure_resident_resources() below when lifecycle
            # allocation is allowed.
            existing_pool_idx = resident_token_pool.get_index(req_id)
            pool_idx = int(
                INVALID_SLOT if existing_pool_idx is None
                else existing_pool_idx)

            is_sparse_decode = self._project_row_cache_semantics(
                row,
                num_prompt_tokens=num_prompt,
                num_output_tokens=num_output,
                num_computed_tokens=num_computed,
                resident_valid_seq_len=resident_len,
                context_full_block_hashes=context_full_blk_hashes,
                budget_block_ids=budget_block_ids,
                sparse_enabled=sparse_enabled,
                sparse_budget_tokens=sparse_budget,
            )
            final_context_len = num_computed + num_scheduled
            needs_dump = (
                num_output > 0
                and num_scheduled > 0
                and final_context_len > 0
                and final_context_len % self.block_size == 0)
            last_prefill_chunk = (
                num_output == 0
                and num_computed + num_scheduled >= num_prompt)

            self.budget_block_id_rows[row] = budget_block_ids
            self.context_full_block_hash_rows[row] = context_full_blk_hashes
            self.num_scheduled_tokens[row] = num_scheduled
            self.num_output_tokens[row] = num_output
            # This is the same non-speculative optimistic length used by the
            # native indexer metadata: tokens already computed plus tokens
            # scheduled in this forward.  Populate it while projecting the
            # scheduler row instead of reading ModelRunner's CPU buffer before
            # _prepare_inputs has refreshed that buffer for the current step.
            self.indexer_key_lens[row] = num_computed + num_scheduled
            self.stages[row] = int(stage)
            self.scheduler_resident_lens[row] = resident_len
            self.sparse_budget_tokens[row] = sparse_budget
            self.target_resident_budget_tokens[row] = target_budget
            self.resident_pool_indices[row] = pool_idx
            self.full_block_dump_mask[row] = needs_dump
            self.last_prefill_chunk_mask[row] = last_prefill_chunk
            self.decode_dump_src_hbm_block_ids[row] = 0
            self.decode_dump_dst_dram_block_ids[row] = (
                KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID)
            self.decode_dump_logical_block_indices[row] = INVALID_SLOT
            if needs_dump:
                if not context_full_blk_hashes or not budget_block_ids:
                    raise RuntimeError(
                        "DSA decode full-block boundary has no source/hash "
                        f"metadata: request={req_id!r}, row={row}")
                dump_idx = self.decode_dump_row_count
                self.decode_dump_row_indices[dump_idx] = row
                self.decode_dump_row_count += 1
                self.decode_dump_src_hbm_block_ids[row] = int(
                    budget_block_ids[-1])
                self.decode_dump_logical_block_indices[row] = (
                    len(context_full_blk_hashes) - 1)
            if is_sparse_decode:
                self.sparse_row_indices[self.sparse_row_count] = row
                self.sparse_row_count += 1
        # Drop references from rows that left the active prefix. Numeric tails
        # are reset as PAD so accidental out-of-prefix reads fail benignly.
        for row in range(row_count, old_row_count):
            self.budget_block_id_rows[row] = None
            self.context_full_block_hash_rows[row] = None
        if old_row_count > row_count:
            tail = slice(row_count, old_row_count)
            self.row_modes[tail] = int(DSADecodeRowMode.PAD)
            self.sparse_mask[tail] = False
            self.resident_pool_indices[tail] = INVALID_SLOT
            self.scheduler_resident_lens[tail] = INVALID_SLOT
            self.decode_dump_src_hbm_block_ids[tail] = 0
            self.decode_dump_dst_dram_block_ids[tail] = (
                KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID)
            self.decode_dump_logical_block_indices[tail] = INVALID_SLOT

        self.valid = True

    def ensure_resident_resources(
        self,
        *,
        resident_token_pool: DSAResidentTokenPool,
        dram_store: DSAHotKVStore,
    ) -> None:
        """Allocate/bind missing resident rows at an eager lifecycle boundary.

        Graph replay deliberately never calls this method: a graphable decode
        row must already have been established by an earlier eager prefill or
        dense-decode step. ENTER_SPARSE_DECODE reuses that row and only switches
        its projected operator ABI to SPARSE. Keeping allocation here prevents
        refresh() from hiding missing lifecycle setup immediately before replay.
        """
        if not self.valid:
            raise RuntimeError("DSA InputBatch state must be refreshed first")
        for row in range(self.row_count):
            req_id = self.request_ids[row]
            if req_id is None:
                raise RuntimeError(
                    f"DSA InputBatch row {row} has no request id")
            pool_idx = int(self.resident_pool_indices[row])
            if pool_idx == INVALID_SLOT:
                pool_idx = int(resident_token_pool.acquire(
                    req_id,
                    target_budget_tokens=int(
                        self.target_resident_budget_tokens[row]),
                ))
                self.resident_pool_indices[row] = pool_idx
            else:
                resident_token_pool.prepare_request(
                    req_id,
                    target_budget_tokens=int(
                        self.target_resident_budget_tokens[row]),
                )
            dram_store.bind_request_pool_index(req_id, pool_idx)

    def bind_query_layout(
        self,
        *,
        cumulative_query_lens: np.ndarray,
        resident_positions_cpu: np.ndarray,
    ) -> None:
        """Bind baseline query layout and finalize per-row decode positions.

        The resident array remains ModelRunner-owned storage. Sparse rows are
        rewritten in place to consecutive resident-space positions,
        本对象只保留 row-aligned query 起点/长度；尾部边界由 LIDU 的
        ``tail_info`` 合约输出，不再复制第二份逐行 position tensor。
        """
        if not self.valid:
            raise RuntimeError("DSA InputBatch state must be refreshed first")
        row_count = self.row_count
        cumulative = np.asarray(cumulative_query_lens).reshape(-1)
        if int(cumulative.size) < row_count:
            raise RuntimeError(
                "DSA query layout is shorter than the active InputBatch: "
                f"query_rows={int(cumulative.size)}, batch_rows={row_count}")
        if row_count > 0:
            self.query_start_locs[0] = 0
            if row_count > 1:
                self.query_start_locs[1:row_count] = cumulative[:row_count - 1]
            np.subtract(
                cumulative[:row_count],
                self.query_start_locs[:row_count],
                out=self.query_lens[:row_count],
                casting="unsafe",
            )
        resident_positions = np.asarray(resident_positions_cpu).reshape(-1)

        if row_count > 0 and self.sparse_row_count > 0:
            # MTP queries are request-major. For q tokens ending at resident
            # length R, cache positions are [R-q, ..., R-1]. Normal decode is
            # the q=1 special case.
            sparse_rows = self.sparse_row_indices[:self.sparse_row_count]
            for row in sparse_rows:
                row = int(row)
                start = int(self.query_start_locs[row])
                query_len = int(self.query_lens[row])
                resident_end = int(self.scheduler_resident_lens[row])
                resident_start = resident_end - query_len
                if query_len <= 0 or resident_start < 0:
                    raise RuntimeError(
                        "DSA sparse query cannot be mapped into the resident "
                        f"window: row={row}, query_len={query_len}, "
                        f"resident_end={resident_end}")
                resident_positions[start:start + query_len] = np.arange(
                    resident_start,
                    resident_end,
                    dtype=resident_positions.dtype,
                )
        self.query_layout_bound = True

    def matches_input_batch(
        self,
        input_batch: InputBatch,
        row_count: int,
    ) -> bool:
        """Cheap stale-state guard; row identity was fixed during refresh."""
        return (
            self.valid
            and self.input_batch is input_batch
            and self.row_count == int(row_count)
            and int(input_batch.num_reqs) == int(row_count)
        )
