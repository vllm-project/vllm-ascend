# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DSA row-mode decode 图模式的纯准入策略。

本文件是 DSA 图模式的**准入策略层**：只描述“当前 model forward 是否满足
DSA 图 replay 条件”，并给出可预期 eager 回退原因。它只读
``DSAInputBatchState`` 的语义投影，不做图 capture/replay 本身，不创建物理
buffer，也不修改请求状态。
开启 DSA 图模式后，DENSE、ENTER_SPARSE、SPARSE、三者混合以及其中任意行
触发新满块 dump 的 single-token decode 都复用原生 FULL 图族，并允许向上匹配
capture size 后补 PAD。prefill、multi-token/spec、资源尚未建立或形状不满足
条件的 step 仍走 eager 或按契约 fail-fast。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm_ascend.dsa_sparse.dsa_types import ReqStage

if TYPE_CHECKING:
    from vllm_ascend.dsa_sparse.dsa_input_batch_state import DSAInputBatchState


DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY = (
    "enable_dsa_row_mode_decode_graph")
DSA_ROW_MODE_DECODE_GRAPH_EXPECTED_EAGER_REASONS = frozenset({
    "empty_batch",
    "total_tokens_mismatch",
    "capture_size_miss",
    "non_single_token_decode",
    "non_row_mode_stage",
})


def is_dsa_row_mode_decode_graph_enabled(
    additional_config: object,
) -> bool:
    """Return whether DSA row-mode decode graph validation is enabled.

    This is intentionally a single switch: if enabled, eligible row-mode
    single-token decode must use the graph. Normal non-graphable stages still
    run eager.
    """
    if not isinstance(additional_config, dict):
        return False
    return additional_config.get(
        DSA_ROW_MODE_DECODE_GRAPH_CONFIG_KEY) is True


def is_dsa_row_mode_decode_graph_expected_eager(reason: str) -> bool:
    """Return whether a disabled graph gate should continue in eager.

    These reasons describe normal execution phases outside the current graph
    contract, not correctness problems in a graphable row-mode decode.
    """
    return reason in DSA_ROW_MODE_DECODE_GRAPH_EXPECTED_EAGER_REASONS


@dataclass(frozen=True)
class DSAGraphGateDecision:
    disabled: bool
    reason: str
    bad_req_id: str | None = None


def evaluate_dsa_row_mode_decode_graph(
    *,
    input_state: "DSAInputBatchState",
    total_tokens: int,
    max_capture_size: int,
    configured_budgets: tuple[int, ...],
    resident_graph_limit: int,
    block_size: int,
) -> DSAGraphGateDecision:
    """Gate DSA row-mode graph replay for single-token decode batches.

    DSA split-cache owns decode metadata even for pure dense rows, so dense,
    sparse, and mixed dense/sparse rows all use the same row-mode DSA graph
    once the batch is a row-mode single-token decode. ENTER_SPARSE_DECODE is
    already projected to the same SPARSE operator ABI and therefore reuses the
    same graph; multi-token/spec decode remains eager. A newly completed full
    block is represented by
    fixed src/dst row metadata and copied by the independent dump op after the
    current layer's attention. The block is still part of this forward's dense
    tail and becomes a LIDU candidate on the next forward, so the copy neither
    changes graph shape/address nor requires a separate graph family.
    """

    if not input_state.valid:
        return DSAGraphGateDecision(True, "missing_input_batch_state")
    row_count = int(input_state.row_count)
    if row_count <= 0:
        return DSAGraphGateDecision(True, "empty_batch")

    if int(total_tokens) != row_count:
        return DSAGraphGateDecision(True, "total_tokens_mismatch")

    # Follow the native FULL-decode dispatcher contract: an actual batch may
    # replay the smallest captured graph whose row count is not smaller than
    # the active row count. The replay serializer materializes the extra rows
    # as explicit PAD rows; only batches larger than every capture size stay
    # eager.
    if int(max_capture_size) < row_count:
        return DSAGraphGateDecision(True, "capture_size_miss")

    row_slice = slice(0, row_count)

    def first_bad_req(mask: np.ndarray) -> str | None:
        if not np.any(mask):
            return None
        return input_state.request_ids[int(np.argmax(mask))]

    bad_req_id = first_bad_req(
        input_state.num_scheduled_tokens[row_slice] != 1)
    if bad_req_id is not None:
        return DSAGraphGateDecision(
            True, "non_single_token_decode", bad_req_id=bad_req_id)

    stages = input_state.stages[row_slice]
    valid_stage_mask = (
        (stages == int(ReqStage.DENSE_DECODE))
        | (stages == int(ReqStage.ENTER_SPARSE_DECODE))
        | (stages == int(ReqStage.SPARSE_DECODE)))
    bad_req_id = first_bad_req(~valid_stage_mask)
    if bad_req_id is not None:
        return DSAGraphGateDecision(
            True, "non_row_mode_stage", bad_req_id=bad_req_id)

    # Both dense and sparse rows use the row-mode graph's resident metadata
    # tensors. Fixed-address replay therefore requires every row to have been
    # established by an earlier eager lifecycle step.
    bad_req_id = first_bad_req(
        input_state.resident_pool_indices[row_slice] < 0)
    if bad_req_id is not None:
        return DSAGraphGateDecision(
            True, "missing_resident_pool_row", bad_req_id=bad_req_id)

    sparse_row_count = int(input_state.sparse_row_count)
    sparse_rows = input_state.sparse_row_indices[:sparse_row_count]
    if sparse_row_count > 0:
        allowed_budgets = np.asarray(
            tuple(int(value) for value in configured_budgets),
            dtype=np.int32,
        )
        if (allowed_budgets.size == 0
                or np.any(allowed_budgets <= 0)):
            return DSAGraphGateDecision(True, "missing_configured_budgets")
        if int(resident_graph_limit) <= 0:
            return DSAGraphGateDecision(
                True, "invalid_resident_graph_limit")
        if int(block_size) <= 0:
            return DSAGraphGateDecision(True, "invalid_block_size")

        sparse_budgets = input_state.sparse_budget_tokens[sparse_rows]
        target_budgets = (
            input_state.target_resident_budget_tokens[sparse_rows])
        # 每请求 budget 在 admission 时按 prompt 长度冻结。图只要求动态
        # 值来自同一个配置档位集合，不能再把所有行硬性限制为最大档位；
        # 6K/10K/12K 行因此可以在同一张 captured graph 中混合 replay。
        budget_mismatch = (
            (sparse_budgets != target_budgets)
            | ~np.isin(sparse_budgets, allowed_budgets))
        if np.any(budget_mismatch):
            bad_row = int(sparse_rows[int(np.flatnonzero(
                budget_mismatch)[0])])
            return DSAGraphGateDecision(
                True,
                "sparse_budget_mismatch",
                bad_req_id=input_state.request_ids[bad_row],
            )

        resident_lens = input_state.scheduler_resident_lens[sparse_rows]
        invalid_resident = (
            (resident_lens <= 0)
            | (resident_lens > int(resident_graph_limit)))
        if np.any(invalid_resident):
            bad_row = int(sparse_rows[int(np.flatnonzero(
                invalid_resident)[0])])
            return DSAGraphGateDecision(
                True,
                "resident_len_out_of_graph_limit",
                bad_req_id=input_state.request_ids[bad_row],
            )

        invalid_context = input_state.context_lens[sparse_rows] <= 0
        if np.any(invalid_context):
            bad_row = int(sparse_rows[int(np.flatnonzero(
                invalid_context)[0])])
            return DSAGraphGateDecision(
                True,
                "invalid_context_len",
                bad_req_id=input_state.request_ids[bad_row],
            )

        below_budget = resident_lens < sparse_budgets
        if np.any(below_budget):
            bad_row = int(sparse_rows[int(np.flatnonzero(below_budget)[0])])
            return DSAGraphGateDecision(
                True,
                "resident_budget_below_required",
                bad_req_id=input_state.request_ids[bad_row],
            )

    return DSAGraphGateDecision(False, "allow_row_mode_decode")
