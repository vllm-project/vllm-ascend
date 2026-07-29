"""DSA 请求满块 hash 的 scheduler 到 worker 增量传输协议。

新请求通过 ``NewRequestData`` 发送完整快照；cached request 只发送
``block_hash_starts`` 及其后的 append-only 后缀，worker 再更新
``CachedRequestState.context_full_blk_hashes``。若发送游标失效或 producer
账本缩短，则自动退回从下标零开始的完整覆盖，避免 preempt/reset 后沿用旧账本。

本模块只定义跨进程账本协议，不分配 block，也不解释本轮 candidate、budget、
tail 或 row mode；这些 forward 语义统一由 ``DSAInputBatchState`` 投影。
"""

from __future__ import annotations


def build_context_full_block_hash_delta(
    block_hashes: list,
    previous_count: int,
) -> tuple[int, list]:
    """Return the append-only suffix not yet sent to a worker.

    vLLM's ``Request.update_block_hashes`` appends hashes only after a new
    logical block becomes full. Sending the complete request-lifetime list on
    every decode token is therefore unnecessary. If the producer ledger ever
    shrinks or its cursor is invalid, return a full snapshot from offset zero
    so streaming/reset paths remain self-healing.
    """
    current_count = len(block_hashes)
    start = int(previous_count)
    if start < 0 or start > current_count:
        start = 0
    return start, list(block_hashes[start:])


def apply_context_full_block_hash_delta(
    block_hashes: list,
    start: int,
    delta: list,
) -> None:
    """Apply a scheduler hash suffix to the worker's lifetime ledger.

    ``start`` may point before the current tail when a scheduler resends a
    snapshot after reset. Truncating first makes replay idempotent. A start
    beyond the local tail means the worker missed required hashes and must be
    rejected rather than silently producing an incorrect DRAM block mapping.
    """
    start = int(start)
    if start < 0 or start > len(block_hashes):
        raise RuntimeError(
            "DSA full-block hash delta has a missing worker prefix: "
            f"start={start}, worker_count={len(block_hashes)}, "
            f"delta_count={len(delta)}")
    if start < len(block_hashes):
        del block_hashes[start:]
    block_hashes.extend(delta)
