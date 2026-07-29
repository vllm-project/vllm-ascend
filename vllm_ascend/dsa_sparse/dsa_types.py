# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DSA 稀疏卸载跨模块共享的轻量类型与常量。

本文件只放轻量、稳定、可被 scheduler/worker/算子边界共同引用的类型。
INVALID_SLOT 是 scheduler/worker 之间传递 resident 状态时使用的哨兵值；
KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID 是通用满块复制算子的目的行空转值；
DSASparseRole 描述 manager 当前运行在 scheduler 侧还是 worker 侧；
ReqStage 描述请求生命周期里的 DSA 阶段；DSADecodeRowMode 描述传给
LIDU/KSC/SFA-Offload 的每行执行模式；ReqType 是跨 scheduler/worker 与
forward dump 表共用的请求标识类型。

不要在这里引入重型运行时依赖，避免基础类型模块反向耦合具体实现。
"""

import enum

INVALID_SLOT = -1
# 通用 kv_cache_full_block_dump 只把目的 block id 的 -1 解释为空转。
# DSA DRAM 逻辑块表自身仍保留 block 0 作为空映射；算子层不占用物理 block 0，
# 因而其他调用方仍可把目的 block 0 当作合法地址。
KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID = -1
DSA_LIDU_OUTPUT_CAPACITY = 16384
DSA_LIDU_TOKEN_CAPACITY = 1 << 18
DSA_LIDU_CACHE_ROW_ALIGNMENT = 256
# The A2/A3 custom kernel supports 6K/10K/12K rows.  A5 uses the
# v0.23 native-lightning-indexer compatibility implementation and is capped at
# 8K, so 8K is also a valid public configuration value.  Device-specific
# validation lives in dsa_config.py.
DSA_LIDU_SUPPORTED_RESIDENT_BUDGETS = (6144, 8192, 10240, 12288)
DSA_SFA_COMPUTE_TOPK = 2048
ReqType = str | int


def max_safe_mtp_drafts_before_block_boundary(
    num_computed_tokens: int,
    guaranteed_tokens: int,
    block_size: int,
) -> int:
    """Return the largest draft count that cannot create a rollback dump.

    A speculative token must not complete an MLA block: the post-attention
    dump is irreversible from the scheduler's point of view until the draft
    has been accepted.  The guaranteed main-model token may complete the next
    block (and therefore trigger its dump), but no draft may accompany that
    boundary step.

    Synchronous v0.23 decode normally has exactly one guaranteed token.  Treat
    a step that would cross a boundary using guaranteed tokens alone as an
    explicit contract violation instead of publishing a DRAM block whose
    lifecycle cannot be represented by the current scheduler metadata.
    """
    num_computed_tokens = int(num_computed_tokens)
    guaranteed_tokens = int(guaranteed_tokens)
    block_size = int(block_size)
    if num_computed_tokens < 0:
        raise ValueError("num_computed_tokens must be non-negative")
    if guaranteed_tokens <= 0:
        raise ValueError("guaranteed_tokens must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    remaining_to_boundary = (
        block_size - num_computed_tokens % block_size
    )
    if guaranteed_tokens > remaining_to_boundary:
        raise RuntimeError(
            "DSA MTP cannot cross an MLA block boundary with multiple "
            "guaranteed tokens in one scheduler step: "
            f"computed={num_computed_tokens}, "
            f"guaranteed={guaranteed_tokens}, block_size={block_size}"
        )
    if guaranteed_tokens == remaining_to_boundary:
        return 0
    return remaining_to_boundary - guaranteed_tokens - 1


class DSASparseRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1


class DSADecodeRowMode(enum.IntEnum):
    """Per-row execution mode for DSA decode operator boundaries.

    ReqStage is scheduler-owned request state.  This enum is the tensorized row
    contract consumed by LIDU/KSC/SFA-Offload decode operators:
    - PAD rows are graph padding and must not touch cache state.
    - DENSE rows perform native top-2048 selection without DRAM IO.
    - SPARSE rows materialize only LIDU misses into resident slots and use
      resident logical slots for SFA-Offload.
    """

    PAD = 0
    DENSE = 1
    SPARSE = 2


class ReqStage(enum.IntEnum):
    """DSA sparse-cache stage for one request in one scheduler step.

    This is the scheduler-owned state machine used by both allocation and the
    worker runtime. It deliberately separates "what cache layout this request
    uses now" from layer-local actions such as dumping a newly completed block.

    Transitions:
    - PREFILL -> DENSE_DECODE after prompt/chunk prefill is done but the full
      context is still below the DSA sparse threshold, or sparse decode is not
      supported for the current step.
    - DENSE_DECODE/PREFILL -> ENTER_SPARSE_DECODE on the first decode step
      whose context can use DSA sparse MLA/SFA. This includes both the classic
      long-prompt first decode and the short-prompt long-decode case where the
      request crosses the threshold later.
    - ENTER_SPARSE_DECODE -> SPARSE_DECODE on the next sparse decode step.

    Full-block dump is an action that may happen in PREFILL or any decode stage
    when a block becomes complete. It is not a separate stage.
    """

    PREFILL = 0
    DENSE_DECODE = 1
    ENTER_SPARSE_DECODE = 2
    SPARSE_DECODE = 3

    @classmethod
    def coerce(cls, value: object) -> "ReqStage":
        if isinstance(value, cls):
            return value
        try:
            return cls(int(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return cls.PREFILL

    @property
    def is_decode(self) -> bool:
        return self != ReqStage.PREFILL

    @property
    def is_sparse_decode(self) -> bool:
        return self in (ReqStage.ENTER_SPARSE_DECODE, ReqStage.SPARSE_DECODE)

    @property
    def is_enter_sparse_decode(self) -> bool:
        return self == ReqStage.ENTER_SPARSE_DECODE
