# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External, unquantized FlashMLA implementation and metadata builder.

Call FlashMLAImpl.load after selecting the worker device. Importing this module
does not load cann_ops_transformer. FlashMLAImpl wraps the operator contract, while
FlashMLAMetadataBuilder owns per-batch inputs, stable buffers and per-build refreshes.
Callers own stream ordering and persistent KV writes. Tensor-content checks belong
in preflight tests: the decode hot path must not copy lengths or tables to the CPU.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.attention.utils import MLA_FLASH_SUPPORTED_Q_HEADS, get_flashmla_ops
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import BatchReqState

FLASHMLA_QK_DIM = 576
FLASHMLA_V_DIM = 512
FLASHMLA_BLOCK_SIZE = 128
FLASHMLA_SUPPORTED_BLOCK_SIZES = tuple(range(FLASHMLA_BLOCK_SIZE, 1024 + 1, FLASHMLA_BLOCK_SIZE))
FLASHMLA_MASK_SIZE = 2048
FLASHMLA_MAX_BATCH_SIZE = 65535
PA_NZ_TILE_WIDTH = 16
PA_NZ_TILE_COUNT = FLASHMLA_QK_DIM // PA_NZ_TILE_WIDTH
FLASHMLA_STREAM_ALLOCATION_ATTEMPTS = 64


def _pa_nz_mask_mode(q_len: int, configured_mask_mode: int) -> int:
    """Decode always uses no-mask; multi-token history keeps the adapter mode."""
    return 0 if q_len == 1 else configured_mask_mode


def _pa_nz_tile_indices(
    slots: torch.Tensor,
    block_size: int,
    tile_offsets: torch.Tensor,
    tile_count: int,
) -> torch.Tensor:
    """Build flattened PA_NZ tile rows from already-sanitized token slots.

    The geometry tensors are owned by ``FlashMLADecode`` and reused for every
    MLA layer.  Only the request-dependent block and in-block offsets are
    calculated during metadata refresh.
    """
    block_ids = torch.div(slots, block_size, rounding_mode="floor")
    offsets = torch.remainder(slots, block_size)
    return (block_ids[:, None] * (tile_count * block_size) + tile_offsets[None, :] + offsets[:, None]).reshape(-1)


def prepare_flashmla_batch_req_state(
    batch_state: "BatchReqState", uniform_token_count: int | None
) -> tuple["BatchReqState", int | None]:
    """Order real decode before FIA prefill before inputs and graph dispatch.

    Reorder the release request-state fields together so positions, KV slots
    and sampling indices inherit the same order. A one-token prompt remains
    FIA prefill and cannot reuse a graph captured for FlashMLA decode.
    """
    order = np.argsort(batch_state.is_prefilling_np, kind="stable")
    batch_state = batch_state._replace(
        req_ids=[batch_state.req_ids[index] for index in order],
        num_scheduled_tokens=batch_state.num_scheduled_tokens[order],
        idx_mapping_np=batch_state.idx_mapping_np[order],
        prefill_len_np=batch_state.prefill_len_np[order],
        num_computed_prefill_tokens_np=batch_state.num_computed_prefill_tokens_np[order],
        is_prefilling_np=batch_state.is_prefilling_np[order],
    )
    if batch_state.has_prefill:
        uniform_token_count = None
    return batch_state, uniform_token_count


def split_flashmla_requests(common) -> tuple[int, int, int, int]:
    """Split real generation stages after the runner's stable phase ordering.

    Read only CPU scheduler metadata. A one-token prompt suffix is still FIA
    prefill; query length alone must never opt a request into FlashMLA.
    """
    flags = common.is_prefilling
    offsets = common.query_start_loc_cpu[: common.num_reqs + 1]
    if flags is None or flags.device.type != "cpu" or offsets.device.type != "cpu":
        raise RuntimeError("FlashMLA requires CPU is_prefilling and query boundaries from the runner")
    boundaries = offsets.clamp_max(common.num_actual_tokens).tolist()
    stages = flags[: common.num_reqs].tolist()
    if len(stages) < common.num_reqs:
        if any(boundaries[i + 1] != boundaries[i] for i in range(len(stages), common.num_reqs)):
            raise RuntimeError("FlashMLA is_prefilling does not cover the active request batch")
        stages.extend([False] * (common.num_reqs - len(stages)))
    # FIA may append a dummy request to own graph padding. Exclude trailing
    # empty rows so they cannot change our captured buffer capacity or route.
    request_count = common.num_reqs
    while request_count and boundaries[request_count] == boundaries[request_count - 1]:
        request_count -= 1
    first_prefill = request_count
    for index, is_prefill in enumerate(stages[:request_count]):
        if boundaries[index + 1] == boundaries[index]:
            continue
        if is_prefill:
            first_prefill = min(first_prefill, index)
        elif first_prefill != request_count:
            raise RuntimeError("FlashMLA requires real decode requests before prefill requests; check runner ordering")
    decode_tokens = boundaries[first_prefill]
    return first_prefill, request_count - first_prefill, decode_tokens, common.num_actual_tokens - decode_tokens


@dataclass(frozen=True)
class FlashMLAConfig:
    num_heads: int
    softmax_scale: float
    mask_mode: int = 3
    win_left: int = -1
    win_right: int = -1
    layout_kv: str = "PA_BBND"
    return_softmax_lse: bool = False
    # The metadata operator derives the per-request local causal boundary, so
    # the main operator keeps one signature and reads those fields from it.
    enable_dcp: bool = False
    cp_world: int = 1
    cp_rank: int = 0
    cp_kv_cache_interleave_size: int = FLASHMLA_BLOCK_SIZE
    cache_block_size: int = FLASHMLA_BLOCK_SIZE

    def __post_init__(self) -> None:
        if self.layout_kv not in ("PA_BBND", "PA_NZ"):
            raise ValueError("FlashMLA layout_kv must be PA_BBND or PA_NZ")
        # The operator's paged cache is laid out for a fixed set of Q head-group
        # widths; the selection gate and this check must agree on that set.
        if self.layout_kv == "PA_BBND" and self.num_heads not in MLA_FLASH_SUPPORTED_Q_HEADS:
            raise ValueError(f"FlashMLA requires one of {MLA_FLASH_SUPPORTED_Q_HEADS} Q heads, got {self.num_heads}")
        if self.layout_kv == "PA_NZ" and self.num_heads <= 0:
            raise ValueError("PA_NZ FlashMLA requires a positive local Q head count")
        if self.mask_mode not in (0, 3, 4):
            raise ValueError("FlashMLA mask_mode must be 0, 3 or 4")
        if self.layout_kv == "PA_NZ" and self.mask_mode not in (0, 3):
            raise ValueError("PA_NZ FlashMLA mask_mode must be 0 or 3")
        if self.cache_block_size not in FLASHMLA_SUPPORTED_BLOCK_SIZES:
            raise ValueError(f"FlashMLA cache block size must be in {FLASHMLA_SUPPORTED_BLOCK_SIZES}")
        if self.mask_mode != 4 and (self.win_left != -1 or self.win_right != -1):
            # Only the bounded sliding window carries non-default edges, and both
            # operators must receive identical values.
            raise ValueError("FlashMLA win_left/win_right must stay -1 outside mask_mode=4")
        if self.mask_mode == 4 and (self.win_left < -1 or self.win_right < -1):
            raise ValueError("FlashMLA win_left/win_right must be -1 or non-negative")
        if self.enable_dcp:
            # DCP owns the mask through the metadata, so no external mask or
            # window may be passed. The kernel validates that the metadata's
            # interleave equals the physical reader page, including 512/896.
            if self.mask_mode != 0 or self.win_left != -1 or self.win_right != -1:
                raise ValueError("FlashMLA DCP requires mask_mode=0 and unbounded windows")
            if self.cp_world <= 0:
                raise ValueError("FlashMLA DCP requires cp_world >= 1")
            if not 0 <= self.cp_rank < self.cp_world:
                raise ValueError("FlashMLA cp_rank must be a 0-based value below cp_world")
            if self.cp_kv_cache_interleave_size != self.cache_block_size:
                raise ValueError("FlashMLA DCP interleave must equal the physical cache block size")
        elif self.cp_world != 1 or self.cp_rank != 0 or self.cp_kv_cache_interleave_size != FLASHMLA_BLOCK_SIZE:
            raise ValueError("FlashMLA disabled DCP requires world=1, rank=0 and interleave=128")


def _check_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if tensor.shape != shape or tensor.dtype != dtype:
        raise ValueError(f"FlashMLA {name} requires shape={shape}, dtype={dtype}; got {tensor.shape}, {tensor.dtype}")


def _check_lengths(
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_q: torch.Tensor | None,
    casusal_seqlens_kv_global: torch.Tensor | None = None,
) -> int:
    if cache_seqlens.ndim != 1 or not 0 < cache_seqlens.numel() <= FLASHMLA_MAX_BATCH_SIZE:
        raise ValueError("FlashMLA cache_seqlens must be a nonempty vector with fewer than 65536 requests")
    batch_size = cache_seqlens.numel()
    _check_tensor("cache_seqlens", cache_seqlens, (batch_size,), torch.int32)
    _check_tensor("cu_seqlens_q", cu_seqlens_q, (batch_size + 1,), torch.int32)
    if cu_seqlens_q.device != cache_seqlens.device:
        raise ValueError("FlashMLA query and KV lengths must be on the same device")
    if seqused_q is not None:
        _check_tensor("seqused_q", seqused_q, (batch_size,), torch.int32)
        if seqused_q.device != cache_seqlens.device:
            raise ValueError("FlashMLA seqused_q must be on the same device as cache_seqlens")
    if casusal_seqlens_kv_global is not None:
        _check_tensor("casusal_seqlens_kv_global", casusal_seqlens_kv_global, (batch_size,), torch.int32)
        if casusal_seqlens_kv_global.device != cache_seqlens.device:
            raise ValueError("FlashMLA global KV lengths must be on the same device as cache_seqlens")
    return batch_size


@dataclass(frozen=True)
class FlashMLAImpl:
    """Keep metadata and attention attributes identical, preserving input storage.

    The installed package remains responsible for its hardware/stride support
    and metadata capacity. Accepting a view here is not evidence that the NPU
    binary supports it. No exception from an operator call is silently retried
    with a contiguous cache or a different attention backend.
    """

    config: FlashMLAConfig
    attention_op: Callable
    metadata_op: Callable

    @classmethod
    def load(cls, config: FlashMLAConfig) -> "FlashMLAImpl":
        available_ops = get_flashmla_ops()
        if available_ops is None:
            raise RuntimeError(
                "FlashMLA requires cann_ops_transformer.ops.flash_mla "
                "and cann_ops_transformer.ops.flash_mla_metadata. "
                "Install a DCP-capable package matching the worker's "
                "CANN and torch_npu versions."
            )
        attention_op, metadata_op = available_ops
        return cls(config, attention_op, metadata_op)

    def build_metadata(
        self,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_q: torch.Tensor | None = None,
        casusal_seqlens_kv_global: torch.Tensor | None = None,
        *,
        q_len: int = 1,
    ) -> torch.Tensor:
        """Generate metadata for these device lengths, or query capacity on Meta.

        The package Meta implementation may query NPU core counts; select the
        worker device first. Do not replace -1 with CPU maximum-length bounds.
        The metadata and the main operator must be generated from the same
        lengths, head counts, mask mode and query layout.
        """
        del q_len  # Native metadata derives query lengths from device inputs.
        if self.config.enable_dcp and casusal_seqlens_kv_global is None:
            raise ValueError("FlashMLA DCP metadata requires casusal_seqlens_kv_global")
        if not self.config.enable_dcp and casusal_seqlens_kv_global is not None:
            raise ValueError("FlashMLA non-DCP metadata forbids casusal_seqlens_kv_global")
        _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q, casusal_seqlens_kv_global)
        window_args = (
            {"win_left": self.config.win_left, "win_right": self.config.win_right} if self.config.mask_mode == 4 else {}
        )
        return self.metadata_op(
            cache_seqlens,
            num_heads_q=self.config.num_heads,
            num_heads_kv=1,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            head_dim_qk=FLASHMLA_QK_DIM,
            head_dim_v=FLASHMLA_V_DIM,
            mask_mode=self.config.mask_mode,
            layout_q="TND",
            enable_dcp=self.config.enable_dcp,
            cp_world=self.config.cp_world,
            cp_rank=self.config.cp_rank,
            cp_kv_cache_interleave_size=self.config.cp_kv_cache_interleave_size,
            casusal_seqlens_kv_global=casusal_seqlens_kv_global,
            **window_args,
        )

    def attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        metadata: torch.Tensor,
        seqused_q: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        q_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapt the custom operator's TND output to the caller's NTD contract.

        Metadata must have been generated from these same lengths and this
        config. Regenerate it when lengths change, including graph replay.
        The operator returns NT FP32 LSE even when layout_out is TND.
        """
        del q_len  # Native attention consumes the query layout from metadata.
        self._check_inputs(q, k_cache, block_table, cache_seqlens, cu_seqlens_q, metadata, seqused_q, attn_mask)
        window_args = (
            {"win_left": self.config.win_left, "win_right": self.config.win_right} if self.config.mask_mode == 4 else {}
        )
        output, lse = self.attention_op(
            q,
            k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            attn_mask=attn_mask,
            metadata=metadata,
            head_dim_v=FLASHMLA_V_DIM,
            softmax_scale=self.config.softmax_scale,
            mask_mode=self.config.mask_mode,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            layout_q="TND",
            layout_kv=self.config.layout_kv,
            layout_out="TND",
            return_softmax_lse=self.config.return_softmax_lse,
            **window_args,
        )
        _check_tensor("output", output, (q.shape[0], self.config.num_heads, FLASHMLA_V_DIM), q.dtype)
        if self.config.return_softmax_lse:
            # Output and LSE have different layouts in the reference API.
            _check_tensor("softmax_lse", lse, (self.config.num_heads, q.shape[0]), torch.float32)
        return output.transpose(0, 1), lse

    def _check_inputs(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        metadata: torch.Tensor,
        seqused_q: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> None:
        batch_size = _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        if q.ndim != 3 or q.shape[0] == 0 or q.shape[1:] != (self.config.num_heads, FLASHMLA_QK_DIM):
            raise ValueError("FlashMLA q must be nonempty TND with configured local heads and head dimension 576")
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FlashMLA q must be BF16 or FP16")
        cache_tail = (self.config.cache_block_size, 1, FLASHMLA_QK_DIM)
        if k_cache.ndim != len(cache_tail) + 1 or k_cache.shape[1:] != cache_tail or k_cache.shape[0] == 0:
            raise ValueError(f"FlashMLA {self.config.layout_kv} cache must have shape (pages, {cache_tail})")
        if k_cache.dtype != q.dtype:
            raise ValueError("FlashMLA q and k_cache must have the same dtype")
        if block_table.ndim != 2 or block_table.shape[0] != batch_size or block_table.shape[1] == 0:
            raise ValueError("FlashMLA block_table must have shape (batch_size, positive_table_capacity)")
        if block_table.dtype != torch.int32:
            raise ValueError("FlashMLA block_table must be int32")
        if metadata.ndim != 1 or metadata.numel() == 0 or metadata.dtype != torch.int32:
            raise ValueError("FlashMLA metadata must be a nonempty int32 vector from the metadata operator")
        if self.config.mask_mode == 0 and attn_mask is not None:
            raise ValueError("FlashMLA mask_mode=0 requires attn_mask=None")
        if self.config.mask_mode in (3, 4):
            if attn_mask is None:
                raise ValueError("FlashMLA mask_mode=3/4 requires the upper-triangular int8 mask")
            _check_tensor("attn_mask", attn_mask, (FLASHMLA_MASK_SIZE, FLASHMLA_MASK_SIZE), torch.int8)
        for tensor in (k_cache, block_table, cache_seqlens, metadata, attn_mask):
            if tensor is not None and tensor.device != q.device:
                raise ValueError("FlashMLA inputs must be on the same device")


@dataclass(frozen=True)
class PaNzFlashMLAImpl:
    """BF16 adapter for the v0.27 PA_NZ flash_mla operator.

    The operator module is imported in ``load`` only, so the default PA_BBND
    path never touches the optional 0.27 CANN/cannbot-dsl packages.
    """

    config: FlashMLAConfig
    ops: Any

    @classmethod
    def load(cls, config: FlashMLAConfig) -> "PaNzFlashMLAImpl":
        if config.layout_kv != "PA_NZ":
            raise ValueError("PaNzFlashMLAImpl requires layout_kv='PA_NZ'")
        from vllm_ascend.ops import flash_mla_pa_nz as ops  # noqa: PLC0415

        # Fail fast here if the optional operators are missing.
        ops._get_pa_nz_flash_mla_ops()
        ops._get_bf16_mla_prolog_op()
        return cls(config, ops)

    def build_metadata(
        self,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_q: torch.Tensor | None = None,
        q_len: int = 1,
    ) -> torch.Tensor:
        _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        q_len = max(1, q_len)
        return self.ops.flash_mla_metadata(
            cache_seqlens,
            num_heads_q=self.config.num_heads,
            cu_seqlens_q=cu_seqlens_q,
            q_len=q_len,
            head_dim_qk=FLASHMLA_QK_DIM,
            head_dim_v=FLASHMLA_V_DIM,
            mask_mode=_pa_nz_mask_mode(q_len, self.config.mask_mode),
            seqused_q=seqused_q,
        )

    def attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        metadata: torch.Tensor,
        seqused_q: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        q_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return NTD latent output and NT FP32 LSE for a TND query."""
        self.validate_cache(k_cache)
        if q.ndim != 3 or q.shape[1:] != (self.config.num_heads, FLASHMLA_QK_DIM):
            raise ValueError("PA_NZ FlashMLA q must be TND with configured local heads and width 576")
        if q.dtype != torch.bfloat16:
            raise ValueError("PA_NZ FlashMLA requires BF16 q")
        if q_len is None:
            # Never read cu_seqlens_q back to the host here: this method runs
            # inside ACL graph capture, where a scalar .item() would
            # synchronize a copy stream and abort capture. Every caller that
            # can see multi-token queries passes flash.q_len explicitly.
            q_len = 1
        q_len = max(1, q_len)
        output, lse = self.ops.flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            q_len,
            self.config.softmax_scale,
            FLASHMLA_V_DIM,
            attn_mask=attn_mask if q_len > 1 else None,
            metadata=metadata,
            seqused_q=seqused_q,
            return_softmax_lse=self.config.return_softmax_lse,
            mask_mode=_pa_nz_mask_mode(q_len, self.config.mask_mode),
        )
        if self.config.return_softmax_lse:
            if lse is None:
                raise RuntimeError("PA_NZ FlashMLA requested LSE but the operator returned None.")
            if lse.ndim == 2 and lse.shape == (q.shape[0], q.shape[1]):
                lse = lse.transpose(0, 1).contiguous()
            if lse.shape != (q.shape[1], q.shape[0]) or lse.dtype != torch.float32:
                raise ValueError(
                    "PA_NZ FlashMLA softmax_lse must have shape "
                    f"({q.shape[1]}, {q.shape[0]}) and dtype float32; got {lse.shape}, {lse.dtype}"
                )
        return output, lse

    def validate_cache(self, kv_cache: torch.Tensor) -> None:
        if (
            kv_cache.ndim != 4
            or kv_cache.shape[1] != FLASHMLA_BLOCK_SIZE
            or kv_cache.shape[2] != 1
            or kv_cache.shape[-1] != FLASHMLA_QK_DIM
        ):
            raise ValueError(f"PA_NZ FlashMLA cache must be [blocks, {FLASHMLA_BLOCK_SIZE}, 1, {FLASHMLA_QK_DIM}]")
        if kv_cache.dtype != torch.bfloat16:
            raise ValueError("PA_NZ FlashMLA cache must be BF16")
        if not kv_cache.is_contiguous():
            raise ValueError("PA_NZ FlashMLA cache must be contiguous")


def create_flash_mla_adapter(config: FlashMLAConfig) -> FlashMLAImpl | PaNzFlashMLAImpl:
    if config.layout_kv == "PA_NZ":
        return PaNzFlashMLAImpl.load(config)
    return FlashMLAImpl.load(config)


@dataclass
class FlashMLACurrent:
    """Replicated current chunk, independent of persistent KV and prefix pages."""

    adapter: FlashMLAImpl | PaNzFlashMLAImpl
    query: torch.Tensor
    schedule: torch.Tensor
    cache: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    attn_mask: torch.Tensor


@dataclass
class FlashMLADecode:
    adapter: FlashMLAImpl | PaNzFlashMLAImpl
    query: torch.Tensor
    schedule: torch.Tensor
    cu: torch.Tensor
    used_q: torch.Tensor
    cache_lens: torch.Tensor
    global_cache_lens: torch.Tensor | None
    block_table: torch.Tensor
    slots: torch.Tensor
    positions: torch.Tensor
    token_live: torch.Tensor
    live_boundaries: torch.Tensor
    attn_mask: torch.Tensor | None
    cos: torch.Tensor | None
    sin: torch.Tensor | None
    current: FlashMLACurrent | None = None
    layout: str = "PA_BBND"
    q_len: int = 1
    block_table_prepared: bool = False
    prolog_slots: torch.Tensor | None = None
    block_columns: torch.Tensor | None = None
    metadata_builder: Any = None
    metadata_started: bool = False
    metadata_async: bool = False
    metadata_start: Any = None
    metadata_ready: Any = None
    metadata_capture_started: bool = False
    metadata_capture_events: list[tuple[Any, Any]] = field(default_factory=list)
    # PA_NZ DCP copy geometry is batch metadata.  The cache contents are
    # layer-local, but these slot/tile addresses are identical for every MLA
    # layer and therefore must not be rebuilt in the layer loop.
    pa_nz_tile_offsets: torch.Tensor | None = None
    pa_nz_current_slots: torch.Tensor | None = None
    pa_nz_history_slots: torch.Tensor | None = None
    pa_nz_copy_src_indices: torch.Tensor | None = None
    pa_nz_copy_dst_indices: torch.Tensor | None = None


class FlashMLAMetadataBuilder:
    def __init__(
        self,
        impl,
        device: torch.device,
        max_num_reqs: int,
        attn_mask: torch.Tensor,
        *,
        cache_block_size: int = FLASHMLA_BLOCK_SIZE,
        enable_async: bool = False,
    ):
        self.dcp_size = getattr(impl, "dcp_size", 1)
        self.dcp_rank = getattr(impl, "dcp_rank", 0)
        # DCP builds both history and current schedules on the same side
        # stream. The attention path waits once before entering the split
        # history/current execution, so both schedules are visible there.
        self.enable_async = bool(enable_async)
        self.metadata_stream = None
        if self.enable_async:
            # Stream() can draw from a runtime pool: distinct Python objects
            # alone do not establish isolation from the Kimi router stream.
            reserved_streams = (
                torch.npu.current_stream(),
                getattr(impl.vllm_config, "_kimi_router_stream", None),
                getattr(impl.vllm_config, "_kimi_shared_comm_stream", None),
                getattr(impl.vllm_config, "_kimi_shared_compute_stream", None),
            )
            for _ in range(FLASHMLA_STREAM_ALLOCATION_ATTEMPTS):
                stream = torch.npu.Stream()
                if all(reserved is None or stream != reserved for reserved in reserved_streams):
                    self.metadata_stream = stream
                    break
            if self.metadata_stream is None:
                raise RuntimeError("Unable to allocate a FlashMLA stream distinct from the Kimi MoE streams")
        # Materialize once: an integer rank would trigger a blocking H2D copy
        # inside get_dcp_local_seq_lens on every metadata refresh.
        self.dcp_rank_tensor = (
            torch.tensor(self.dcp_rank, dtype=torch.int32, device=device) if self.dcp_size > 1 else None
        )
        self.use_spec_decode = impl.vllm_config.speculative_config is not None
        self.cp_interleave = (
            getattr(impl.vllm_config.parallel_config, "cp_kv_cache_interleave_size", FLASHMLA_BLOCK_SIZE)
            if self.dcp_size > 1
            else FLASHMLA_BLOCK_SIZE
        )
        noncausal_draft = getattr(impl, "is_draft_cache", False) and getattr(
            impl, "non_causal_multi_token_decode", False
        )
        dcp = self.dcp_size > 1 and not noncausal_draft
        layout = "PA_NZ" if getattr(impl, "flash_mla_layout", "pa_bbnd") == "pa_nz" else "PA_BBND"
        if layout == "PA_NZ":
            # The legacy PA_NZ reader splits speculative current KV from
            # sharded history. Native PA_BBND keeps its device DCP boundary.
            adapter = create_flash_mla_adapter(
                FlashMLAConfig(
                    impl.num_heads,
                    impl.scale,
                    layout_kv=layout,
                    return_softmax_lse=self.dcp_size > 1,
                )
            )
            self.current_adapter = adapter
            history_adapter = (
                replace(
                    adapter,
                    config=replace(adapter.config, num_heads=impl.num_heads * self.dcp_size, mask_mode=0),
                )
                if self.dcp_size > 1
                else adapter
            )
            self.adapters = {True: history_adapter, False: history_adapter}
        else:
            self._init_native_adapters(impl, dcp, noncausal_draft, cache_block_size)
        self.device = device
        self.dtype = impl.dtype
        self.use_rope = impl.use_mla_rope
        self.max_num_reqs = max_num_reqs
        self.buffers: dict[tuple, FlashMLADecode] = {}
        self.attn_mask = attn_mask

    def _init_native_adapters(self, impl, dcp, noncausal_draft, cache_block_size):
        adapter = FlashMLAImpl.load(
            FlashMLAConfig(
                # Q is gathered across the DCP group before attending each KV
                # shard. The output exchange restores this rank's local heads.
                impl.num_heads * self.dcp_size,
                impl.scale,
                mask_mode=0 if dcp or noncausal_draft else 3,
                # DCP produces only the local attention and local statistics; the
                # caller owns the inter-rank reduction.
                return_softmax_lse=self.dcp_size > 1,
                enable_dcp=dcp,
                cp_world=self.dcp_size if dcp else 1,
                cp_rank=self.dcp_rank if dcp else 0,
                cp_kv_cache_interleave_size=self.cp_interleave if dcp else FLASHMLA_BLOCK_SIZE,
                cache_block_size=cache_block_size,
            )
        )
        # Noncausal queries attend the complete local shard without the
        # operator's DCP causal boundary. Framework DCP still gathers heads
        # and merges local outputs/LSE with the real group size.
        noncausal_adapter = replace(
            adapter,
            config=replace(
                adapter.config,
                mask_mode=0,
                enable_dcp=False,
                cp_world=1,
                cp_rank=0,
                cp_kv_cache_interleave_size=FLASHMLA_BLOCK_SIZE,
            ),
        )
        self.adapters = {False: noncausal_adapter}
        if not noncausal_draft:
            self.adapters[True] = adapter

    def _allocate(self, tokens: int, rows: int, columns: int, causal: bool, q_len: int = 1) -> FlashMLADecode:
        ints = {"dtype": torch.int32, "device": self.device}
        adapter = self.adapters[causal]
        cu = torch.zeros(rows + 1, **ints)
        used = torch.zeros(rows, **ints)
        lengths = torch.zeros(rows, **ints)
        global_lengths = torch.zeros(rows, **ints) if adapter.config.enable_dcp else None
        if adapter.config.layout_kv == "PA_NZ":
            # Prefer the package Meta implementation so full-graph capture
            # never records a device metadata op. Older packages may not
            # implement Meta for this operator; fall back to one device call
            # before capture (the schedule is retained at a stable address).
            try:
                schedule_meta = adapter.build_metadata(
                    torch.empty_like(lengths, device="meta"),
                    torch.empty_like(cu, device="meta"),
                    torch.empty_like(used, device="meta"),
                    q_len=q_len,
                )
                schedule = torch.empty_like(schedule_meta, device=self.device)
            except (AttributeError, NotImplementedError, RuntimeError):
                cu.copy_(torch.arange(rows + 1, dtype=torch.int32, device=self.device))
                used.fill_(1)
                lengths.fill_(1)
                schedule = adapter.build_metadata(lengths, cu, used, q_len=q_len)
        else:
            schedule_meta = adapter.build_metadata(
                torch.empty_like(lengths, device="meta"),
                torch.empty_like(cu, device="meta"),
                torch.empty_like(used, device="meta"),
                None if global_lengths is None else torch.empty_like(global_lengths, device="meta"),
            )
            schedule = torch.empty_like(schedule_meta, device=self.device)
        rope_shape = (tokens, 1, 1, FLASHMLA_QK_DIM - FLASHMLA_V_DIM)
        flash = FlashMLADecode(
            adapter=adapter,
            query=torch.empty(
                (tokens, adapter.config.num_heads, FLASHMLA_QK_DIM), dtype=self.dtype, device=self.device
            ),
            schedule=schedule,
            cu=cu,
            used_q=used,
            cache_lens=lengths,
            global_cache_lens=global_lengths,
            block_table=torch.zeros((rows, columns), **ints),
            slots=torch.full((tokens,), -1, dtype=torch.int64, device=self.device),
            positions=torch.zeros(tokens, dtype=torch.int64, device=self.device),
            token_live=torch.zeros(tokens, dtype=torch.bool, device=self.device),
            live_boundaries=torch.zeros(tokens + 1, **ints),
            attn_mask=(
                self.attn_mask
                if (adapter.config.layout_kv == "PA_BBND" and adapter.config.mask_mode in (3, 4))
                or (adapter.config.layout_kv == "PA_NZ" and q_len > 1)
                else None
            ),
            cos=torch.empty(rope_shape, dtype=self.dtype, device=self.device) if self.use_rope else None,
            sin=torch.empty(rope_shape, dtype=self.dtype, device=self.device) if self.use_rope else None,
            layout=adapter.config.layout_kv,
            q_len=q_len,
            prolog_slots=torch.full((tokens,), -1, dtype=torch.int64, device=self.device)
            if adapter.config.layout_kv == "PA_NZ"
            else None,
            block_columns=torch.arange(columns, dtype=torch.int64, device=self.device)
            if adapter.config.layout_kv == "PA_NZ"
            else None,
        )
        if adapter.config.layout_kv == "PA_NZ" and self.dcp_size > 1 and causal and self.use_spec_decode:
            # Only PA_NZ needs the replicated speculative current chunk.
            block_columns = (tokens + FLASHMLA_BLOCK_SIZE - 1) // FLASHMLA_BLOCK_SIZE
            current_schedule = self.current_adapter.build_metadata(
                torch.empty_like(used, device="meta"),
                torch.empty_like(cu, device="meta"),
                torch.empty_like(used, device="meta"),
            )
            flash.current = FlashMLACurrent(
                adapter=self.current_adapter,
                query=torch.empty(
                    (tokens, self.current_adapter.config.num_heads, FLASHMLA_QK_DIM),
                    dtype=self.dtype,
                    device=self.device,
                ),
                schedule=torch.empty_like(current_schedule, device=self.device),
                cache=torch.empty(
                    (block_columns + rows, FLASHMLA_BLOCK_SIZE, 1, FLASHMLA_QK_DIM),
                    dtype=self.dtype,
                    device=self.device,
                ),
                block_table=torch.zeros((rows, block_columns), **ints),
                slots=torch.full_like(flash.slots, -1),
                attn_mask=self.attn_mask,
            )
            if adapter.config.layout_kv == "PA_NZ":
                # The PA_NZ prolog/cache contract uses a fixed 128-token page
                # and 16-element tiles.  Keep every tensor at a stable address
                # so graph replay and all MLA layers can reuse the metadata.
                tile_offsets = (
                    torch.arange(PA_NZ_TILE_COUNT, dtype=torch.int64, device=self.device) * FLASHMLA_BLOCK_SIZE
                )
                flash.pa_nz_tile_offsets = tile_offsets
                flash.pa_nz_current_slots = torch.zeros_like(flash.current.slots)
                flash.pa_nz_history_slots = torch.zeros_like(flash.slots)
                index_shape = (tokens * PA_NZ_TILE_COUNT,)
                index_ints = {"dtype": torch.int64, "device": self.device}
                flash.pa_nz_copy_src_indices = torch.zeros(index_shape, **index_ints)
                flash.pa_nz_copy_dst_indices = torch.zeros(index_shape, **index_ints)
        return flash

    def build(
        self, common, num_decodes: int, num_decode_tokens: int, has_prefill: bool, *, retain_for_graph: bool = False
    ) -> FlashMLADecode:
        # Mixed batches run outside FULL graph. Do not retain every prefill
        # shape. Retain only explicitly captured decode capacities: retaining
        # every eager token count grows total query storage quadratically.
        # Replay (and eager with that capacity) reuses the captured addresses.
        tokens = num_decode_tokens if has_prefill else max(common.num_actual_tokens, common.num_input_tokens)
        rows = num_decodes + 1 if has_prefill else max(num_decodes, min(tokens, self.max_num_reqs)) + 1
        columns = common.block_table_tensor.shape[1]
        q_len = max(1, int(getattr(common, "max_query_len", 1) or 1))
        # Native metadata derives variable query lengths from device boundaries;
        # only PA_NZ specializes its schedule by the host query-length bound.
        pa_nz = self.adapters[common.causal].config.layout_kv == "PA_NZ"
        key = tokens, rows, columns, common.causal, q_len if pa_nz else 1
        # Captured buffers run their producer in the graph on every replay.
        # Other batches can fork as soon as the schedule inputs are ready.
        start_during_refresh = self.enable_async and not retain_for_graph and (has_prefill or key not in self.buffers)
        if has_prefill:
            if retain_for_graph:
                raise ValueError("FlashMLA graph buffers require a decode-only batch")
            flash = self._allocate(tokens, rows, columns, common.causal, q_len=q_len)
            flash.metadata_async = self.enable_async
        else:
            cached_flash = self.buffers.get(key)
            if cached_flash is None:
                flash = self._allocate(tokens, rows, columns, common.causal, q_len=q_len)
                # Captured forwards record the stream fork/join inside the
                # graph, so replay rebuilds the schedule without Python.
                flash.metadata_async = self.enable_async
                if retain_for_graph:
                    self.buffers[key] = flash
            else:
                flash = cached_flash

        flash.q_len = q_len
        flash.metadata_builder = self

        def refresh():
            flash.metadata_started = False
            flash.metadata_capture_started = False
            # The extra zero-used request owns physical graph/SP padding.
            flash.cu.fill_(tokens)
            flash.cu[: num_decodes + 1].copy_(common.query_start_loc[: num_decodes + 1].clamp_max(num_decode_tokens))
            flash.used_q.zero_()
            flash.used_q[:num_decodes].copy_(flash.cu[1 : num_decodes + 1] - flash.cu[:num_decodes])
            flash.used_q[:num_decodes].masked_fill_(common.seq_lens[:num_decodes] <= 0, 0)
            flash.cache_lens.zero_()
            if flash.global_cache_lens is not None:
                flash.global_cache_lens.zero_()
            # One call covers the whole decode range, so the KV length keeps the
            # current rows; the metadata operator derives the causal boundary.
            global_lengths = common.seq_lens[:num_decodes]
            if flash.current is not None:
                global_lengths = (global_lengths - flash.used_q[:num_decodes]).clamp_min(0)
            lengths = global_lengths
            if self.dcp_size > 1:
                # Both causal and noncausal attention read only this rank's
                # shard, independently of the operator's causal mode.
                lengths = get_dcp_local_seq_lens(
                    lengths,
                    dcp_size=self.dcp_size,
                    dcp_rank=self.dcp_rank_tensor,
                    cp_kv_cache_interleave_size=self.cp_interleave,
                )
            else:
                lengths = lengths.clamp_min(0)
            flash.cache_lens[:num_decodes].copy_(lengths)
            flash.cache_lens.masked_fill_(flash.used_q == 0, 0)
            if flash.global_cache_lens is not None:
                flash.global_cache_lens[:num_decodes].copy_(global_lengths)
                flash.global_cache_lens.masked_fill_(flash.used_q == 0, 0)
            if start_during_refresh:
                self.start_metadata(flash)
            flash.block_table.zero_()
            flash.block_table[:num_decodes].copy_(common.block_table_tensor[:num_decodes])
            flash.slots.fill_(-1)
            slots = common.slot_mapping[:num_decode_tokens]
            flash.slots[: slots.numel()].copy_(slots)
            flash.live_boundaries.zero_()
            live = (flash.used_q > 0).to(torch.int32)
            flash.live_boundaries.scatter_add_(0, flash.cu[:-1].long(), live)
            flash.live_boundaries.scatter_add_(0, (flash.cu[:-1] + flash.used_q).long(), -live)
            flash.token_live.copy_(flash.live_boundaries.cumsum(0)[:tokens] > 0)
            flash.slots.masked_fill_(~flash.token_live, -1)
            flash.positions.zero_()
            positions = common.positions[:num_decode_tokens]
            flash.positions[: positions.numel()].copy_(positions)
            if self.use_rope:
                assert flash.cos is not None and flash.sin is not None
                cos, sin = get_cos_and_sin_mla(flash.positions)
                flash.cos.copy_(cos)
                flash.sin.copy_(sin)
            if flash.prolog_slots is not None:
                # Prolog must never receive -1 for graph padding or an empty
                # request.  Do this once with the batch metadata instead of
                # masking the same slots in every MLA layer.
                flash.prolog_slots.copy_(torch.where(flash.token_live & (flash.slots >= 0), flash.slots, 0))
            if flash.current is not None:
                current = flash.current
                page_counts = (flash.used_q + FLASHMLA_BLOCK_SIZE - 1) // FLASHMLA_BLOCK_SIZE
                page_starts = page_counts.cumsum(0) - page_counts
                columns = torch.arange(current.block_table.shape[1], device=self.device)
                current.block_table.copy_(page_starts[:, None] + columns)
                current.block_table.masked_fill_(columns >= page_counts[:, None], 0)
                indices = torch.arange(tokens, device=self.device)
                requests = torch.searchsorted(flash.cu[1:], indices.to(torch.int32), right=True)
                current.slots.copy_(page_starts[requests] * FLASHMLA_BLOCK_SIZE + indices - flash.cu[requests])
                current.slots.masked_fill_(~flash.token_live, -1)
                if flash.pa_nz_current_slots is not None:
                    assert flash.pa_nz_history_slots is not None
                    assert flash.pa_nz_copy_src_indices is not None
                    assert flash.pa_nz_copy_dst_indices is not None
                    assert flash.pa_nz_tile_offsets is not None
                    current_valid = flash.token_live & (current.slots >= 0)
                    history_valid = flash.token_live & (flash.slots >= 0)
                    scratch_slot = (current.cache.shape[0] - 1) * FLASHMLA_BLOCK_SIZE
                    flash.pa_nz_current_slots.copy_(torch.where(current_valid, current.slots, scratch_slot))
                    flash.pa_nz_history_slots.copy_(torch.where(history_valid, flash.slots, 0))
                    # Keep the old copy semantics: invalid source rows read
                    # the current scratch page and invalid destination rows
                    # target slot zero.  The slot decoding now happens once
                    # per batch rather than once per MLA layer.
                    flash.pa_nz_copy_src_indices.copy_(
                        _pa_nz_tile_indices(
                            flash.pa_nz_current_slots,
                            FLASHMLA_BLOCK_SIZE,
                            flash.pa_nz_tile_offsets,
                            PA_NZ_TILE_COUNT,
                        )
                    )
                    flash.pa_nz_copy_dst_indices.copy_(
                        _pa_nz_tile_indices(
                            flash.pa_nz_history_slots,
                            FLASHMLA_BLOCK_SIZE,
                            flash.pa_nz_tile_offsets,
                            PA_NZ_TILE_COUNT,
                        )
                    )

        # Refresh on the caller's stream before return; captured buffer caching
        # preserves the addresses used by graph replay.
        refresh()
        if not flash.metadata_async:
            self._refresh_schedule(flash)
        return flash

    def _refresh_schedule(self, flash: FlashMLADecode) -> None:
        if flash.layout == "PA_NZ":
            schedule = flash.adapter.build_metadata(flash.cache_lens, flash.cu, flash.used_q, q_len=flash.q_len)
        else:
            schedule = flash.adapter.build_metadata(flash.cache_lens, flash.cu, flash.used_q, flash.global_cache_lens)
        flash.schedule.copy_(schedule)
        if flash.current is not None:
            current = flash.current
            current.schedule.copy_(
                current.adapter.build_metadata(
                    flash.used_q,
                    flash.cu,
                    flash.used_q,
                    q_len=flash.q_len,
                )
            )

    def start_metadata(self, flash: FlashMLADecode) -> None:
        """Build this step's schedule before the layer loop."""
        if not flash.metadata_async or flash.metadata_started:
            return
        if self.metadata_stream is None:
            flash.metadata_async = False
            self._refresh_schedule(flash)
            flash.metadata_started = True
            return
        main_stream = torch.npu.current_stream()
        if self.metadata_stream == main_stream:
            raise RuntimeError("FlashMLA metadata stream aliases the current execution stream")
        flash.metadata_start = main_stream.record_event()
        self.metadata_stream.wait_event(flash.metadata_start)
        self._record_metadata_stream(flash)
        with torch.npu.stream(self.metadata_stream):
            self._refresh_schedule(flash)
            flash.metadata_ready = torch.npu.current_stream().record_event()
        flash.metadata_started = True

    def _record_metadata_stream(self, flash: FlashMLADecode) -> None:
        """Keep side-stream schedule inputs alive until the fork/join completes."""
        if self.metadata_stream is None:
            return
        for tensor in (flash.schedule, flash.cache_lens, flash.cu, flash.used_q, flash.global_cache_lens):
            if tensor is not None:
                tensor.record_stream(self.metadata_stream)
        if flash.current is not None:
            flash.current.schedule.record_stream(self.metadata_stream)

    def wait_metadata(self, flash: FlashMLADecode) -> None:
        """Make the main stream wait for an in-flight async schedule build."""
        if not flash.metadata_async or not flash.metadata_started:
            return
        assert flash.metadata_ready is not None
        torch.npu.current_stream().wait_event(flash.metadata_ready)
