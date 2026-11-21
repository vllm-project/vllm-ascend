# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Paged verify and decode backend for head_dim=512 speculative attention.

On A2/A3 hardware, ``npu_fused_infer_attention_score`` in TND layout does not
support ``head_dim == 512``, while ``torch_npu._npu_paged_attention`` assumes a
single query token per batch row and does not support multi-query speculative
verification natively without per-token row expansion.

For decoder self-attention layers with ``head_dim == 512`` and
``sliding_window is None`` when speculative decoding is enabled on devices
with ``HardwareCapability.PAGED_ATTENTION``, this backend:

* In ``FULL_DECODE_ONLY`` (and eager) mode, allocates an Hkv-major
  (``[num_blocks, num_kv_heads, block_size, head_dim]``) contiguous KV cache
  and runs uniform verify/decode steps through
  ``npu_fused_infer_attention_score_v2`` in ``BSND`` query layout with
  per-request ``block_table`` and either the static causal mask with
  ``sparse_mode=3`` (for ``qlen > 1``) or maskless ``sparse_mode=0`` (for
  ``qlen == 1``).
* In ``CUDAGraphMode.FULL`` mode, keeps the natural LBNHC KV cache layout and
  expands multi-query rows into one-query-per-row metadata for
  ``full_graph_pa``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch_npu
import vllm.envs as envs_vllm

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import (
    _PA_WORKSPACE_KEY,
    SWA_INT_MAX,
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import (
    FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE,
    build_pa_per_token_rows,
    in_full_cudagraph_mode,
)
from vllm_ascend.compilation.updatable_graph import (
    get_capture_resource,
    register_task,
)
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.device.utils import (
    _gather_paged_kv_to_dense,
    _run_large_head_fusion_attention,
)

_FIA_V2_WORKSPACE_KEY = "npu_fused_infer_attention_score_v2.workspace"


def _supports_large_head_paged_path() -> bool:
    """Return True on devices that use A2/A3 paged attention for 512-dim heads."""
    return get_current_hardware_profile().supports(HardwareCapability.PAGED_ATTENTION)


def _uniform_query_len(actual_seq_lengths_q: list[int] | None) -> int | None:
    """Return the per-request query length if uniform and positive across the batch."""
    if not actual_seq_lengths_q:
        return None
    prev = 0
    query_len = actual_seq_lengths_q[0]
    if query_len <= 0:
        return None
    for end in actual_seq_lengths_q:
        if end - prev != query_len:
            return None
        prev = end
    return query_len


def build_paged_verify_mask(
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_speculative_tokens: int,
    block_size: int,
    device: torch.device,
    *,
    kpos: torch.Tensor | None = None,
    qpos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the ``[B, 1, k+1, W]`` bool verify mask for BNSD paged attention.

    Query row ``j`` (``0..k``) of request ``b`` attends key positions
    ``0..seq_lens[b] - (k + 1) + j`` (inclusive), where ``True`` means masked
    out. Real rows satisfy ``seq_lens[b] >= k + 1``; ``clamp_min(0)`` keeps key
    slot 0 unmasked on padded/dummy rows with ``seq_lens[b] < k + 1`` so
    softmax never observes an all-masked row.
    """
    qlen = num_speculative_tokens + 1
    batch_size = seq_lens.shape[0]
    width = block_table.shape[1] * block_size
    if kpos is None:
        kpos = torch.arange(width, dtype=torch.int32, device=device).view(1, 1, 1, width)
    if qpos is None:
        qpos = torch.arange(qlen, dtype=torch.int32, device=device).view(1, 1, qlen, 1)
    seqv = seq_lens.to(device=device, dtype=torch.int32).view(batch_size, 1, 1, 1)
    last_allowed = (seqv - qlen + qpos).clamp_min(0)
    return kpos > last_allowed


def _can_use_large_head_bnsd_verify(impl: Any, attn_metadata: Any, num_tokens: int) -> bool:
    """Return True when the layer geometry and query count fit BNSD paged verify/decode."""
    if (
        getattr(impl, "head_size", None) != FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
        or getattr(impl, "sliding_window", None) is not None
        or not _supports_large_head_paged_path()
    ):
        return False
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None or seq_lens.shape[0] <= 0 or num_tokens % seq_lens.shape[0] != 0:
        return False
    query_len = num_tokens // seq_lens.shape[0]
    spec_cfg = getattr(getattr(impl, "vllm_config", None), "speculative_config", None)
    verify_len = spec_cfg.num_speculative_tokens + 1 if spec_cfg is not None else None
    if _EXTRA_CTX.is_draft_model:
        return query_len == 1 or (verify_len is not None and query_len == verify_len)
    return verify_len is not None and query_len == verify_len


@dataclass
class AscendLargeHeadPagedMetadata(AscendMetadata):
    """Per-layer metadata carrying the BNSD verify mask or FULL PA toggle."""

    paged_verify_mask: torch.Tensor | None = None
    use_pa_graph: bool = False
    pa_block_tables: torch.Tensor | None = None
    pa_context_lens: torch.Tensor | None = None


def _clamp_verify_seq_kvlen(seq_lens_list: Sequence[int], query_len: int) -> list[int]:
    """Clamp padded/dummy KV lengths to ``>= query_len`` for ``sparse_mode=3``."""
    if query_len <= 1:
        return [int(s) for s in seq_lens_list]
    return [max(int(s), query_len) for s in seq_lens_list]


def large_head_verify_graph_params(
    layer_name: str | None,
    metadata: AscendLargeHeadPagedMetadata,
) -> dict[str, Any]:
    """Return replay-time graph update kwargs for a head_dim=512 verify/draft task."""
    del layer_name
    if getattr(metadata, "use_pa_graph", False):
        if metadata.pa_block_tables is not None and metadata.pa_context_lens is not None:
            return {
                "block_table": metadata.pa_block_tables,
                "context_lens": metadata.pa_context_lens,
            }
        # FULL mode fallback when caller expanded block_tables directly.
        num_pa_rows = metadata.seq_lens.shape[0] if metadata.seq_lens is not None else 0
        block_table = (
            metadata.block_tables[:num_pa_rows]
            if num_pa_rows > 0 and metadata.block_tables is not None and metadata.block_tables.shape[0] > num_pa_rows
            else metadata.block_tables
        )
        return {
            "block_table": block_table,
            "context_lens": metadata.seq_lens,
        }
    num_reqs = len(metadata.actual_seq_lengths_q) if metadata.actual_seq_lengths_q else 0
    block_table = (
        metadata.block_tables[:num_reqs]
        if num_reqs > 0 and metadata.block_tables is not None and metadata.block_tables.shape[0] > num_reqs
        else metadata.block_tables
    )
    query_len = int(metadata.actual_seq_lengths_q[0]) if metadata.actual_seq_lengths_q else 0
    seq_lens_list = (
        metadata.seq_lens_list[:num_reqs]
        if num_reqs > 0 and metadata.seq_lens_list is not None and len(metadata.seq_lens_list) > num_reqs
        else metadata.seq_lens_list
    )
    actual_seq_kvlen = (
        _clamp_verify_seq_kvlen(seq_lens_list, query_len)
        if query_len > 1 and seq_lens_list is not None
        else seq_lens_list
    )
    return {
        "block_table": block_table,
        "actual_seq_kvlen": actual_seq_kvlen,
    }


def large_head_decode_graph_params(
    metadata: AscendLargeHeadPagedMetadata,
    seq_lens: list[int],
) -> dict[str, Any]:
    """Return replay-time graph update kwargs for a draft head_dim=512 decode step."""
    num_reqs = len(seq_lens)
    block_table = (
        metadata.block_tables[:num_reqs]
        if num_reqs > 0 and metadata.block_tables is not None and metadata.block_tables.shape[0] > num_reqs
        else metadata.block_tables
    )
    if getattr(metadata, "use_pa_graph", False):
        return {
            "block_table": block_table,
            "context_lens": torch.tensor(seq_lens, dtype=torch.int32, device=metadata.seq_lens.device),
        }
    return {
        "block_table": block_table,
        "actual_seq_kvlen": seq_lens,
    }


@dataclass(frozen=True, slots=True)
class LargeHeadPagedVerifyParamProvider:
    """Graph parameter provider keyed solely by layer name for SharedSource matching."""

    layer_name: str | None

    def resolve(self, attn_metadata: dict[str, AscendLargeHeadPagedMetadata]) -> dict[str, Any]:
        return large_head_verify_graph_params(self.layer_name, attn_metadata[self.layer_name])  # type: ignore[index]


class AscendLargeHeadPagedMetadataBuilder(AscendAttentionMetadataBuilder):
    """Metadata builder for head_dim=512 non-sliding speculative attention layers."""

    metadata_cls: type[AscendLargeHeadPagedMetadata] = AscendLargeHeadPagedMetadata

    def _attach_pa_graph_rows(self, metadata: AscendLargeHeadPagedMetadata, *, for_capture: bool) -> None:
        metadata.use_pa_graph = True
        if not for_capture:
            max_capture = getattr(
                getattr(self.vllm_config, "compilation_config", None),
                "max_cudagraph_capture_size",
                None,
            )
            if (
                isinstance(max_capture, int)
                and max_capture > 0
                and metadata.actual_seq_lengths_q
                and metadata.actual_seq_lengths_q[-1] > max_capture
            ):
                return
        seq_lens_device = (
            metadata.seq_lens.device if isinstance(metadata.seq_lens, torch.Tensor) else torch.device("cpu")
        )
        expanded = build_pa_per_token_rows(
            metadata.block_tables,
            metadata.actual_seq_lengths_q,
            metadata.seq_lens_list,
            context_lens_device=seq_lens_device,
        )
        if expanded is not None:
            metadata.pa_block_tables, metadata.pa_context_lens = expanded

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> AscendLargeHeadPagedMetadata:
        metadata = cast(
            AscendLargeHeadPagedMetadata,
            super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build),
        )
        if getattr(
            self.kv_cache_spec, "head_size", None
        ) == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE and in_full_cudagraph_mode(self.vllm_config):
            self._attach_pa_graph_rows(metadata, for_capture=False)
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs) -> AscendLargeHeadPagedMetadata:
        # Skip runtime-only metadata work on the capture stream; graph capture
        # binds the static causal mask with sparse_mode=3 instead.
        metadata = cast(
            AscendLargeHeadPagedMetadata,
            AscendAttentionMetadataBuilder.build(
                self,
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            ),
        )
        if getattr(
            self.kv_cache_spec, "head_size", None
        ) == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE and in_full_cudagraph_mode(self.vllm_config):
            self._attach_pa_graph_rows(metadata, for_capture=True)
        return metadata


class AscendLargeHeadPagedAttentionBackendImpl(AscendAttentionBackendImpl):
    """Ascend attention impl adding BSND FIA-v2 paged verify/decode for 512-dim heads."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.full_cudagraph = in_full_cudagraph_mode(getattr(self, "vllm_config", None))
        if (
            not self.full_cudagraph
            and not self.use_bnsd_kv_cache
            and self.head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
            and _supports_large_head_paged_path()
        ):
            self.use_bnsd_kv_cache = True

    def _should_capture_bnsd_verify(self, attn_metadata: Any, num_tokens: int) -> bool:
        return (
            bool(_EXTRA_CTX.capturing)
            and not self.full_cudagraph
            and _can_use_large_head_bnsd_verify(self, attn_metadata, num_tokens)
        )

    def _maybe_init_kv_cache(self, kv_cache: Any = None) -> None:
        if (
            self.key_cache is None
            and kv_cache is not None
            and (
                (isinstance(kv_cache, torch.Tensor) and kv_cache.dim() > 0 and kv_cache.shape[0] == 2)
                or (isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2)
            )
        ):
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

    def _get_static_verify_mask(self, attn_metadata: AscendMetadata, device: torch.device) -> torch.Tensor:
        """Return the static [2048, 2048] int8 causal mask required by sparse_mode=3."""
        attn_mask = getattr(attn_metadata, "attn_mask", None)
        if (
            isinstance(attn_mask, torch.Tensor)
            and attn_mask.ndim == 2
            and attn_mask.shape[0] == attn_mask.shape[1]
            and attn_mask.dtype == torch.int8
            and attn_mask.device == device
        ):
            return attn_mask
        return AttentionMaskBuilder(device).get_splitfuse_attn_mask()

    def _should_eager_bnsd_verify(self, attn_metadata: Any, num_tokens: int, kv_cache: Any = None) -> bool:
        self._maybe_init_kv_cache(kv_cache)
        if (
            _EXTRA_CTX.capturing
            or self.full_cudagraph
            or not self.use_bnsd_kv_cache
            or self.key_cache is None
            or self.value_cache is None
            or not (self.key_cache.is_contiguous() and self.value_cache.is_contiguous())
        ):
            return False
        if (
            not getattr(attn_metadata, "causal", True)
            or getattr(attn_metadata, "attn_state", None) == AscendAttentionState.PrefillNoCache
            or getattr(attn_metadata, "num_prefills", 0) > 0
            or getattr(attn_metadata, "block_tables", None) is None
        ):
            return False
        if not _can_use_large_head_bnsd_verify(self, attn_metadata, num_tokens):
            return False
        actual_seq_lengths_q = getattr(attn_metadata, "actual_seq_lengths_q", None)
        qlen = _uniform_query_len(actual_seq_lengths_q)
        if qlen is None or actual_seq_lengths_q is None or num_tokens != qlen * len(actual_seq_lengths_q):
            return False
        num_reqs = len(actual_seq_lengths_q)
        if attn_metadata.block_tables.shape[0] < num_reqs:
            return False
        seq_lens_list = getattr(attn_metadata, "seq_lens_list", None) or []
        return len(seq_lens_list) >= num_reqs and all(int(s) > 0 for s in seq_lens_list[:num_reqs])

    def _can_use_large_head_eager_phase_split(
        self,
        attn_metadata: Any,
        num_tokens: int,
        kv_cache: Any = None,
    ) -> bool:
        self._maybe_init_kv_cache(kv_cache)
        if (
            _EXTRA_CTX.capturing
            or self.head_size != FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
            or self.sliding_window is not None
            or not _supports_large_head_paged_path()
            or self.key_cache is None
            or self.value_cache is None
            or not (self.key_cache.is_contiguous() and self.value_cache.is_contiguous())
        ):
            return False
        if (
            not getattr(attn_metadata, "causal", True)
            or getattr(attn_metadata, "attn_state", None) == AscendAttentionState.PrefillNoCache
            or getattr(attn_metadata, "block_tables", None) is None
        ):
            return False
        actual_seq_q = getattr(attn_metadata, "actual_seq_lengths_q", None)
        seq_lens_list = getattr(attn_metadata, "seq_lens_list", None)
        num_decodes = getattr(attn_metadata, "num_decodes", 0)
        num_prefills = getattr(attn_metadata, "num_prefills", 0)
        num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", 0)
        if not actual_seq_q or not seq_lens_list or num_decodes <= 0 or num_prefills < 0 or num_decode_tokens <= 0:
            return False
        num_reqs = len(actual_seq_q)
        if (
            num_decodes + num_prefills != num_reqs
            or int(actual_seq_q[num_decodes - 1]) != num_decode_tokens
            or int(actual_seq_q[-1]) > num_tokens
            or len(seq_lens_list) < num_reqs
            or attn_metadata.block_tables.shape[0] < num_reqs
        ):
            return False
        prev = 0
        for end, seq_len in zip(actual_seq_q, seq_lens_list):
            qlen_i = int(end) - prev
            if qlen_i <= 0 or int(seq_len) < qlen_i:
                return False
            prev = int(end)
        return True

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Capture 512-dim FULL mode PagedAttention using non-mutating per-token PA metadata."""
        assert output is not None, "Output tensor must be provided for full_graph_pa."
        pa_block_table = getattr(attn_metadata, "pa_block_tables", None)
        block_table = pa_block_table if pa_block_table is not None else attn_metadata.block_tables
        pa_context_lens = getattr(attn_metadata, "pa_context_lens", None)
        context_lens = pa_context_lens if pa_context_lens is not None else attn_metadata.seq_lens
        workspace = get_capture_resource(
            _PA_WORKSPACE_KEY,
            lambda: torch_npu._npu_paged_attention_get_workspace(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=block_table,
                context_lens=context_lens,
                out=output,
            ),
        )
        register_task(
            torch_npu._npu_paged_attention,
            {
                "query": query,
                "key_cache": self.key_cache,
                "value_cache": self.value_cache,
                "num_kv_heads": self.num_kv_heads,
                "num_heads": self.num_heads,
                "scale_value": self.scale,
                "block_table": block_table,
                "context_lens": context_lens,
                "out": output,
                "workspace": workspace,
            },
            LargeHeadPagedVerifyParamProvider(self._layer_name),
        )
        return output

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        self._maybe_init_kv_cache(kv_cache)
        if self._should_capture_bnsd_verify(attn_metadata, query.shape[0]):
            return self.full_graph_fia_v2_paged_verify(query, attn_metadata, output)
        if (
            _EXTRA_CTX.capturing
            and self.head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
            and self.sliding_window is None
            and _supports_large_head_paged_path()
        ):
            return self.full_graph_pa(query, attn_metadata, output)
        if self._should_eager_bnsd_verify(attn_metadata, query.shape[0], kv_cache):
            return self._eager_fia_v2_paged_verify(query, attn_metadata, output)
        if self._can_use_large_head_eager_phase_split(attn_metadata, query.shape[0], kv_cache):
            return self._forward_large_head_chunked_prefill(query, key, value, attn_metadata, output)
        return super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._should_capture_bnsd_verify(attn_metadata, query.shape[0]):
            assert output is not None
            return self.full_graph_fia_v2_paged_verify(query, attn_metadata, output)
        if self._should_eager_bnsd_verify(attn_metadata, query.shape[0]):
            assert output is not None
            return self._eager_fia_v2_paged_verify(query, attn_metadata, output)
        return super().forward_paged_attention(query, attn_metadata, output)

    def _run_bsnd_fia_v2(
        self,
        q_bsnd: torch.Tensor,
        block_table: torch.Tensor,
        atten_mask: torch.Tensor | None,
        actual_seq_qlen: list[int],
        actual_seq_kvlen: list[int],
        block_size: int,
        *,
        sparse_mode: int = 0,
    ) -> torch.Tensor:
        """Execute ``npu_fused_infer_attention_score_v2`` in BSND query + BNSD KV layout."""
        extra_args = {"pre_tokens": SWA_INT_MAX, "next_tokens": 0} if sparse_mode == 3 else {}
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=q_bsnd,
            key=self.key_cache,
            value=self.value_cache,
            atten_mask=atten_mask,
            block_table=block_table,
            input_layout="BSND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=sparse_mode,
            softmax_scale=self.scale,
            **extra_args,
        )
        return attn_output

    def _run_bnsd_fia_v2(
        self,
        q_bnsd: torch.Tensor,
        block_table: torch.Tensor,
        verify_mask: torch.Tensor | None,
        actual_seq_qlen: list[int],
        actual_seq_kvlen: list[int],
        block_size: int,
    ) -> torch.Tensor:
        """Fallback BNSD FIA-v2 execution for synthetic eager rows with ``seq_len < qlen``."""
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=q_bnsd,
            key=self.key_cache,
            value=self.value_cache,
            atten_mask=verify_mask,
            block_table=block_table,
            input_layout="BNSD",
            block_size=block_size,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=0,
            softmax_scale=self.scale,
        )
        return attn_output

    def _execute_bsnd_verify_or_fallback(
        self,
        q_bsnd: torch.Tensor,
        block_table: torch.Tensor,
        attn_metadata: AscendMetadata,
        out_slice: torch.Tensor,
        num_reqs: int,
        qlen: int,
        seq_lens: list[int],
        block_size: int,
    ) -> None:
        if qlen == 1 or all(s >= qlen for s in seq_lens):
            atten_mask = self._get_static_verify_mask(attn_metadata, q_bsnd.device) if qlen > 1 else None
            sparse_mode = 3 if qlen > 1 else 0
            attn_out = self._run_bsnd_fia_v2(
                q_bsnd,
                block_table,
                atten_mask,
                [qlen] * num_reqs,
                seq_lens,
                block_size,
                sparse_mode=sparse_mode,
            )
            out_slice.copy_(attn_out.reshape(out_slice.shape))
            return

        verify_mask = getattr(attn_metadata, "paged_verify_mask", None)
        expected_shape = (num_reqs, 1, qlen, block_table.shape[1] * block_size)
        if verify_mask is None or verify_mask.shape != expected_shape:
            verify_mask = build_paged_verify_mask(
                torch.tensor(seq_lens, dtype=torch.int32, device=q_bsnd.device),
                block_table,
                qlen - 1,
                block_size,
                q_bsnd.device,
            )
        q_bnsd = q_bsnd.permute(0, 2, 1, 3).contiguous()
        attn_out = self._run_bnsd_fia_v2(q_bnsd, block_table, verify_mask, [qlen] * num_reqs, seq_lens, block_size)
        out_slice.copy_(attn_out.permute(0, 2, 1, 3).reshape(out_slice.shape))

    def _forward_large_head_eager_decode_slice(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
    ) -> None:
        """Run decode/verify requests via paged BSND FIA-v2 or PA without dense KV gather."""
        dec_q = attn_metadata.actual_seq_lengths_q[:num_decodes]
        dec_kv = [int(s) for s in attn_metadata.seq_lens_list[:num_decodes]]
        dec_bt = attn_metadata.block_tables[:num_decodes]
        out_slice = output[:num_decode_tokens]
        if not self.use_bnsd_kv_cache:
            expanded = build_pa_per_token_rows(dec_bt, dec_q, dec_kv, context_lens_device=torch.device("cpu"))
            assert expanded is not None
            torch_npu._npu_paged_attention(
                query=query[:num_decode_tokens],
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=expanded[0],
                context_lens=expanded[1],
                out=out_slice,
            )
            return

        assert self.key_cache is not None
        _, _, block_size, _ = self.key_cache.shape
        qlen = _uniform_query_len(dec_q)
        if qlen is not None and qlen > 1:
            q_bsnd = query[:num_decode_tokens].contiguous().view(num_decodes, qlen, self.num_heads, self.head_size)
            self._execute_bsnd_verify_or_fallback(
                q_bsnd, dec_bt, attn_metadata, out_slice, num_decodes, qlen, dec_kv, block_size
            )
            return

        if qlen == 1:
            row_count, row_bt, row_kv = num_decodes, dec_bt, dec_kv
        else:
            expanded = build_pa_per_token_rows(dec_bt, dec_q, dec_kv)
            assert expanded is not None
            row_bt, row_kv_tensor = expanded
            row_count, row_kv = num_decode_tokens, row_kv_tensor.tolist()
        q_bsnd = query[:num_decode_tokens].contiguous().view(row_count, 1, self.num_heads, self.head_size)
        attn_out = self._run_bsnd_fia_v2(q_bsnd, row_bt, None, [1] * row_count, row_kv, block_size, sparse_mode=0)
        out_slice.copy_(attn_out.reshape(out_slice.shape))

    def _forward_large_head_eager_prefill_slice(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        total_tokens: int,
    ) -> None:
        """Run only prefill requests through ``npu_fusion_attention`` (zero decode gather)."""
        num_reqs = len(attn_metadata.actual_seq_lengths_q)
        prefill_seq_qlen = [
            int(attn_metadata.actual_seq_lengths_q[i]) - num_decode_tokens for i in range(num_decodes, num_reqs)
        ]
        prefill_qlens = [
            int(attn_metadata.actual_seq_lengths_q[i])
            - (int(attn_metadata.actual_seq_lengths_q[i - 1]) if i > 0 else 0)
            for i in range(num_decodes, num_reqs)
        ]
        prefill_seq_lens = [int(attn_metadata.seq_lens_list[i]) for i in range(num_decodes, num_reqs)]
        all_new_prefill = (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and all(kv_len == q_len for kv_len, q_len in zip(prefill_seq_lens, prefill_qlens))
        )
        if all_new_prefill:
            prefill_k = key[num_decode_tokens:total_tokens].contiguous()
            prefill_v = value[num_decode_tokens:total_tokens].contiguous()
            prefill_seq_kvlen = prefill_seq_qlen
        else:
            assert self.key_cache is not None and self.value_cache is not None
            prefill_k, prefill_v = _gather_paged_kv_to_dense(
                self.key_cache,
                self.value_cache,
                attn_metadata.block_tables[num_decodes:num_reqs],
                prefill_seq_lens,
                self.num_kv_heads,
                self.head_size,
                use_bnsd=self.use_bnsd_kv_cache,
            )
            prefill_seq_kvlen = []
            cumsum = 0
            for length in prefill_seq_lens:
                cumsum += length
                prefill_seq_kvlen.append(cumsum)
        prefill_out = _run_large_head_fusion_attention(
            query[num_decode_tokens:total_tokens].contiguous(),
            prefill_k,
            prefill_v,
            prefill_seq_qlen,
            prefill_seq_kvlen,
            attn_metadata.attn_mask,
            num_heads=self.num_heads,
            scale=self.scale,
            causal=True,
        )
        out_slice = output[num_decode_tokens:total_tokens]
        out_slice.copy_(prefill_out.reshape(out_slice.shape))

    def _forward_large_head_chunked_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Split eager mixed/non-uniform large-head batches into paged decode and prefill."""
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        total_tokens = int(attn_metadata.actual_seq_lengths_q[-1])
        self._forward_large_head_eager_decode_slice(query, attn_metadata, output, num_decodes, num_decode_tokens)
        if attn_metadata.num_prefills > 0 and total_tokens > num_decode_tokens:
            self._forward_large_head_eager_prefill_slice(
                query, key, value, attn_metadata, output, num_decodes, num_decode_tokens, total_tokens
            )
        return output

    def _prepare_bsnd_paged_inputs(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> tuple[int, int, int, int, torch.Tensor, torch.Tensor]:
        if not self.use_bnsd_kv_cache:
            raise RuntimeError(
                "Large-head paged verify requires BNSD KV cache layout "
                "(AscendLargeHeadPagedBackend.get_kv_cache_shape)."
            )
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("Large-head paged verify requires initialized KV caches.")
        if not (self.key_cache.is_contiguous() and self.value_cache.is_contiguous()):
            raise RuntimeError("Large-head paged verify requires contiguous BNSD KV cache tensors.")

        num_tokens = query.shape[0]
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        num_reqs = len(actual_seq_lengths_q) if actual_seq_lengths_q else 0
        if num_reqs == 0 or num_tokens % num_reqs != 0:
            raise RuntimeError(
                f"Large-head paged verify requires num_tokens divisible by request count; "
                f"got {num_tokens} tokens for {num_reqs} requests."
            )
        qlen = num_tokens // num_reqs
        if [qlen * (i + 1) for i in range(num_reqs)] != actual_seq_lengths_q:
            raise RuntimeError(f"Large-head paged verify requires uniform query lengths, got {actual_seq_lengths_q}.")

        _, _, block_size, _ = self.key_cache.shape
        if attn_metadata.block_tables is None:
            raise RuntimeError("Large-head paged verify requires non-None block_tables.")
        block_table = attn_metadata.block_tables[:num_reqs]
        q_bsnd = query[:num_tokens].contiguous().view(num_reqs, qlen, self.num_heads, self.head_size)
        return num_tokens, num_reqs, qlen, block_size, q_bsnd, block_table

    def full_graph_fia_v2_paged_verify(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Capture a 512-dim paged verify or draft decode step as FIA-v2 BSND."""
        num_tokens, num_reqs, qlen, block_size, q_bsnd, block_table = self._prepare_bsnd_paged_inputs(
            query, attn_metadata
        )
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        output_view = output[:num_tokens]
        zero_copy_out = output_view.is_contiguous()
        attn_output = (
            output_view.view(num_reqs, qlen, self.num_heads, self.head_size)
            if zero_copy_out
            else torch.empty(
                (num_reqs, qlen, self.num_heads, self.head_size),
                dtype=query.dtype,
                device=query.device,
            )
        )
        actual_seq_qlen = [qlen] * num_reqs
        seq_lens_slice = attn_metadata.seq_lens_list[:num_reqs]
        if qlen > 1:
            verify_mask: torch.Tensor | None = self._get_static_verify_mask(attn_metadata, query.device)
            sparse_mode = 3
            actual_seq_kvlen = _clamp_verify_seq_kvlen(seq_lens_slice, qlen)
            sparse_kwargs: dict[str, int] = {"pre_tokens": SWA_INT_MAX, "next_tokens": 0}
        else:
            verify_mask = None
            sparse_mode = 0
            actual_seq_kvlen = list(seq_lens_slice)
            sparse_kwargs = {}

        workspace = get_capture_resource(
            _FIA_V2_WORKSPACE_KEY,
            lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=q_bsnd,
                key=self.key_cache,
                value=self.value_cache,
                atten_mask=verify_mask,
                block_table=block_table,
                input_layout="BSND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=sparse_mode,
                **sparse_kwargs,
            ),
            use_max_workspace=True,
        )
        register_task(
            torch_npu.npu_fused_infer_attention_score_v2.out,
            {
                "query": q_bsnd,
                "key": self.key_cache,
                "value": self.value_cache,
                "atten_mask": verify_mask,
                "block_table": block_table,
                "input_layout": "BSND",
                "block_size": block_size,
                "actual_seq_qlen": actual_seq_qlen,
                "actual_seq_kvlen": actual_seq_kvlen,
                "num_key_value_heads": self.num_kv_heads,
                "num_query_heads": self.num_heads,
                "sparse_mode": sparse_mode,
                "softmax_scale": self.scale,
                "workspace": workspace,
                "out": [attn_output, softmax_lse],
                **sparse_kwargs,
            },
            LargeHeadPagedVerifyParamProvider(self._layer_name),
        )
        if not zero_copy_out:
            output_view.copy_(attn_output.reshape(output_view.shape))
        return output

    def _eager_fia_v2_paged_verify(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Execute a non-captured uniform 512-dim verify/decode step via FIA-v2 BSND."""
        num_tokens, num_reqs, qlen, block_size, q_bsnd, block_table = self._prepare_bsnd_paged_inputs(
            query, attn_metadata
        )
        seq_lens = [int(s) for s in attn_metadata.seq_lens_list[:num_reqs]]
        self._execute_bsnd_verify_or_fallback(
            q_bsnd,
            block_table,
            attn_metadata,
            output[:num_tokens],
            num_reqs,
            qlen,
            seq_lens,
            block_size,
        )
        return output


class AscendLargeHeadPagedBackend(AscendAttentionBackend):
    """Backend for head_dim=512 non-sliding layers under speculative decoding."""

    @staticmethod
    def get_impl_cls() -> type[AscendLargeHeadPagedAttentionBackendImpl]:
        return AscendLargeHeadPagedAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        return AscendLargeHeadPagedMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        if (
            head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
            and _supports_large_head_paged_path()
            and not in_full_cudagraph_mode()
        ):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        if envs_vllm.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND"):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)


__all__ = [
    "AscendLargeHeadPagedAttentionBackendImpl",
    "AscendLargeHeadPagedBackend",
    "AscendLargeHeadPagedMetadata",
    "AscendLargeHeadPagedMetadataBuilder",
    "LargeHeadPagedVerifyParamProvider",
    "build_paged_verify_mask",
    "large_head_decode_graph_params",
    "large_head_verify_graph_params",
]
