# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""B1 paged verify for Model Runner V2.

The V1 lane serves the Gemma4-style head_dim=512 MTP verify step with the
FIA-v2 BNSD paged op (the B1 blocks of the V1 patch). Model Runner V2 shares
``attention_v1`` with every other GQA backend and must not be specialized
there, so this V2 lane keeps the same op contract in a dedicated backend that
``AscendPlatform.get_attn_backend_cls`` selects per layer:

* only when ``VLLM_ASCEND_ENABLE_B1_PAGED_VERIFY`` is on,
* only on Model Runner V2 with speculative decoding (MTP),
* only for decoder self-attention with ``head_dim == 512`` and
  ``sliding_window is None``, and with no MLA/sparse/compress/PCP/DCP/
  KV-connector/sinks,
* never on A5, where the stock PagedAttention capture already works.

The head_dim=512 layers then allocate an Hkv-major K/V cache
(``[blocks, heads, block, dim]``) and capture the verify step as
``npu_fused_infer_attention_score_v2`` BNSD paged ops: per-seq block_table (no
per-query expansion), an explicit ``[B, 1, k+1, W]`` bool mask and
``sparse_mode=0``. FIA TND rejects head_dim=512, which is why the stock V2
capture paths cannot serve this geometry.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch_npu
import vllm.envs as envs_vllm

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.utils import FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
from vllm_ascend.compilation.updatable_graph import (
    get_capture_resource,
    register_task,
)

_FIA_V2_WORKSPACE_KEY = "npu_fused_infer_attention_score_v2.workspace"

_VERIFY_MASK_KPOS_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}
_VERIFY_MASK_KPOS_CACHE_MAX = 8


def build_paged_verify_mask(
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_speculative_tokens: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the [B, 1, k+1, W] bool mask for the B1 FIA-v2 BNSD paged verify.

    MTP verify semantics: query row j (j = 0..k) of request b may attend kv
    positions 0..seq_lens[b] - (k+1) + j. True = masked out. W is tied to the
    per-step block-table width so long contexts only pay for reachable blocks.

    Runs on the default stream at metadata-build time; the ACLGraph replay path
    synchronizes the current stream before the graph-param update stream
    consumes the result.
    """
    k = num_speculative_tokens
    qlen = k + 1
    B = seq_lens.shape[0]
    W = block_table.shape[1] * block_size
    cache_key = (W, device)
    kpos = _VERIFY_MASK_KPOS_CACHE.get(cache_key)
    if kpos is None:
        if len(_VERIFY_MASK_KPOS_CACHE) >= _VERIFY_MASK_KPOS_CACHE_MAX:
            _VERIFY_MASK_KPOS_CACHE.clear()
        kpos = torch.arange(W, device=device).view(1, 1, 1, W)
        _VERIFY_MASK_KPOS_CACHE[cache_key] = kpos
    seqv = seq_lens.to(device=device, dtype=torch.int32).view(B, 1, 1, 1)
    qpos = torch.arange(qlen, device=device).view(1, 1, qlen, 1)
    mask = kpos > (seqv - qlen + qpos)
    return mask


def _cudagraph_mode(vllm_config=None):
    """Resolved cudagraph mode of the active vLLM config, or None."""
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    compilation_config = getattr(vllm_config, "compilation_config", None)
    return getattr(compilation_config, "cudagraph_mode", None)


def _in_full_cudagraph_mode(vllm_config=None) -> bool:
    """True when the engine runs whole steps inside FULL cudagraphs.

    FULL graphs capture every op of the step, so the head_dim=512 layers are
    captured by the stock PagedAttention path (the one the no-spec FULL
    profile uses) and must keep the natural LBNHC cache layout. The B1
    FIA-v2 BNSD verify capture stays exclusive to the FULL_DECODE_ONLY lane,
    where prefill runs eagerly and only the uniform k+1-query steps are
    captured.
    """
    from vllm.config import CUDAGraphMode

    return _cudagraph_mode(vllm_config) == CUDAGraphMode.FULL


def maybe_route_512_capture(impl, attn_metadata) -> bool:
    """True if a head_dim=512 non-sliding layer must route to the
    paged-attention graph path during capture.

    FIA TND does not support head_dim=512 (Gemma4 global attention), and during
    MTP capture forward_impl's PA gate never fires because the dummy metadata is
    not DecodeOnly. This covers both the target's verify layers and the
    KV-sharing draft 512 layer, which runs one token per request and reads the
    target's layer-59 cache. A5 (950) is skipped: full_graph_pa faults there and
    its original FIA path works.
    """
    from vllm_ascend.utils import is_950

    return (
        getattr(impl, "head_size", None) == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
        and getattr(impl, "sliding_window", None) is None
        and not is_950()
    )


def _maybe_b1_paged_verify(impl, attn_metadata, num_tokens: int) -> bool:
    """True when a captured 512-dim step can use the FIA-v2 BNSD paged op.

    Capture-time dummy metadata is not labeled SpecDecoding in V2 (the Ascend
    builders never emit that state), so the query shape is the discriminator.
    The target runs the MTP verify with k+1 queries per request; the draft runs
    step 0 with the same verify layout and its remaining steps with one query
    per request. Anything else keeps the PagedAttention path.
    """
    if not maybe_route_512_capture(impl, attn_metadata):
        return False
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None:
        return False
    num_seqs = seq_lens.shape[0]
    if num_seqs <= 0 or num_tokens % num_seqs:
        return False
    query_len = num_tokens // num_seqs
    spec_cfg = getattr(getattr(impl, "vllm_config", None), "speculative_config", None)
    verify_len = spec_cfg.num_speculative_tokens + 1 if spec_cfg is not None else None
    if _EXTRA_CTX.is_draft_model:
        return query_len == 1 or query_len == verify_len
    return verify_len is not None and query_len == verify_len


@dataclass
class AscendB1Metadata(AscendMetadata):
    # [B, 1, k+1, block_table_width * block_size] bool verify mask for the
    # 512-dim MTP verify capture (True = masked out). Only the 512-dim group
    # builds it, and only for uniform k+1-query steps; the graph-update
    # provider rebinds it every replay.
    paged_verify_mask: torch.Tensor | None = None
    # FULL cudagraph mode: the 512-dim group is captured by full_graph_pa,
    # so the draft graph update must rebind PagedAttention kwargs instead of
    # the FIA-v2 ones. Set on every runtime build of that group.
    b1_pa_graph: bool = False
    # FULL cudagraph mode: block_tables/seq_lens above were re-laid as one
    # row per query token (see _expand_pa_per_token_rows); guards rebuilds.
    b1_pa_rows_expanded: bool = False


def _uniform_query_len(actual_seq_lengths_q: list[int] | None) -> int | None:
    """Per-request query length if every request has the same value."""
    if not actual_seq_lengths_q:
        return None
    prev = 0
    query_len = actual_seq_lengths_q[0]
    for end in actual_seq_lengths_q:
        if end - prev != query_len:
            return None
        prev = end
    return query_len


def _expand_pa_per_token_rows(metadata) -> bool:
    """Re-lay the 512-dim group's PagedAttention metadata as one query per row.

    ``torch_npu._npu_paged_attention`` is decode-only: it splits the query
    tensor evenly over the rows of ``context_lens``, so every row must hold
    exactly one query.  FULL cudagraph steps do not satisfy that -- the MTP
    verify runs k+1 queries per request and prefills run many per request --
    and the op then silently returns wrong values (verified on NPU: a single
    multi-query row matches neither causal nor full-context attention, while
    one-query-per-row stays within bf16 noise of a torch reference for every
    context length tried).

    Splitting each request's query block into one row per token -- the row's
    context length being the request's computed prefix plus its own causal
    position -- restores exact causal attention with the same output row
    order.  ``block_tables`` and ``seq_lens`` are replaced in place because
    they are exactly the two fields full_graph_pa binds and PAParamProvider
    rebinds on every replay.  Zero-query rows (the pad row the runner appends)
    contribute nothing.
    """
    if getattr(metadata, "b1_pa_rows_expanded", False):
        return True
    block_tables = getattr(metadata, "block_tables", None)
    query_lengths = getattr(metadata, "actual_seq_lengths_q", None)
    seq_lens_list = getattr(metadata, "seq_lens_list", None)
    if block_tables is None or not query_lengths or not seq_lens_list:
        return False
    if len(query_lengths) != len(seq_lens_list):
        return False
    query_lens: list[int] = []
    prev = 0
    for end in query_lengths:
        query_lens.append(int(end) - prev)
        prev = int(end)
    num_tokens = int(query_lengths[-1])
    if sum(query_lens) != num_tokens:
        return False
    if all(q == 1 for q in query_lens):
        # Already one query per row; the per-request metadata is exact.
        metadata.b1_pa_rows_expanded = True
        return True
    context_lens: list[int] = []
    for query_len, seq_len in zip(query_lens, seq_lens_list):
        if query_len <= 0:
            continue
        # seq_len counts the computed prefix plus this step's queries; the
        # query at index j therefore attends prefix + j + 1 KV positions.
        prefix = max(int(seq_len) - query_len, 0)
        context_lens.extend(prefix + j + 1 for j in range(query_len))
    if len(context_lens) != num_tokens:
        return False
    counts = torch.tensor(query_lens, dtype=torch.long, device=block_tables.device)
    metadata.block_tables = block_tables.repeat_interleave(counts, dim=0)
    metadata.seq_lens = torch.tensor(context_lens, dtype=torch.int32)
    metadata.b1_pa_rows_expanded = True
    return True


def b1_verify_graph_params(layer_name: str | None, metadata) -> dict[str, Any]:
    """Per-step runtime args for one B1 paged-verify graph task.

    Only the args that change per step are returned; query/KV/workspace/output
    stay bound to the captured buffers. block_table and the mask are re-bound
    to the current step's buffers, mirroring FIAParamProvider/PAParamProvider.
    The caller adds ``layer_name`` when the params travel through SharedSource.

    bind() merges these over the captured kwargs without filtering, so the keys
    must be exactly the ones registered by full_graph_fia_v2_paged_verify:
    ``block_table``, ``actual_seq_kvlen`` and optionally ``atten_mask``.

    In FULL cudagraph mode (b1_pa_graph) the 512-dim group is captured by
    full_graph_pa instead, so the same function feeds the draft graph update
    with PagedAttention kwargs (``block_table``, ``context_lens``).
    """
    if getattr(metadata, "b1_pa_graph", False):
        return {
            "block_table": metadata.block_tables,
            "context_lens": metadata.seq_lens,
        }
    verify_mask = metadata.paged_verify_mask
    query_lengths = metadata.actual_seq_lengths_q
    # AscendMetadata defaults this field to None; guard before indexing.
    query_len = int(query_lengths[0]) if query_lengths else 0
    params: dict[str, Any] = {
        "block_table": metadata.block_tables,
        "actual_seq_kvlen": metadata.seq_lens_list,
    }
    if verify_mask is not None and verify_mask.shape[2] == query_len:
        params["atten_mask"] = verify_mask
    elif query_len > 1:
        raise RuntimeError(
            "B1 paged verify replay requires a matching paged_verify_mask in the runtime metadata of "
            f"layer {layer_name!r} (query_len={query_len}); the 512-dim group builder must attach it "
            "to uniform k+1-query verify steps."
        )
    # A single-query row needs no mask: the capture bound an all-visible
    # placeholder, which stays correct for every step.
    return params


def b1_decode_graph_params(metadata, seq_lens: list[int]) -> dict[str, Any]:
    """Per-step runtime args for one draft 512-dim graph task.

    ``seq_lens`` is the padded per-step KV length list the draft runner
    computes (it applies the step offset for Eagle-style proposers). The
    captured op decides the names: FULL cudagraph mode captures this layer
    through full_graph_pa and needs a device ``context_lens`` tensor, while
    the FULL_DECODE_ONLY B1 capture takes the FIA-v2 ``actual_seq_kvlen`` list.
    """
    if getattr(metadata, "b1_pa_graph", False):
        return {
            "block_table": metadata.block_tables,
            "context_lens": torch.tensor(seq_lens, dtype=torch.int32, device=metadata.seq_lens.device),
        }
    return {
        "block_table": metadata.block_tables,
        "actual_seq_kvlen": seq_lens,
    }


@dataclass(frozen=True, slots=True)
class FIAPagedVerifyParamProvider:
    """Provider for the B1 FIA-v2 BNSD paged capture, keyed by layer name.

    The provider deliberately has no query-length field: the draft registers
    one task per step with the same provider instance, and SharedSource maps
    them back by (provider, registration order). A query-length field would
    split that identity and break the mapping. The query length is instead read
    from the runtime metadata.
    """

    layer_name: str | None

    def resolve(self, attn_metadata) -> dict[str, Any]:
        return b1_verify_graph_params(self.layer_name, attn_metadata[self.layer_name])


class AscendB1AttentionMetadataBuilder(AscendAttentionMetadataBuilder):
    """Metadata builder for the head_dim=512 B1 paged verify layers.

    Attaches the shared verify mask on real (non-capture) builds so the
    graph-update provider can rebind it on every replay. The capture path binds
    an all-visible placeholder of the same shape instead, because the real mask
    needs a seq_lens host-to-device copy that is illegal on the capture stream.
    """

    metadata_cls: type[AscendB1Metadata] = AscendB1Metadata

    def _b1_verify_len(self) -> int | None:
        spec_cfg = self.speculative_config
        if spec_cfg is None:
            return None
        return spec_cfg.num_speculative_tokens + 1

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> AscendB1Metadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build)
        if getattr(self.kv_cache_spec, "head_size", None) != FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE:
            return metadata
        if _in_full_cudagraph_mode(self.vllm_config):
            # FULL cudagraph mode captures this group through full_graph_pa:
            # no verify mask is needed, and the draft graph update must emit
            # PagedAttention kwargs (see b1_verify_graph_params). The decode-only
            # PA op also needs one query per row, so re-lay multi-query steps.
            metadata.b1_pa_graph = True
            _expand_pa_per_token_rows(metadata)
            return metadata
        verify_len = self._b1_verify_len()
        if verify_len is None or metadata.block_tables is None:
            return metadata
        # Only the uniform k+1-query verify layout has a graph to rebind into.
        # Capture-time dummy metadata and draft single-query steps are excluded
        # by the query length; mixed prefill chunks of exactly k+1 tokens are
        # harmless (the same formula is plain causal for those rows).
        if _uniform_query_len(metadata.actual_seq_lengths_q) != verify_len:
            return metadata
        metadata.paged_verify_mask = build_paged_verify_mask(
            metadata.seq_lens,
            metadata.block_tables,
            self.speculative_config.num_speculative_tokens,
            self.kv_cache_spec.block_size,
            self.device,
        )
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs) -> AscendB1Metadata:
        # Deliberately bypass the mask attach above: it needs a seq_lens
        # host-to-device copy that the capture stream forbids. The captured
        # task binds an all-visible placeholder instead and every replay
        # rebinds the runtime mask (or keeps the placeholder for single-query
        # draft steps). ``kwargs`` carries builder-specific capture extras the
        # generic builder does not take (e.g. the V4.1 indexer arguments).
        metadata = AscendAttentionMetadataBuilder.build(
            self,
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        if getattr(self.kv_cache_spec, "head_size", None) == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE and _in_full_cudagraph_mode(
            self.vllm_config
        ):
            # FULL captures this group through full_graph_pa; the decode-only
            # PA op needs the same one-query-per-row layout as at runtime.
            metadata.b1_pa_graph = True
            _expand_pa_per_token_rows(metadata)
        return metadata


class AscendB1AttentionBackendImpl(AscendAttentionBackendImpl):
    """Base GQA implementation plus the B1 FIA-v2 BNSD paged verify capture."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # FULL graphs capture the 512-dim group through the stock
        # PagedAttention path, which needs the natural cache layout; BNSD is
        # only for the own B1 FIA-v2 verify capture.
        self.full_cudagraph = _in_full_cudagraph_mode(getattr(self, "vllm_config", None))
        # Keep this in lockstep with AscendB1AttentionBackend.get_kv_cache_shape:
        # head_dim=512 layers are allocated in BNSD on A2/A3 even when the
        # model-wide layout (and every other layer) stays natural.
        if (
            not self.full_cudagraph
            and not self.use_bnsd_kv_cache
            and self.head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
        ):
            from vllm_ascend.utils import is_950

            self.use_bnsd_kv_cache = not is_950()

    def _b1_route(self, attn_metadata, num_tokens: int) -> bool:
        """True when this captured step must run through the B1 paged op."""
        return (
            bool(_EXTRA_CTX.capturing)
            and not self.full_cudagraph
            and _maybe_b1_paged_verify(self, attn_metadata, num_tokens)
        )

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        if self._b1_route(attn_metadata, query.shape[0]):
            # Dispatch note: MTP capture metadata is not labeled SpecDecoding,
            # so forward_impl's PA gate never routes here; full_graph_pa is
            # called directly and B1 hooks in first.
            return self.full_graph_fia_v2_paged_verify(query, attn_metadata, output)
        if _EXTRA_CTX.capturing and maybe_route_512_capture(self, attn_metadata):
            return self.full_graph_pa(query, attn_metadata, output)
        return super().forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._b1_route(attn_metadata, query.shape[0]):
            return self.full_graph_fia_v2_paged_verify(query, attn_metadata, output)
        return super().forward_paged_attention(query, attn_metadata, output)

    def full_graph_fia_v2_paged_verify(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """B1: capture a 512-dim paged step as FIA-v2 BNSD.

        Op contract: query [B, Hq, qlen, D] BNSD against the paged KV cache
        views [num_blocks, Hkv, block_size, D], per-seq block_table (NO
        per-query expansion), plain per-request actual_seq_qlen/kvlen lists,
        explicit [B, 1, qlen, W] bool mask (True = masked out), sparse_mode=0.
        TND is not an option at head_dim=512 (FIA rejects the layout), which is
        why the default FIA capture paths cannot serve this layer.

        qlen is derived from the captured token count: the target MTP verify
        step runs k+1 queries per request, the KV-sharing draft runs step 0
        with the same verify layout and its later steps with one query per
        request. Sequence lengths must be uniform across requests, which holds
        for every FULL_DECODE_ONLY capture shape.
        """
        num_tokens = query.shape[0]
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        num_reqs = len(actual_seq_lengths_q)
        if num_reqs == 0 or num_tokens % num_reqs:
            raise RuntimeError(
                f"B1 paged verify needs num_tokens divisible by the request count; got "
                f"{num_tokens} tokens for {num_reqs} requests."
            )
        qlen = num_tokens // num_reqs
        if [qlen * (i + 1) for i in range(num_reqs)] != actual_seq_lengths_q:
            raise RuntimeError(f"B1 paged verify requires a uniform query length, got {actual_seq_lengths_q}.")

        if not self.use_bnsd_kv_cache:
            raise RuntimeError(
                "B1 paged verify requires the head_dim=512 layers to be allocated "
                "in BNSD (see AscendB1AttentionBackend.get_kv_cache_shape); got the "
                "natural layout, whose permuted view the FIA-v2 paged op rejects."
            )
        _, _, block_size, _ = self.key_cache.shape  # type: ignore
        key = self.key_cache
        value = self.value_cache
        if not (key.is_contiguous() and value.is_contiguous()):  # type: ignore
            raise RuntimeError(
                "B1 paged verify requires contiguous K/V buffers, but the runner "
                "allocated block-interleaved views."
            )
        # [num_tokens, Hq*D] -> [B, Hq, qlen, D]; contiguous() is a captured op
        # that re-reads the static query buffer on every replay.
        q_bnsd = query.view(num_reqs, qlen, self.num_heads, self.head_size).permute(0, 2, 1, 3).contiguous()
        block_table = attn_metadata.block_tables
        verify_mask = attn_metadata.paged_verify_mask
        if verify_mask is None or verify_mask.shape[2] != qlen:
            # Capture-time dummy metadata carries no mask, and single-query
            # draft steps get one sized for k+1 queries. The real mask needs a
            # seq_lens H2D, illegal on the capture stream: bind an all-visible
            # placeholder of the right shape. Target replays and draft step 0
            # rebind the real mask via the task update; a single query row is
            # always fully causal, so the placeholder stays correct for the
            # other draft steps.
            verify_mask = torch.zeros(
                (num_reqs, 1, qlen, block_table.shape[1] * block_size),
                dtype=torch.bool,
                device=query.device,
            )
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        attn_output = torch.empty(
            num_reqs, self.num_heads, qlen, self.head_size, dtype=query.dtype, device=query.device
        )
        actual_seq_qlen = [qlen] * num_reqs
        actual_seq_kvlen = attn_metadata.seq_lens_list

        # Shared int ws key: B1 must always compete for byte-max; plain-FIA
        # first-capture would undersize it.
        workspace = get_capture_resource(
            _FIA_V2_WORKSPACE_KEY,
            lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=q_bnsd,
                key=key,
                value=value,
                atten_mask=verify_mask,
                block_table=block_table,
                input_layout="BNSD",
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=0,
            ),
            use_max_workspace=True,
        )
        register_task(
            torch_npu.npu_fused_infer_attention_score_v2.out,
            {
                "query": q_bnsd,
                "key": key,
                "value": value,
                "atten_mask": verify_mask,
                "block_table": block_table,
                "input_layout": "BNSD",
                "block_size": block_size,
                "actual_seq_qlen": actual_seq_qlen,
                "actual_seq_kvlen": actual_seq_kvlen,
                "num_key_value_heads": self.num_kv_heads,
                "num_query_heads": self.num_heads,
                "sparse_mode": 0,
                "softmax_scale": self.scale,
                "workspace": workspace,
                "out": [attn_output, softmax_lse],
            },
            FIAPagedVerifyParamProvider(self._layer_name),
        )
        # [B, Hq, qlen, D] -> [num_tokens, Hq*D]; captured copy op, replays
        # from the attn_output buffer the updated FIA task refills every step.
        output.view(num_reqs, qlen, self.num_heads * self.head_size).copy_(
            attn_output.permute(0, 2, 1, 3).reshape(num_reqs, qlen, self.num_heads * self.head_size)
        )
        return output


class AscendB1AttentionBackend(AscendAttentionBackend):
    """Backend for head_dim=512 layers served by the B1 paged verify."""

    @staticmethod
    def get_impl_cls() -> type[AscendB1AttentionBackendImpl]:
        return AscendB1AttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type[AscendAttentionMetadataBuilder]:
        return AscendB1AttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # Gemma4-style head_dim=512 global attention is captured through FIA-v2
        # BNSD paged ops, which need the Hkv-major [blocks, heads, block, dim]
        # cache. The bytes per page are identical to the natural layout, so
        # only this layer's view changes, not the block table. FULL cudagraph
        # mode captures these layers through PagedAttention, whose layout is
        # natural, so the BNSD override must not apply there.
        if head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE:
            from vllm_ascend.utils import is_950

            if not is_950() and not _in_full_cudagraph_mode():
                return (2, num_blocks, num_kv_heads, block_size, head_size)
        if envs_vllm.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND"):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)


__all__ = [
    "AscendB1AttentionBackend",
    "AscendB1AttentionBackendImpl",
    "AscendB1AttentionMetadataBuilder",
    "AscendB1Metadata",
    "FIAPagedVerifyParamProvider",
    "b1_verify_graph_params",
    "build_paged_verify_mask",
]
