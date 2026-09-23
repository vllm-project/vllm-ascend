#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""C8-MXFP (MXFP8) KV cache attention backend (QFA dual-operator interface).

Installed per layer by the ModelSlim C8 quantization method
(quantization/methods/kv_cache/mxfp_c8.py) instead of being resolved through
the backend registry: create_weights assigns ``layer.attn_backend`` and swaps
``layer.impl.__class__``, so importing this module is only required on the
C8 path.
"""

from typing import Any

import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionCGSupport,
    AttentionLayer,
)

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.mxfp_kv_cache import (
    mxfp_k_scale_slot_index,
    scatter_mxfp_k_scale_cache,
    scatter_mxfp_pa_nz_kv_cache,
)
from vllm_ascend.attention.utils import enable_dcp, enable_pcp, notify_kv_cache_written
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    record_attention_compute_start,
)
from vllm_ascend.utils import uses_mooncake_connector


class AscendC8MXFPAttentionBackend(AscendAttentionBackend):
    """Backend for C8-MXFP KV cache layers (QFA dual-operator interface).

    C8-MXFP QFA requires 512-token pages (the D=256 requirement doc allows
    512/1024). This must not be advertised by the generic backend because
    hybrid BF16 models (e.g. Qwen3.5/3.6 linear-attention mixes) use its
    128-token logical block layout when reshaping their KV cache.
    """

    @staticmethod
    def get_impl_cls() -> type["AscendC8MXFPAttentionBackendImpl"]:
        if enable_pcp() or enable_dcp():
            raise NotImplementedError("C8_MXFP attention does not support PCP/DCP yet.")
        return AscendC8MXFPAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendC8MXFPMetadataBuilder"]:
        if enable_pcp() or enable_dcp():
            raise NotImplementedError("C8_MXFP attention does not support PCP/DCP yet.")
        return AscendC8MXFPMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [512]


class AscendC8MXFPMetadataBuilder(AscendAttentionMetadataBuilder):
    """Metadata builder for the C8-MXFP (QFA) backend.

    The generic Ascend builder sizes its block-table width with the
    128-token generic block; this backend uses 512-token kernel blocks.

    Cudagraph support (final form, no .out wrapper variant needed):
    - PIECEWISE: QFA executes outside the compiled region as a plain call.
    - FULL / FULL_DECODE_ONLY: npugraph_ex captures the allocating QFA
      wrapper plus the in-graph metadata op natively (golden-test
      GRAPH_PATH=7 methodology). Replay correctness relies on the model
      runner's persistent length buffers (query_start_loc_gpu /
      seq_lens_gpu), which the impl derives QFA's cu_seqlens/seqused from
      through captured device-side ops, so every replay re-reads the
      current step's lengths.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        mode = vllm_config.compilation_config.cudagraph_mode
        if mode.has_piecewise_cudagraphs() and not mode.has_full_cudagraphs():
            # PIECEWISE: QFA runs outside the compiled region as a plain
            # call (validated on-device). UNIFORM_BATCH is the support
            # level the other piecewise-capable Ascend backends report
            # (e.g. mla_v1); AttentionCGSupport has no PARTIAL member.
            return AttentionCGSupport.UNIFORM_BATCH
        # FULL (incl. FULL_DECODE_ONLY): npugraph_ex captures the
        # allocating QFA wrapper natively (golden-test GRAPH_PATH=7).
        # Spec decode (MTP) included: the draft graph is a plain
        # torch.npu.graph ACLGraphWrapper capture whose metadata goes
        # through the same builder.build() (field passthrough) and whose
        # per-step lengths live in the proposer's persistent buffers,
        # refreshed in place before each draft replay -- the same
        # stable-address contract as the main-model graphs. During draft
        # capture _EXTRA_CTX.capturing is set by the ACLGraphWrapper
        # (shared forward-context object), so the QFA metadata op is
        # still executed inline inside the captured region.
        return AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendC8MXFPAttentionBackend.get_supported_kernel_block_sizes()[0]
        )


QFA_QUANT_MODE_MXFP8 = 1
QFA_MASK_MODE_NO_MASK = 0
QFA_MASK_MODE_CAUSAL = 3
QFA_LAYOUT_TND = "TND"
QFA_LAYOUT_N2TGD = "N2TGD"
QFA_LAYOUT_PA_NZ = "PA_NZ"
# MXFP8 takes the q scale in one of two layouts, and the choice selects the
# kernel: TND (Q_T, Q_N, D/64, 2) compiles the prefill template, N2TGD
# (KV_N, Q_T, G, D/64, 2) the decode one, which merges the S1 and G axes.
# The operator doc puts the boundary at G*Q_S, recommending N2TGD at or below
# this value. Both layouts hold the same scales, so the split is a performance
# choice, not a correctness one.
QFA_QSCALE_N2TGD_MAX_G_TIMES_QS = 80


def _build_qfa_cu_seqlens(cumulative_seq_lengths: list[int], device: torch.device) -> torch.Tensor:
    """Build QFA ``cu_seqlens``: int32 (B+1,) with a leading 0.

    FIA-style ``actual_seq_lengths`` are B cumulative entries; QFA expects the
    cumulative sums prefixed with 0 so batch i spans
    ``[cu_seqlens[i], cu_seqlens[i + 1])`` (QFA requirement doc, sequence
    length conversion rules).
    """
    return torch.tensor([0, *cumulative_seq_lengths], dtype=torch.int32, device=device)


# Resolved lazily and cached: (main_op, metadata_op). The QFA dual operators
# are delivered through the cann_ops_transformer package shipped with the
# CANN toolkit (confirmed final delivery form). That wrapper's call shape
# (verified on-device): positional q/k/v/q_descale/k_descale/v_descale/
# quant_mode, p_scale instead of quant_scale_p, an extra layout_q_descale,
# no pa_block_size, and a required non-null v_descale placeholder on the
# metadata call for quant_mode=1 (batch_size must not be passed with a TND
# layout_q; the op infers it from cu_seqlens_q).
# Graph capture: torch_npu's npugraph_ex backend (the mechanism this vLLM
# build uses for FULL graphs, confirmed in the on-device capture stack)
# captures the ALLOCATING wrapper directly -- internal at::empty allocations
# land in the graph's private pool and replay safely. The ops-transformer
# golden tests exercise exactly this path (GRAPH_PATH=7: torch.compile with
# backend="npugraph_ex", metadata op called INSIDE forward, no .out variant).
# No task_group/update machinery is needed on our side.
_QFA_OPS: tuple[Any, Any] | None = None


def _get_qfa_ops() -> tuple[Any, Any]:
    global _QFA_OPS
    if _QFA_OPS is None:
        try:
            from cann_ops_transformer.ops import quant_flash_attn as main_op  # type: ignore[import-not-found]
            from cann_ops_transformer.ops import (  # type: ignore[import-not-found]
                quant_flash_attn_metadata as metadata_op,
            )
        except ImportError:
            raise RuntimeError(
                "C8_MXFP requires the QFA dual operators delivered in the "
                "cann_ops_transformer package (shipped with the CANN toolkit): "
                "cann_ops_transformer.ops.quant_flash_attn(_metadata) could not "
                "be imported in this environment."
            ) from None
        _QFA_OPS = (main_op, metadata_op)
    return _QFA_OPS


class AscendC8MXFPAttentionBackendImpl(AscendAttentionBackendImpl):
    """MXFP8 KV cache backend computed by the QFA dual-operator interface.

    forward() quantizes Q/K dynamically (``npu_dynamic_mx_quant``, FP8 E4M3 +
    per-token-group E8M0 scales) and V statically (the checkpoint's
    per-channel E8M0 scale), scatters quantized K/V plus their scale caches
    into the paged cache, and calls
    ``cann_ops_transformer.ops.quant_flash_attn_metadata`` +
    ``cann_ops_transformer.ops.quant_flash_attn`` directly on the paged cache.

    Layout: PA_NZ (QFA layout_kv="PA_NZ"). The K/V caches keep the natural
    ``[num_blocks, block_size, num_kv_heads, head_dim]`` storage that
    allocation, hybrid partitioning, PD transfer and prefix-cache CoW see;
    reshape_and_cache writes and QFA reads both go through the NZ 5-D view
    ``(num_blocks, num_kv_heads, head_dim//32, block_size, 32)`` -- the same
    NZ layout ``npu_scatter_pa_kv_cache`` already consumes on the FIA C8
    path, so no layer ever transposes or copies the cache storage. The two
    E8M0 scale caches are allocated directly in the PA_NZ 6-D shapes
    (K: ``[num_blocks, num_kv_heads, block_size//16, head_dim//64, 16, 2]``,
    V: ``[num_blocks, num_kv_heads, head_dim//16, block_size//64, 16, 2]``).

    One QFA call per step: decode and prefill requests share a single
    invocation (cu_seqlens_q over the whole batch) instead of per-subset
    calls -- the causal mask already covers decode rows and the batch is
    smaller to feed. A step whose every query is one row long drops the mask
    (NO_MASK), which is equivalent there and picks a cheaper kernel.
    PrefillNoCache also reads from pages: reshape_and_cache has written this
    step's K/V before attention runs.

    Graph capture: handled natively by torch_npu's npugraph_ex backend (the
    FULL-graph mechanism of this vLLM build), following the ops-transformer
    golden-test methodology (GRAPH_PATH=7). The allocating wrapper is
    captured directly (at::empty outputs land in the graph pool) and the
    AICPU metadata op runs inline inside the graph. Replay safety relies on
    every per-step input being a stable-address tensor whose content is
    refreshed outside Python: block_table / slot_mapping come from the
    model runner's persistent CpuGpuBuffer storages, and cu_seqlens_q /
    seqused_kv are derived IN-GRAPH from the runner's persistent
    query_start_loc / seq_lens buffers (captured clamp/cummax ops re-execute
    each replay). The K-scale scatter parks padded rows on the null block
    (no host sync). No Python-side buffer refresh exists in the captured
    region -- ACL-graph replay never re-runs Python, so such refreshes
    would freeze at capture values. Speculative decoding (MTP) uses the
    same derivation chain: the draft metadata builder routes through
    AscendAttentionMetadataBuilder.build(), and the MTP proposer
    refreshes its persistent query_start_loc/seq_lens/block-table
    buffers in place before each draft-step replay.

    NOTE: the QFA dual operators are called through the main_op/metadata_op
    properties, which resolve cann_ops_transformer.ops.quant_flash_attn
    (_metadata) through the module-level lazy cache -- the confirmed final
    delivery form, shipped with the CANN toolkit. That wrapper's signature
    (verified on-device via inspect + the vendored-QFA bring-up) differs
    from the requirement doc's torch_npu example: positional
    q_descale/k_descale/v_descale/quant_mode, p_scale instead of
    quant_scale_p, an extra layout_q_descale, no pa_block_size, and a
    required v_descale placeholder on the metadata call for quant_mode=1.
    """

    # Installed via ``layer.impl.__class__`` assignment, which does not call
    # this subclass's constructor. Class-level defaults are therefore
    # required for objects that predate the class swap.
    enable_hamming_sparse: bool = False

    # NZ fragment size of the PA_NZ K/V cache view (matches the FIA C8 path's
    # _nz_5d_view and QFA's PA_NZ fp8 layout [Bn, N, D//32, Bs, 32]).
    _KV_NZ_DIM_FRAG = 32

    @property
    def main_op(self) -> Any:
        """The QFA main operator, resolved through the module-level cache.

        A property rather than an __init__ attribute: this impl is installed
        by ``layer.impl.__class__`` assignment, which never calls the
        subclass constructor.
        """
        return _get_qfa_ops()[0]

    @property
    def metadata_op(self) -> Any:
        """The QFA metadata (AICPU planning) operator."""
        return _get_qfa_ops()[1]

    def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -> torch.Tensor:
        """View a natural (num_blocks, block_size, num_kv_heads, head_dim) C8
        MXFP cache tensor in the PA_NZ layout QFA reads:
        (num_blocks, num_kv_heads, head_dim//32, block_size, 32). Head count
        and head dim are derived from the cache itself so models whose V head
        dim differs from the Q/K head dim stay correct."""
        num_kv_heads = cache.shape[2]
        head_dim = cache.shape[3]
        return cache.view(
            -1,
            num_kv_heads,
            head_dim // self._KV_NZ_DIM_FRAG,
            block_size,
            self._KV_NZ_DIM_FRAG,
        )

    def _qfa_step_cache(self, attn_metadata: AscendMetadata) -> dict:
        cache = getattr(attn_metadata, "qfa_metadata_cache", None)
        if cache is None:
            cache = {}
            attn_metadata.qfa_metadata_cache = cache
        return cache

    def _qfa_step_lengths(self, attn_metadata: AscendMetadata, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this step's (cu_seqlens_q, seqused_kv), derived once.

        Both come from the runner's persistent length buffers and depend only
        on per-step data, yet every full-attention layer used to re-derive
        them -- three device ops each, 23 times a step for nothing.

        Caching here stays correct under graph capture, and the reason is
        worth spelling out because the metadata-op cache above does the
        opposite. What capture cannot tolerate is a value produced OUTSIDE
        the captured region, because replay never re-runs Python. These ops
        run inside it: the first layer's derivation is recorded, the other
        layers simply consume the tensor it produced, and every replay
        re-executes that one recorded derivation against the refreshed
        buffers. The metadata op is bypassed instead because its plan is
        consumed by the very call that produced it, not because caching
        across layers would freeze anything.
        """
        cache = self._qfa_step_cache(attn_metadata)
        lengths = cache.get("lengths")
        if lengths is None:
            # Sanitize the tail beyond the current requests: unused
            # query_start_loc slots carry -1 (the FIA padding convention) and
            # may also hold stale entries from larger earlier steps (the FULL
            # dummy-request padding re-copies the whole CPU buffer to GPU).
            # clamp to [0, num_tokens] bounds both; cummax restores
            # monotonicity, turning the tail into zero-length requests whose
            # cu_seqlens_q[-1] still equals the token total. Unused seq_lens
            # slots are zero-filled by the runner every step; clamp(min=1)
            # matches the dummy-request convention (block 0, one token). On
            # clean eager data both ops are identity transforms.
            lengths = (
                attn_metadata.query_start_loc_gpu.clamp(min=0, max=num_tokens).cummax(dim=0).values,
                attn_metadata.seq_lens_gpu.clamp(min=1),
            )
            cache["lengths"] = lengths
        return lengths

    def _qfa_k_scale_slot_index(
        self, attn_metadata: AscendMetadata, slot_mapping: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return this step's K-scale slot decomposition, derived once.

        Same story as _qfa_step_lengths: six device ops over slot_mapping
        that every layer used to repeat, cached inside the captured region so
        replay re-derives them exactly once.
        """
        cache = self._qfa_step_cache(attn_metadata)
        slot_index = cache.get("k_scale_slots")
        if slot_index is None:
            slot_index = mxfp_k_scale_slot_index(slot_mapping, block_size)
            cache["k_scale_slots"] = slot_index
        return slot_index

    @staticmethod
    def _qfa_v_descale_placeholder(value_scale_cache: torch.Tensor) -> torch.Tensor:
        """The v_descale the metadata op insists on, without allocating one.

        quant_mode=1 refuses a null v_descale at the aclnn entry
        (quant_flash_attn_metadata_check.h), but under PA_NZ nothing reads
        it, so a minimal 6-D E8M0 tensor is all it takes. That used to be a
        ``torch.zeros`` inside the step, and an allocation in a captured
        region records its zero-fill as a graph node that every replay runs
        again. The first two bytes of the layer's own V scale cache serve
        just as well: a view launches nothing, the cache exists before any
        capture -- draft layers included, which a lazily built stub would not
        be able to promise -- and its address is the one the main operator
        already reads in the same graph.

        ``view(-1)`` rather than ``flatten()``: a cache that ever stopped
        being contiguous should fail here, not start copying in the step.

        NOTE: torch_npu.float8_e8m0fnu is the integer dtype ID (293) on this
        torch_npu build, not a torch.dtype; tensor.view() would parse it as a
        target shape. Bitcast with the stock torch dtype instead.
        """
        placeholder = value_scale_cache.view(-1)[:2].view(1, 1, 1, 1, 1, 2)
        if placeholder.dtype != torch.float8_e8m0fnu:
            placeholder = placeholder.view(torch.float8_e8m0fnu)
        return placeholder

    def _get_qfa_metadata(
        self,
        attn_metadata: AscendMetadata,
        *,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        value_scale_cache: torch.Tensor,
        max_seqlen_q: int,
        mask_mode: int,
        layout_q_descale: str,
    ):
        """Return the QFA metadata plan (AICPU op output), derived once a step.

        The plan is a load-balance schedule computed on the AICPU, and its
        inputs are exactly this method's arguments plus the device topology:
        head counts, head dim, quant mode, cu_seqlens_q, seqused_kv, mask
        mode, the window and the four layouts. None of them is layer-specific,
        so one plan serves every full-attention layer of a step -- hence the
        key below, which is the full set of non-tensor inputs (the tensor ones
        are per-step by construction).

        Caching it across graph capture is safe for the same reason
        _qfa_step_lengths is: the deriving call runs INSIDE the captured
        region. The first layer's metadata op is recorded ahead of every QFA
        call that reads its output, stream order makes that write-before-read
        on each replay, and the main operator declares ``metadata`` as a plain
        read-only Input -- it never writes back into the plan, so layers
        sharing one cannot interfere. What capture could not tolerate is a
        plan produced OUTSIDE the region, because replay never re-runs Python
        and the tensor would freeze at its capture-time contents.

        This used to bypass the cache while capturing so that every layer
        issued its own metadata op, on the theory that a captured call has to
        consume the plan it just produced. The operator contract does not ask
        for that, and the cost was real: one AICore-to-AICPU round trip per
        full-attention layer per step (23 of them on Qwen3.8-2.4T), each doing
        work that grows with the batch.
        """
        cache = self._qfa_step_cache(attn_metadata)
        plan_key = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            max_seqlen_q,
            mask_mode,
            layout_q_descale,
        )
        metadata = cache.get(plan_key)
        if metadata is None:
            # TND + PA: pass cu_seqlens_q only; the KV side is addressed via
            # block_table + seqused_kv (QFA requirement doc, 3.2.3).
            # batch_size must NOT be passed with a TND layout_q (the checker
            # rejects it); the op infers it from cu_seqlens_q.
            metadata = self.metadata_op(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                QFA_QUANT_MODE_MXFP8,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                v_descale=self._qfa_v_descale_placeholder(value_scale_cache),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=-1,
                mask_mode=mask_mode,
                win_left=-1,
                win_right=-1,
                layout_q=QFA_LAYOUT_TND,
                layout_q_descale=layout_q_descale,
                layout_kv=QFA_LAYOUT_PA_NZ,
                layout_out=QFA_LAYOUT_TND,
            )
            cache[plan_key] = metadata
        return metadata

    def _qfa_int8_mask(self, attn_metadata: AscendMetadata) -> torch.Tensor | None:
        """QFA's attn_mask is INT8/UINT8/bool; the shared builder already
        hands out an int8 2048x2048 causal mask, so only convert when some
        other mask source slips in."""
        if attn_metadata.attn_mask is None:
            return None
        if attn_metadata.attn_mask.dtype == torch.int8:
            return attn_metadata.attn_mask
        return attn_metadata.attn_mask.to(torch.int8)

    def _qfa_query_scale_for_layout(self, query_scale: torch.Tensor, max_seqlen_q: int) -> tuple[torch.Tensor, str]:
        """Return the q scale in the layout that selects the right QFA kernel.

        The scale comes out of npu_dynamic_mx_quant as TND
        ``(Q_T, Q_N, D/64, 2)``, which is what the operator's prefill template
        wants. Its decode template wants the same values as N2TGD
        ``(KV_N, Q_T, G, D/64, 2)``: the query-head axis split into
        ``(KV_N, G)`` and the KV-head half hoisted in front of the token axis,
        so one kv head's whole group is contiguous. Query heads are laid out
        GQA-contiguous (head ``n`` serves kv head ``n // G``), which is what
        makes the split a plain reshape and the rest a permute.

        Which one to send is a throughput choice -- both carry identical
        scales, and a mismatch only costs the wrong kernel, never a wrong
        result -- so it follows the operator doc's G*Q_S boundary. Decode
        lands well inside it (G is 8-16 per rank on Qwen3.8 and Q_S is 1, or
        1+num_spec under MTP) and prefill well outside. The decision reads
        the query shape, not the scheduler state, so MTP verify steps
        (SpecDecoding, 1+spec query tokens) take the decode layout too.
        """
        # Head counts that do not split into whole kv-head groups (possible
        # on MTP draft layers) cannot be reshaped; keep TND instead of
        # producing a miscounted layout. num_kv_heads == 0 would otherwise
        # raise ZeroDivisionError here.
        if self.num_kv_heads == 0 or query_scale.shape[1] % self.num_kv_heads != 0:
            return query_scale, QFA_LAYOUT_TND
        group_size = query_scale.shape[1] // self.num_kv_heads
        if group_size * max_seqlen_q > QFA_QSCALE_N2TGD_MAX_G_TIMES_QS:
            return query_scale, QFA_LAYOUT_TND
        # Permute the byte view: transpose and the copy behind .contiguous()
        # either reject float8 outright or fall back to AICPU, which stalls
        # the device. _run_qfa bitcasts back to E8M0 at the call boundary.
        scale_bytes = query_scale.view(torch.uint8)
        num_tokens = scale_bytes.shape[0]
        n2tgd = (
            scale_bytes.view(num_tokens, self.num_kv_heads, group_size, *scale_bytes.shape[2:])
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        return n2tgd, QFA_LAYOUT_N2TGD

    def _run_qfa(
        self,
        quant_query: torch.Tensor,
        query_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        *,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        qfa_metadata,
        max_seqlen_q: int,
        mask_mode: int,
        layout_q_descale: str,
        num_tokens: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, key_scale, value_scale = kv_cache
        # The K/V caches keep the natural (num_blocks, block_size,
        # num_kv_heads, head_dim) storage; QFA reads them through the PA_NZ
        # 5-D view (same NZ layout npu_scatter_pa_kv_cache wrote them in).
        # The scale caches are already allocated in the PA_NZ 6-D shapes.
        # The scale caches are stored as raw uint8 (index_put_ on float8
        # either errors or falls back to AICPU); QFA's checker wants E8M0, so
        # bitcast at the call boundary (torch.float8_e8m0fnu -- the torch
        # dtype; torch_npu.float8_e8m0fnu is the integer ID 293 on this
        # build and would be parsed as a view *shape*). Same for the q scale
        # when the quant helper returns it as uint8 bytes.
        key = self._nz_5d_view(key, key.shape[1])
        value = self._nz_5d_view(value, value.shape[1])
        if key_scale.dtype != torch.float8_e8m0fnu:
            key_scale = key_scale.view(torch.float8_e8m0fnu)
        if value_scale.dtype != torch.float8_e8m0fnu:
            value_scale = value_scale.view(torch.float8_e8m0fnu)
        if query_scale.dtype != torch.float8_e8m0fnu:
            query_scale = query_scale.view(torch.float8_e8m0fnu)
        main_op = self.main_op
        # cann_ops_transformer delivery signature (verified on-device):
        # q/k/v/q_descale/k_descale/v_descale/quant_mode positional, p_scale
        # instead of quant_scale_p, layout_q_descale, and no pa_block_size
        # (the op infers the block size from the k/v cache shapes).
        # The allocating wrapper is capture-safe under npugraph_ex (internal
        # at::empty allocations land in the graph pool); the ops-transformer
        # golden tests capture exactly this call (GRAPH_PATH=7).
        result = main_op(
            quant_query,
            key,
            value,
            query_scale,
            key_scale,
            value_scale,
            QFA_QUANT_MODE_MXFP8,
            block_table=attn_metadata.block_tables,
            p_scale=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            sinks=None,
            # The doc forbids attn_mask under NO_MASK and requires it under
            # CAUSAL, so the two travel together.
            attn_mask=(None if mask_mode == QFA_MASK_MODE_NO_MASK else self._qfa_int8_mask(attn_metadata)),
            metadata=qfa_metadata,
            softmax_scale=self.scale,
            mask_mode=mask_mode,
            win_left=-1,
            win_right=-1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=-1,
            layout_q=QFA_LAYOUT_TND,
            layout_q_descale=layout_q_descale,
            layout_kv=QFA_LAYOUT_PA_NZ,
            layout_out=QFA_LAYOUT_TND,
            return_softmax_lse=False,
        )
        # return_softmax_lse=False yields an empty LSE tensor in the
        # cann_ops flavor; tolerate both tuple and single-tensor returns.
        attn_output = result[0] if isinstance(result, tuple) else result
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _forward_mxfp8_attention(
        self,
        quant_query: torch.Tensor,
        query_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """One QFA call per step for the whole batch (decode + prefill alike).

        The causal mask covers decode rows as well, so the per-subset split of
        the FIA-based design is unnecessary. PrefillNoCache is included:
        reshape_and_cache has already written this step's K/V into the paged
        cache before attention runs (single-call design validated on-device
        by the vendored-QFA bring-up).

        npugraph_ex capture compatibility (golden-test methodology,
        GRAPH_PATH=7): cu_seqlens_q / seqused_kv are derived ON DEVICE from
        the model runner's persistent int32 buffers (query_start_loc.gpu /
        seq_lens), which _prepare_inputs refreshes in place every step.
        The deriving ops are captured together with the QFA calls, so every
        replay re-executes them and the operators always see the current
        step's lengths. This replaces the former Python-side staging
        writes: ACL-graph replay never re-runs Python, so those refreshes
        only executed during capture and froze the buffers at capture
        values (and the pinned staging buffer itself was racy under the
        async scheduler, where the host could overwrite it while the
        previous step's async H2D copy was still in flight).
        """
        if not attn_metadata.causal:
            raise NotImplementedError("C8_MXFP attention does not support non-causal attention yet.")
        if self.sliding_window is not None:
            raise NotImplementedError("C8_MXFP attention does not support sliding window attention yet.")

        # T as QFA sees it: the query rows actually fed to the operator
        # (forward slices query[:num_actual_tokens] before quantizing), so
        # the TND constraint cu_seqlens_q[-1] == T, the view/output row
        # counts and the sanitize clamp bound below all share one source.
        # At graph capture this is the bucket size; eagerly it is the real
        # token count -- both stay clean where actual_seq_lengths_q[-1]
        # may carry stale tail entries (padded batches).
        num_tokens = quant_query.shape[0]
        if num_tokens <= 0:
            return output

        qsl_gpu = attn_metadata.query_start_loc_gpu
        seq_lens_gpu = attn_metadata.seq_lens_gpu
        if qsl_gpu is None or seq_lens_gpu is None:
            raise RuntimeError(
                "C8_MXFP attention requires the GPU-side length sources "
                "(query_start_loc_gpu / seq_lens_gpu) on AscendMetadata."
            )
        cu_seqlens_q, seqused_kv = self._qfa_step_lengths(attn_metadata, num_tokens)
        # The longest single query in the batch -- NOT the batch token total.
        # The metadata op seeds its querySeqSize with this attr and then raises
        # it per request with max(attr, cu_seqlens_q[i+1] - cu_seqlens_q[i]),
        # so a value that is too small is corrected by the op while one that is
        # too large is never walked back: a 256-request decode step used to
        # declare 256 where every query is 1. max_query_len is the same
        # quantity, already computed on the CPU by _prepare_inputs, so this
        # costs neither a device sync nor a host reduction. Under graph capture
        # the attr freezes at the capture value; the op's max() is what makes
        # that safe for every replay.
        max_seqlen_q = attn_metadata.max_query_len or num_tokens

        # Drop the mask entirely when every request contributes a single query
        # row. hasAttenMask is one of the six tiling-key axes
        # (quant_flash_attn_tiling_mxfp8.cpp), so NO_MASK does not merely skip
        # a load -- it selects a kernel template that never reads the mask at
        # all. At Q_S == 1 that is exactly equivalent to CAUSAL: the lone
        # query row sees all of [0, seqused_kv), and the KV range is bounded
        # by seqused_kv rather than by the mask. Like the layout choice below
        # this reads max_query_len -- the same quantity vLLM uses to call a
        # graph uniform-decode -- so a captured decode graph and all of its
        # replays agree on it. MTP verify steps (Q_S = 1 + num_spec) keep
        # CAUSAL; MTP draft steps are Q_S == 1 and do not.
        mask_mode = QFA_MASK_MODE_NO_MASK if max_seqlen_q == 1 else QFA_MASK_MODE_CAUSAL

        # Both operators have to agree on the q scale layout: it is what picks
        # the prefill or the decode kernel, and the metadata plan is computed
        # for that kernel. Under graph capture the layout and the permuted
        # shape are both baked in, which is safe because the decision reads
        # max_query_len -- the same quantity vLLM uses to decide that a graph
        # is a uniform-decode one, so every replay of a captured graph agrees
        # with the capture.
        query_scale, layout_q_descale = self._qfa_query_scale_for_layout(query_scale, max_seqlen_q)

        qfa_metadata = self._get_qfa_metadata(
            attn_metadata,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            value_scale_cache=kv_cache[3],
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            layout_q_descale=layout_q_descale,
        )
        return self._run_qfa(
            quant_query,
            query_scale,
            kv_cache,
            attn_metadata,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            qfa_metadata=qfa_metadata,
            max_seqlen_q=max_seqlen_q,
            mask_mode=mask_mode,
            layout_q_descale=layout_q_descale,
            num_tokens=num_tokens,
            output=output,
        )

    # KV cache writes for C8_MXFP happen in reshape_and_cache(), invoked from forward()
    # when key/value are present. This hook is only reached when attention is split from
    # cache update, e.g. Attention.forward with forward_includes_kv_cache_update=False
    # (unified_kv_cache_update -> do_kv_cache_update). AscendAttentionBackend keeps
    # forward_includes_kv_cache_update=True, so normal inference never calls this.
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        raise NotImplementedError("C8_MXFP KV cache update is only supported via reshape_and_cache in forward().")

    def reshape_and_cache(  # type: ignore[override]
        self,
        quant_key: torch.Tensor,
        quant_value: torch.Tensor,
        key_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
    ) -> None:
        num_actual_tokens = quant_key.shape[0]
        slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens]
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        block_size = key_cache.shape[1]
        # Write the K/V payloads in the PA_NZ layout QFA reads: the scatter
        # declares cache_mode="PA_NZ" and hands the operator the 5-D view,
        # while allocation, hybrid partitioning, PD and CoW keep seeing the
        # natural (num_blocks, block_size, num_kv_heads, head_dim) storage.
        scatter_mxfp_pa_nz_kv_cache(
            quant_key,
            quant_value,
            key_cache,
            value_cache,
            slot_mapping,
            block_size,
        )

        # Only K's scale is per-token. V's is the checkpoint's static
        # per-channel scale, broadcast over its whole cache once by
        # NPUModelRunner._fill_c8_mxfp_v_scale_caches at KV cache setup, so
        # nothing about it belongs on this path.
        scatter_mxfp_k_scale_cache(
            # Byte view: index_put_ on float8 either errors or falls back to
            # AICPU (the cache side is already uint8 raw storage).
            key_scale.view(torch.uint8) if key_scale.dtype != torch.uint8 else key_scale,
            kv_cache[2],
            self._qfa_k_scale_slot_index(attn_metadata, slot_mapping, block_size),
        )
        notify_kv_cache_written()

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for AscendC8MXFPAttentionBackendImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if getattr(self, "enable_hamming_sparse", False):
            raise NotImplementedError("C8_MXFP attention does not support hamming sparse KV compression yet.")
        if self.vllm_config.kv_transfer_config is not None and not uses_mooncake_connector(
            self.vllm_config.kv_transfer_config
        ):
            raise NotImplementedError(
                "C8_MXFP v1 PD disaggregation (kv_transfer) is only supported with "
                "MooncakeConnectorV1, whose block-level transfer registers and moves "
                "every per-layer cache tensor (FP8 K/V plus both E8M0 scale caches) "
                "as raw blocks. Connectors that only move the K/V pair would "
                "silently drop the scale caches."
            )
        if kv_cache is None or len(kv_cache) < 4:
            raise RuntimeError(
                "C8_MXFP attention requires a (k, v, k_scale, v_scale) KV cache "
                f"tuple, got: {type(kv_cache)} with length "
                f"{len(kv_cache) if kv_cache is not None else 0}."
            )

        record_attention_compute_start()

        query_mxfp8, query_scale = torch_npu.npu_dynamic_mx_quant(
            query[: attn_metadata.num_actual_tokens],
            dst_type=torch.float8_e4m3fn,
        )

        # KV-sharing consumer layers reuse another layer's cache; writing
        # their (dummy) K/V would corrupt the shared slots, so only the
        # owner layer quantizes and scatters K/V. key/value may also be None
        # on pure decode paths.
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            key_mxfp8, key_scale = torch_npu.npu_dynamic_mx_quant(
                key[: attn_metadata.num_actual_tokens],
                dst_type=torch.float8_e4m3fn,
            )

            original_value_shape = value.shape
            value = value.view(original_value_shape[0], -1)
            value_mxfp8 = torch_npu.npu_quantize(
                value[: attn_metadata.num_actual_tokens],
                layer.v_cache_scale_float_reciprocal,
                None,
                torch.float8_e4m3fn,
                -1,
                False,
            )
            value_mxfp8 = value_mxfp8.view((attn_metadata.num_actual_tokens, *original_value_shape[1:]))

            self.reshape_and_cache(key_mxfp8, value_mxfp8, key_scale, kv_cache, attn_metadata)

        # PA_NZ: QFA reads the paged cache through the NZ view taken in
        # _run_qfa, so the cache tuple is passed through as-is -- no
        # transpose, no storage copy.
        return self._forward_mxfp8_attention(query_mxfp8, query_scale, kv_cache, attn_metadata, output)
