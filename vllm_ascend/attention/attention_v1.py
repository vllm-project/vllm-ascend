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
import os as _r144_ovl_os
import sys as _r144_ovl_sys

print(
    f"[R144-OVL-LOAD] R144 overlay attention_v1 LOADED pid={_r144_ovl_os.getpid()} file={__file__}",
    file=_r144_ovl_sys.stderr,
    flush=True,
)
# R218ID: child-side source-identity banner (stderr -- EngineCore stdout is
# dropped, only stderr reaches the serve log; probe legs proved this). The
# wrapper gates expensive legs on file+sha matching the frozen candidate.
try:
    _r218_id_stat = _r144_ovl_os.stat(__file__)
    import hashlib as _r218_id_hl

    _r218_id_sha = _r218_id_hl.sha256(open(__file__, "rb").read()).hexdigest()[:16]  # noqa: SIM115 - one-shot identity banner read
    print(
        f"[R218ID] file={__file__} sha={_r218_id_sha} "
        f"mtime={int(_r218_id_stat.st_mtime)} size={_r218_id_stat.st_size} "
        f"pid={_r144_ovl_os.getpid()}",
        file=_r144_ovl_sys.stderr,
        flush=True,
    )
except Exception as _r218_id_e:
    print(f"[R218ID] FAILED {_r218_id_e!r}", file=_r144_ovl_sys.stderr, flush=True)

from dataclasses import dataclass, field  # noqa: E402 - sectioned/lazy import (port layout)

# OSCAR INT2: shared helpers live in their own module (relocated from this
# file's prologue); importing them here keeps the monolith-era call sites in
# the class bodies below working without re-inlining them.
from vllm_ascend.attention.int2.helpers import (  # noqa: F401,E402 - sectioned/lazy import (port layout)
    _R170_EDGEFENCE_MODE,
    _R170_EDGEFENCE_N,
    _R170_MID2_ON,
    _R170_ROWSLOT_ON,
    _oscar_diag_write,
    _r136_p0_dump,
    _r218_diag,
)

_r218_diag("IMPORT file={}".format(__file__))
# R170_ROWSLOT (default OFF): decode-graph ROW-SLOT contract. FULL decode
# graphs bake hk[row] addresses at capture; the runtime rid->slot binding
# (monotonic under R164_SLOTMAP=1) puts request k's ring at slot k != row.
# Proof R170 (r170e all-slot dump): solo_k>=1 prefill wrote slot_k (fresh,
# frozen), every replay read+slid slot0 = previous generation's final window
# => E27 foreign-token corruption. Fix: (a) at capture bake slot=row
# (torch.npu.is_current_stream_capturing in _tt_slot), (b) bind rid->slot via
# the R166 free-list allocator (_r164_bind_slots first branch), (c) before
# each pure-decode replay, _r170_rowslot_migrate copies ring content so the
# rid at row b lives at slot b (evicting live non-decoding owners out of
# row/pad/scratch slots). Requires R161_FENCE in (1,3) so the update stream's
# copies are ordered after eager prefill writes (F1) and the previous replay
# (F3). All changes gated by this single env; OFF => byte-identical legacy
# behavior.
# R170_MIDDUMP2: host-side pre-replay paged-cache dump (the in-graph
# R170_MIDDUMP hook cannot fire at replay — python never re-executes inside
# captured code; trap found 2026-09-22 when h10 produced zero records).
# R170_EDGEFENCE: host-side full device sync at the END of update_graph_params
# (before the caller launches the graph replay). Root evidence 2026-09-22: the
# outlse census on h10/h1_fix shows the FIRST pure-replay decode step reading
# seq_lens/bt-era values for a L0..L12(h1_fix)/L18(h10) layer PREFIX — exact
# empty-attention signature (lse=-9999.0, out=0.0), extent varying per leg =>
# a race (the cross-stream carrier/input fill lands MID-replay), not structure.
# R161_FENCE=3 orders update_stream AFTER the previous replay, but nothing
# orders the replay AFTER this step's fills at the capture->replay edge.
# =1: sync every step (correctness probe; perf-costly). =2: only the first 3
# update calls (the edge itself). OFF => byte-identical legacy behavior.
# R160: install warmup-fix monkeypatch. Runs in the EngineCore process (where
# attention_v1 is imported, before graph capture). Gated by R160_WFIX / R160_WDIAG
# internally (no-op if both off). Patch module sits at overlay root (on sys.path).
try:
    import importlib as _il160

    _r160_wfix = _il160.import_module("r160_warmup_fix")
    _r160_wfix.start()
except Exception as _r160_e:
    print(f"[R160-OVL] warmup_fix start failed: {_r160_e!r}", file=_r144_ovl_sys.stderr, flush=True)

# R161: race-edge fence matrix (diagnostic; default OFF via R161_FENCE=0).
# Locates the missing cross-stream dependency edge for B>1 INT2 graph replay.
try:
    import importlib as _il161
    from typing import Any

    _r161_f: Any = _il161.import_module("r161_fence")
    print(
        f"[R161-OVL] fence loaded label={_r161_f.fence_label()} "
        f"pid={_r144_ovl_os.getpid()} "
        f"PBCARRIER={_r144_ovl_os.environ.get('R159K_PBCARRIER', '?')} "
        f"PBCVERIFY={_r144_ovl_os.environ.get('R161_PBCVERIFY', '?')} "
        f"ENPU={_r144_ovl_os.environ.get('ENPU_ENABLE', '?')}",
        file=_r144_ovl_sys.stderr,
        flush=True,
    )
except Exception as _r161_e:
    print(f"[R161-OVL] fence load failed: {_r161_e!r}", file=_r144_ovl_sys.stderr, flush=True)
    import types as _r161_t

    _r161_f = _r161_t.SimpleNamespace(
        on_update_enter=lambda *a, **k: None,
        on_update_exit=lambda *a, **k: None,
        on_replay_enter=lambda *a, **k: None,
        on_replay_done=lambda *a, **k: None,
        on_cu_done=lambda *a, **k: None,
        on_middle_enter=lambda *a, **k: None,
        fence_label=lambda: "F0-off",
    )

# ---- R164 tier-dump atexit (default OFF via R164_TIERDUMP) ----
if _r144_ovl_os.environ.get("R164_TIERDUMP", "0") == "1":
    import atexit as _r164_atx

    def _r164_dump_tiers():
        try:
            import json as _j164
            import sys as _s164

            _m164 = _s164.modules.get("vllm_ascend.attention.attention_v1")
            _buf = None
            if _m164 is not None:
                _c164 = getattr(_m164, "AscendINT2AttentionBackendImpl", None)
                if _c164 is not None:
                    _buf = getattr(_c164, "_r164_tdbuf", None)
            if _buf is None:
                return
            _cpu = _buf.detach().to("cpu")
            _rows = {}
            for _l in range(_cpu.shape[0]):
                for _b in range(_cpu.shape[1]):
                    _v = _cpu[_l, _b]
                    if float(_v.abs().sum()) == 0.0:
                        continue
                    _rows[f"L{_l}b{_b}"] = [round(float(x), 4) for x in _v.tolist()]
            with open("/tmp/r164_tierdump.jsonl", "a") as _f164:
                _f164.write(_j164.dumps({"pid": _r144_ovl_os.getpid(), "rows": _rows}) + "\n")
        except Exception as _e164:
            try:
                with open("/tmp/r164_tierdump.jsonl", "a") as _f164e:
                    _f164e.write("dump_error " + repr(_e164) + "\n")
            except Exception:
                pass

    _r164_atx.register(_r164_dump_tiers)

# ---- R150 P0 trace helper (file-based; default OFF via R150_TRACE) ----
# R166_REFTRACE: PFQUANT lifecycle reference trace (default OFF). Cheap host-side
# ints only — no device syncs, no allocation. One jsonl per pid; correlate with
# R164_STEPDUMP via rids. Covers: quantize calls, paged-write coverage (slot
# span), CU skip decisions, lazy-CU fires, three-tier prefill/decode boundaries.
# R203_METAFIX (default OFF): per-STEP host mirror of step-constant metadata.
# kprof r202b step_25 (composed stack, pass-2 big step): every layer re-reads
# the SAME query_start_loc device tensor to host — q_start.cpu() at
# three-tier-prefill entry + _qs201.cpu() in the R201_PFNOMASK core = 2
# full-stream-sync D2H per layer (71 x ~1.0-1.2ms host-blocked aten::to/copy_
# pairs per step + the device starvation behind them; ~160ms/step device-idle
# gap class). The tensor is built ONCE per step (init_ascent_attn_metadata
# H2D) and never mutated within the step, so all 36 layer calls read identical
# values. Mirror it once per metadata OBJECT; mismatch (length vs
# seq_lens_list) or any failure falls back to the exact base read.
# R216 (default OFF): in-vivo LSE forensics. The product's 3-tier merge consumes
# l_h (the middle op's softmaxLse OUTPUT) via exp(l_h - m) weighting. R215 proved
# the op's LSE output is racy on real data at the B=1 base path; whether the
# GROUP-M decode path (graph replay, what serving actually runs) feeds the merge
# a correct l_h has NEVER been adjudicated against an independent reference
# (SR_CMP compares graph-vs-eager arms of the SAME op — a shared defect cancels).
# These helpers capture, at the REAL consumption boundary:
#   _r216_mid  — called at the R146 convergence point right after the carrier
#                out/lse are filled (covers BOTH graph-replay and eager calls):
#                the exact op INPUTS (qrot, Kc/Vc packed INT2, bt, per-token
#                affines, askv) + OUTPUTS (out, lse) + row identity.
#   _r216_merge— called right before the tier-merge compute: the four merge
#                operands (l_pr, l_h, o_pr, o_h) as consumed.
# Offline (RAW/r216_adjudicate.py) rebuilds the middle-tier LSE in fp32 from the
# captured INT2 operands and adjudicates. Full dumps are budget-capped; a light
# per-call ledger (lse vectors + identity) runs for every GROUP-M call to expose
# intermittent corruption across a whole leg.
# R203_D2HTRACE (DIAGNOSTIC_ONLY/default OFF): count long device-to-host reads
# (Tensor.cpu/tolist/item over the threshold) with their Python call sites —
# each is a full-stream sync on the eager path (the kprof r202b gap class).
# Used to enumerate REMAINING D2H sites after the R203_METAFIX edits; never
# enabled in A/B legs. Log: /tmp/r203_d2h.log (container /tmp — docker cp).

import logging  # noqa: E402 - sectioned/lazy import (port layout)
import os  # noqa: E402 - sectioned/lazy import (port layout)
from enum import Enum  # noqa: E402 - sectioned/lazy import (port layout)
from typing import Any  # noqa: E402 - sectioned/lazy import (port layout)

import torch  # noqa: E402 - sectioned/lazy import (port layout)
import torch_npu  # noqa: E402 - sectioned/lazy import (port layout)
import vllm.envs as envs_vllm  # noqa: E402 - sectioned/lazy import (port layout)
from vllm.config import (  # noqa: E402 - sectioned/lazy import (port layout)
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (  # noqa: E402 - sectioned/lazy import (port layout)
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.utils.math_utils import cdiv  # noqa: E402 - sectioned/lazy import (port layout)
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype  # noqa: E402 - sectioned/lazy import (port layout)
from vllm.v1.attention.backend import (  # type: ignore  # noqa: E402 - sectioned/lazy import (port layout)
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore  # noqa: E402 - sectioned/lazy import (port layout)
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.ops.pcp import (  # noqa: E402 - sectioned/lazy import (port layout)
    _gather_prefill_cache_inputs,  # type: ignore[import-not-found]  # noqa: E402 - sectioned/lazy import (port layout)
)
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402 - sectioned/lazy import (port layout)
from vllm.v1.kv_cache_interface import (  # noqa: E402 - sectioned/lazy import (port layout)
    AttentionSpec,
    CrossAttentionSpec,
)

from vllm_ascend.ascend_forward_context import _EXTRA_CTX  # noqa: E402 - sectioned/lazy import (port layout)
from vllm_ascend.attention.attention_mask import (  # noqa: E402 - sectioned/lazy import (port layout)
    AttentionMaskBuilder,  # noqa: E402 - sectioned/lazy import (port layout)
)
from vllm_ascend.attention.context_parallel.common_cp import (  # noqa: E402 - sectioned/lazy import (port layout)
    get_pcp_num_replicated_tokens,
    is_pcp_decode_sharding_enabled,
)
from vllm_ascend.attention.utils import (  # noqa: E402 - sectioned/lazy import (port layout)
    AscendCommonAttentionMetadata,
    _select_seq_lens,
    is_dcp_enabled,
    needs_layer_aware_fia_graph_replay,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    uses_pcp_sharded_layout,
    using_paged_attention,
)
from vllm_ascend.compilation.updatable_graph import (  # noqa: E402 - sectioned/lazy import (port layout)
    get_capture_resource,
    register_task,
)
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402 - sectioned/lazy import (port layout)
from vllm_ascend.device.hardware_profile import (  # noqa: E402 - sectioned/lazy import (port layout)
    HardwareCapability,
    get_current_hardware_profile,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (  # noqa: E402 - sectioned/lazy import (port layout)
    attention_transfer_window,  # noqa: E402 - sectioned/lazy import (port layout)
)

logger = logging.getLogger(__name__)


# default max value of sliding window size
SWA_INT_MAX = 2147483647
_FIA_WORKSPACE_KEY = "npu_fused_infer_attention_score.workspace"
_FIA_V2_WORKSPACE_KEY = "npu_fused_infer_attention_score_v2.workspace"
_PA_WORKSPACE_KEY = "npu_paged_attention.workspace"


# R144 P0: module-level dynamic-lifecycle counters (gated by R144_TT_DEBUG=1).
# Bumped by _tt_resolve_slots when the req_id slot map finishes/reuses a slot.
# R136 P0: env-gated (R136_P0_DIAG=1, default OFF) dump of the middle launch
# OPERANDS (shapes + content hashes + key scalars) at BOTH the V1 eager launch
# site (_r109_middle_candidate) and the V1 graph replay updater site
# (_r113_update_oscar_mid), keyed by (mode, layer, step). Purpose: locate the
# FIRST step/layer/FIELD where F_eager and B_graph middle operands diverge —
# BEFORE the idx-8/idx-2 token symptom. Production byte-identical when OFF.
@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # vLLM #60271 added the batch-invariance support check in
        # `init_attn_backend` in both supported vLLM revisions.
        return True

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if is_dcp_enabled():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPImpl

            return AscendAttentionDCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if is_dcp_enabled():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPMetadataBuilder

            return AscendAttentionDCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        # vLLM checks this capability before any instance-level PCP dispatch.
        # The main backend owns both the ordinary GQA implementation and its
        # DCP specialization. Keep derived backends such as 310P and C8 opted
        # out unless they declare support themselves.
        return cls is AscendAttentionBackend

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "int2_per_token_head":
            # OSCAR INT2: packed 4xINT2 per byte divides K/V head_size by 4
            # before the paged-cache shape is derived.
            head_size = head_size // 4
        elif envs_vllm.VLLM_KV_CACHE_LAYOUT in ("LBHNC", "HND"):
            return (2, num_blocks, num_kv_heads, block_size, head_size)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        # C8-MXFP layers carry (k, v, k_scale, v_scale) tuples; generalize to
        # any cache tuple length so scale caches are swapped alongside K/V.
        for src_cache, dst_cache in zip(src_kv_cache, dst_kv_cache):
            dst_cache[dst_indices] = src_cache[src_indices].to(dst_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            for cache in kv_cache:
                cache[dst_indices] = cache[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        return [128]


# R116 quiet mode: gate THIS MODULE's debug prints (the ~25.7K-line/leg noise
# source). Module-level `print` shadow keeps the rest of the process
# untouched; allowlisted prefixes pass through. Default OFF = R115 behavior.
if os.environ.get("R116_QUIET", "0") == "1":
    _r116_builtin_print = print

    def _r116_gate_print(*args, **kwargs):
        try:
            _s = " ".join(str(a) for a in args)
        except Exception:
            _s = ""
        if _s.startswith(("PERFLEG", "PHASE_", "R116-KEY", "[R116-ERR")):
            _r116_builtin_print(*args, **kwargs)

    print = _r116_gate_print


# R114 steady-token hook: runs INSIDE EngineCore (the overlay is what the
# engine imports; the R112/R113 parent-side monkeypatch never reached the
# engine subprocess). One timestamp per decode step = layer-0 attention entry.
class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # Persistent GPU-side length sources, refreshed in place by the model
    # runner every step: query_start_loc_gpu is the 0-prefixed cumulative
    # query boundaries (int32) and seq_lens_gpu the per-request KV lengths
    # (int32). Graph-capturing backends must source cu_seqlens/seqused from
    # these instead of the CPU lists above -- ACL-graph replay re-executes
    # only device work, so Python-side refreshes of hand-managed buffers
    # never run again after capture.
    query_start_loc_gpu: torch.Tensor = None
    seq_lens_gpu: torch.Tensor = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    # Per-step scratch for C8-MXFP (QFA) layers. The builder creates one
    # AscendMetadata per step shared by all attention layers, so the QFA
    # metadata operator output is computed once per step instead of once
    # per layer. Key: layout_q_descale -> QFA metadata tensor, because that
    # layout is what selects the prefill or decode kernel the plan is computed
    # for. Only used on the eager path -- graph capture bypasses the cache so
    # the metadata op executes inside the captured region.
    qfa_metadata_cache: dict = field(default_factory=dict)
    pcp_local_num_input_tokens: int | None = None
    # Resolved by the builder; execution must not infer layout from tensor sizes.
    uses_pcp_sharded_layout: bool = False

    # R144 P0: stable per-request identity (req_id strings, batch_idx->req_id)
    # plumbed from InputBatch.req_ids via AscendCommonAttentionMetadata. None
    # when R144_TT_REQID=0 (b-keyed verbatim path) or non-ascend paths.
    request_ids: list[str] | None = None

    # R144 P0.1: explicit scheduler lifecycle events (consumed by the resolver
    # instead of absence-inference). None when R144_TT_REQID=0.
    finished_req_ids: set[str] | None = None
    preempted_req_ids: set[str] | None = None
    new_req_ids: set[str] | None = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1
    metadata_cls: type[AscendMetadata] = AscendMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.use_pcp = self.pcp_size > 1
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def _split_decodes_and_prefills(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> tuple[int, int, int, int]:
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=not self.use_pcp,
        )

    def _build_backend_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> dict[str, Any]:
        """Extension point for layouts such as DCP.

        The base builder owns common token classification, padding, masks and
        cache metadata. Specialized backends only add their phase-specific
        metadata here.
        """
        return {}

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        pcp_shard_slot_mapping = (
            common_attn_metadata.slot_mapping if uses_pcp_sharded_layout(common_attn_metadata, self.use_pcp) else None
        )
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = self._split_decodes_and_prefills(
            common_attn_metadata
        )

        block_table = common_attn_metadata.block_table_tensor

        # TODO: This will be moved to propose and deleted after fia ops fixed.
        seq_lens = _select_seq_lens(
            common_attn_metadata,
            kv_cache_spec=self.kv_cache_spec,
            speculative_config=self.speculative_config,
            vllm_config=self.vllm_config,
        )
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
        # vLLM main keeps speculative-decode request metadata at the real
        # request extent, so a padded FULL-graph decode query tensor can exceed
        # the last cumulative query length. FIA's TND layout requires that last
        # entry to equal the model input token count, so append the padded tail
        # as a final synthetic request. This only applies to multi-token
        # (speculative) decode where one request spans several query tokens:
        # single-token decode never pads requests beyond the query start loc,
        # and prefill (including the EAGLE drafter's merged prefill) reuses the
        # target's num_input_tokens while its own query is shorter. Only the V2
        # runner needs this: it applies the metadata to the FULL graph at every
        # replay, while V1 bakes actual_seq_lengths_q into the captured graph
        # (and trims the query to its last entry), so padding there only changes
        # the captured graph. The V2 runner reports speculative decode as
        # ChunkedPrefill, so decide from the per-request is_prefilling flags
        # (falling back to the state when they are unavailable).
        if self.vllm_config.use_v2_model_runner:
            is_prefilling = common_attn_metadata.is_prefilling
            if is_prefilling is None:
                decode_like = common_attn_metadata.attn_state in (
                    AscendAttentionState.DecodeOnly,
                    AscendAttentionState.SpecDecoding,
                )
            else:
                decode_like = not bool(is_prefilling.any())
            if (
                decode_like
                and (common_attn_metadata.max_query_len or 0) > 1
                and actual_seq_lengths_q
                and common_attn_metadata.num_input_tokens > actual_seq_lengths_q[-1]
            ):
                actual_seq_lengths_q = actual_seq_lengths_q + [common_attn_metadata.num_input_tokens]
        seq_lens_list = seq_lens.tolist()
        # Sequence-parallel (or cudagraph) padding makes the model runner insert a
        # dummy padding request into query_start_loc to satisfy the FIA TND-layout
        # constraint (sum of q lengths == hidden_states.shape[0]), bumping the
        # q-derived batchSize by one. The query_start_loc buffer is sized
        # `max_num_reqs + 2` to hold it, but the seq_lens and block_table buffers
        # are only `max_num_reqs`, so when the batch is full the padded request
        # overflows and `[:num_reqs_padded]` silently truncates them. FIA then
        # fails (error 561002) checking, in order, the `actualSeqLengthsKv` length
        # and then the block_table row count against batchSize. Pad them to match:
        # the dummy request points at block 0, and its output is harmless because:
        #   (1) read side: the attention output for padding tokens is trimmed by
        #       `hidden_states = hidden_states[:-pad_size, :]` downstream;
        #   (2) write side: reshape_and_cache slices key/value/slot_mapping to
        #       `[:num_actual_tokens]` (unpadded count), so the dummy request
        #       never writes to KV cache.
        # So any valid positive KV length / zero block row is fine. Pad both
        # seq_lens_list and the seq_lens tensor: full_graph_fia_v2 passes the
        # seq_lens tensor (not seq_lens_list) as actual_seq_kvlen during graph
        # capture, and _get_fia_params derives the PrefillCacheHit batch size from
        # seq_lens.shape[0], so the tensor has to carry the dummy request too.
        num_reqs_fia = len(actual_seq_lengths_q)
        if len(seq_lens_list) < num_reqs_fia:
            padding_len = num_reqs_fia - len(seq_lens_list)
            seq_lens_list = seq_lens_list + [1] * padding_len
            seq_lens = torch.cat([seq_lens, seq_lens.new_ones(padding_len)])
        if block_table is not None and block_table.shape[0] < num_reqs_fia:
            block_table = torch.cat(
                [
                    block_table,
                    block_table.new_zeros((num_reqs_fia - block_table.shape[0], block_table.shape[1])),
                ],
                dim=0,
            )

        backend_metadata = self._build_backend_metadata(
            common_attn_metadata,
            block_table=block_table,
            query_lens=query_start_loc_cpu[1:] - query_start_loc_cpu[:-1],
            seq_lens=seq_lens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
        )
        attn_metadata = self.metadata_cls(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_start_loc_gpu=common_attn_metadata.query_start_loc,
            seq_lens_gpu=common_attn_metadata.seq_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens_list,
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=actual_seq_lengths_q,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            **backend_metadata,
            request_ids=getattr(common_attn_metadata, "request_ids", None),
            finished_req_ids=getattr(common_attn_metadata, "finished_req_ids", None),
            preempted_req_ids=getattr(common_attn_metadata, "preempted_req_ids", None),
            new_req_ids=getattr(common_attn_metadata, "new_req_ids", None),
        )
        if pcp_shard_slot_mapping is not None:
            self._finalize_pcp_metadata(attn_metadata, pcp_shard_slot_mapping)
        return attn_metadata

    def _finalize_pcp_metadata(
        self,
        metadata: AscendMetadata,
        expanded_slot_mapping: torch.Tensor,
    ) -> None:
        if expanded_slot_mapping.numel() % self.pcp_size != 0:
            raise RuntimeError(
                "PCP slot mapping size must be divisible by the PCP world size: "
                f"{expanded_slot_mapping.numel()} % {self.pcp_size} != 0."
            )

        local_num_input_tokens = expanded_slot_mapping.numel() // self.pcp_size
        if metadata.num_actual_tokens > local_num_input_tokens:
            raise RuntimeError(
                "PCP actual token count exceeds the rank-local padded token count: "
                f"{metadata.num_actual_tokens} > {local_num_input_tokens}."
            )

        metadata.uses_pcp_sharded_layout = True
        metadata.slot_mapping = expanded_slot_mapping
        metadata.pcp_local_num_input_tokens = local_num_input_tokens
        if metadata.num_prefills > 0:
            metadata.attn_state = AscendAttentionState.ChunkedPrefill

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


@dataclass(frozen=True, slots=True)
class FIAParamProvider:
    layer_name: str | None
    sliding_window: int | None
    is_draft_model: bool = False
    is_non_pa: bool = False

    def resolve(self, attn_metadata) -> dict[str, Any]:
        metadata = attn_metadata[self.layer_name]

        if self.is_draft_model or not self.sliding_window:
            return {
                "actual_seq_lengths": metadata.actual_seq_lengths_q,
                "actual_seq_lengths_kv": metadata.seq_lens_list,
                "block_table": None if self.is_non_pa else metadata.block_tables,
            }
        else:
            return {
                "actual_seq_lengths": metadata.actual_seq_lengths_q,
                "actual_seq_lengths_kv": metadata.seq_lens_list,
            }


@dataclass(frozen=True, slots=True)
class FIAV2ParamProvider:
    layer_name: str | None

    def resolve(self, attn_metadata) -> dict[str, Any]:
        metadata = attn_metadata[self.layer_name]
        return {
            "actual_seq_qlen": metadata.actual_seq_lengths_q,
            "actual_seq_kvlen": metadata.seq_lens_list,
        }


@dataclass(frozen=True, slots=True)
class PAParamProvider:
    layer_name: str | None

    def resolve(self, attn_metadata) -> dict[str, Any]:
        metadata = attn_metadata[self.layer_name]
        return {
            "context_lens": metadata.seq_lens,
            "block_table": metadata.block_tables,
        }


class AscendAttentionBackendImpl(AttentionImpl):
    # R113: class-level defaults so lazily-created instances (the __new__ path in
    # _forward_int2, class-surgery rebinds) also expose the attributes the graph
    # path queries at init (first hit: 'sliding_window' during EAGER=0 init).
    sliding_window = None
    alibi_slopes = None
    logits_soft_cap = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.use_pcp = (
            type(self) is AscendAttentionBackendImpl
            and self.vllm_config.parallel_config.prefill_context_parallel_size > 1
        )
        self.is_pcp_decode_sharded = is_pcp_decode_sharding_enabled(self.vllm_config)
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.kv_cache_layout = envs_vllm.VLLM_KV_CACHE_LAYOUT
        self.use_bnsd_kv_cache = self.kv_cache_layout in ("LBHNC", "HND")

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(kv_cache_dtype, self.vllm_config.model_config)
        self.enable_c8_quant = self.vllm_config.quant_config is not None and getattr(
            self.vllm_config.quant_config, "enable_c8_quant", False
        )
        # Note: Some embedding model weights are in float32 format.
        if self.kv_cache_dtype in [torch.float16, torch.bfloat16, torch.float32]:
            self.enable_c8_quant = False
        elif self.vllm_config.cache_config.cache_dtype == "int2_per_token_head":
            # OSCAR True-INT2 stores its own per-token-head scales in the page
            # layout (prepared by the INT2 backend), so it does not require a
            # recipe-derived quantized checkpoint like the C8 dtypes above.
            self.enable_c8_quant = False
        else:
            from vllm_ascend.utils import is_c8_mxfp_kv_quant

            # MXFP8 is switched on by --kv-cache-dtype itself, so it must
            # not be gated by the recipe-derived C8 flag.
            if not self.enable_c8_quant and not is_c8_mxfp_kv_quant(self.vllm_config):
                raise ValueError(
                    "The current GQA‑related models adopt static quantization. "
                    "KV quantization is configured via `--kv‑cache‑dtype` upon service startup, "
                    "and the corresponding quantized weights are required."
                )
        self._use_layer_aware_fia_graph_replay = needs_layer_aware_fia_graph_replay()
        self._use_max_workspace_for_fia_graph = self._use_layer_aware_fia_graph_replay
        self.sinks = sinks
        self.layerIndex = 0
        # Some mixed-attention models cannot rely on the iteration order of
        # attn_metadata during graph replay. Record the captured layer name only
        # for that path.
        self._layer_name: str | None = None

    def _graph_metadata_layer_name(self, layer: AttentionLayer | None = None) -> str | None:
        layer_name = layer.layer_name if layer is not None else self._layer_name
        # KV-sharing layers replay with the target layer's metadata instead of
        # their own module name, matching vLLM's shared KV-cache ownership.
        return self.kv_sharing_target_layer_name or layer_name

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        raise NotImplementedError("FIA and PA should be use UpdatableGraph.")

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer=None,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        input_layout = "TND"
        attn_mask = attn_metadata.attn_mask
        sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0
        pre_tokens = self.sliding_window or SWA_INT_MAX
        next_tokens = 0 if self.sliding_window else SWA_INT_MAX
        output_view = output[: attn_metadata.num_actual_tokens]

        extra_args = {}
        if self.enable_c8_quant and layer is not None:
            extra_args = {
                "key_antiquant_scale": layer._c8_k_aq_scale_nz_bnsd,
                "value_antiquant_scale": layer._c8_v_aq_scale_nz_bnsd,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
                "inner_precise": 1,
            }

            # change key/value shape
            _, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self._nz_5d_view(self.key_cache, block_size)
            value = self._nz_5d_view(self.value_cache, block_size)

            # TODO: change layerout from BNSD to TND.
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output_view = output_view.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0

        if block_table is None:
            key = key.contiguous()
            value = value.contiguous()

        use_max_workspace = self._use_max_workspace_for_fia_graph
        workspace = get_capture_resource(
            _FIA_WORKSPACE_KEY,
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
                sparse_mode=sparse_mode,
                **extra_args,
            ),
            use_max_workspace,
        )
        register_task(
            torch_npu.npu_fused_infer_attention_score.out,
            {
                "query": query,
                "key": key,
                "value": value,
                "atten_mask": attn_mask,
                "block_table": block_table,
                "input_layout": input_layout,
                "block_size": block_size,
                "actual_seq_lengths": actual_seq_lengths_q,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
                "num_key_value_heads": self.num_kv_heads,
                "num_heads": self.num_heads,
                "pre_tokens": pre_tokens,
                "next_tokens": next_tokens,
                "scale": self.scale,
                "sparse_mode": sparse_mode,
                "workspace": workspace,
                "out": [output_view, softmax_lse],
                **extra_args,
            },
            FIAParamProvider(
                self._layer_name,
                self.sliding_window,
                _EXTRA_CTX.is_draft_model,
                is_non_pa=block_table is None,
            ),
        )
        return output, num_tokens

    def full_graph_fia_v2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        actual_seq_lengths_kv = attn_metadata.seq_lens
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        output_view = output[: attn_metadata.num_actual_tokens]
        if block_table is None:
            key = key.contiguous()
            value = value.contiguous()

        use_max_workspace = self._use_max_workspace_for_fia_graph
        workspace = get_capture_resource(
            _FIA_V2_WORKSPACE_KEY,
            lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=4 if self.sliding_window is not None else 3,
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                learnable_sink=self.sinks,
            ),
            use_max_workspace,
        )
        register_task(
            torch_npu.npu_fused_infer_attention_score_v2.out,
            {
                "query": query,
                "key": key,
                "value": value,
                "atten_mask": attn_metadata.attn_mask,
                "block_table": block_table,
                "input_layout": "TND",
                "block_size": block_size,
                "actual_seq_qlen": actual_seq_lengths_q,
                "actual_seq_kvlen": actual_seq_lengths_kv,
                "num_key_value_heads": self.num_kv_heads,
                "num_query_heads": self.num_heads,
                "sparse_mode": 4 if self.sliding_window is not None else 3,
                "pre_tokens": self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                "next_tokens": 0,
                "softmax_scale": self.scale,
                "learnable_sink": self.sinks,
                "workspace": workspace,
                "out": [output_view, softmax_lse],
            },
            FIAV2ParamProvider(self._layer_name),
        )
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        workspace = get_capture_resource(
            _PA_WORKSPACE_KEY,
            lambda: torch_npu._npu_paged_attention_get_workspace(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
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
                "block_table": attn_metadata.block_tables,
                "context_lens": attn_metadata.seq_lens,
                "out": output,
                "workspace": workspace,
            },
            PAParamProvider(self._layer_name),
        )
        return output

    def _get_kv_cache_view(self, key: torch.Tensor, value: torch.Tensor):
        if not self.use_bnsd_kv_cache:
            num_block, block_size, _, _ = key.shape
            key = key.view(num_block, block_size, -1)
            value = value.view(num_block, block_size, -1)
        else:
            _, _, block_size, _ = key.shape
        return key, value, block_size

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # PrefillNoCache doesn't need key_cache, but other modes do
        # Only initialize/require cache for modes that actually use it
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            # Initialize cache from kv_cache if not already set (for DecodeOnly mode)
            if self.key_cache is None and kv_cache is not None:
                if (
                    isinstance(kv_cache, torch.Tensor)
                    and kv_cache.dim() > 0
                    and kv_cache.shape[0] == 2
                    or isinstance(kv_cache, (list, tuple))
                    and len(kv_cache) >= 2
                ):
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if self.key_cache is None or self.value_cache is None:
                raise RuntimeError(
                    "key_cache or value_cache is None in _get_fia_params for "
                    f"mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                )

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            key, value, block_size = self._get_kv_cache_view(self.key_cache, self.value_cache)
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            key, value, block_size = self._get_kv_cache_view(self.key_cache, self.value_cache)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            key, value, block_size = self._get_kv_cache_view(self.key_cache, self.value_cache)
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            if self.sinks is not None:
                attn_output, num_tokens = self.full_graph_fia_v2(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            else:
                attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
        passed_value = value
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        if block_table is None:
            key = key.contiguous()
            value = value.contiguous()

        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                sparse_mode = 4
            else:
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key.contiguous(),
                value.contiguous(),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=attn_metadata.attn_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            if not attn_metadata.causal:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=0,
                )
            elif self.sliding_window is not None:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    pre_tokens=self.sliding_window,
                    next_tokens=0,
                    sparse_mode=4,
                )
            else:
                # ChunkedPrefill mixing prefill+decode: split into a per-phase
                # FIA call each (A5 only).
                # NOTE: Batch-invariant execution also requires prefill and
                # decode to be processed separately, regardless of the device
                # generation or attention state. Outside batch-invariant mode,
                # preserve the existing A5 behavior for performance optimization.
                if (
                    (
                        envs_vllm.VLLM_BATCH_INVARIANT
                        or (
                            get_current_hardware_profile().supports(HardwareCapability.CHUNKED_PREFILL_PHASE_SPLIT)
                            and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
                        )
                    )
                    and attn_metadata.num_decodes > 0
                    and attn_metadata.num_prefills > 0
                ):
                    return self._forward_fia_chunked_prefill_split(
                        query, key, value, key, passed_value, block_size, block_table, attn_metadata, output
                    )
                attn_output, _ = DeviceOperator.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    head_size=self.head_size,
                    scale=self.scale,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    current_key=key,
                    current_value=passed_value,
                    attn_metadata=attn_metadata,
                    is_prefill_no_cache=attn_metadata.attn_state == AscendAttentionState.PrefillNoCache,
                    sparse_mode=3,
                )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _forward_fia_chunked_prefill_split(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        block_size: int,
        block_table: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """ChunkedPrefill with mixed prefill/decode: run decode and prefill in
        separate FIA calls. split_decodes_and_prefills has reordered the batch
        so decodes occupy [0, num_decode_tokens) and prefills the rest
        """
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        seq_lens_list = attn_metadata.seq_lens_list
        num_tokens = int(actual_seq_qlen[-1])

        # decode part
        if num_decode_tokens > 0:
            decode_out, _ = DeviceOperator.npu_fused_infer_attention_score(
                query=query[:num_decode_tokens],
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table[:num_decodes],
                input_layout="TND",
                block_size=block_size,
                # cumulative offset from 0; leading num_decodes entries used as-is
                actual_seq_lengths=actual_seq_qlen[:num_decodes],
                actual_seq_lengths_kv=seq_lens_list[:num_decodes],
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                head_size=self.head_size,
                scale=self.scale,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                current_key=key,
                current_value=value,
                attn_metadata=attn_metadata,
                is_prefill_no_cache=False,
                sparse_mode=3,
            )
            output[:num_decode_tokens] = decode_out.view(num_decode_tokens, self.num_heads, self.head_size)

        # prefill part
        if attn_metadata.num_prefills > 0:
            # rebase cumulative q offsets to start at 0 for the prefill slice
            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]
            prefill_out, _ = DeviceOperator.npu_fused_infer_attention_score(
                query=query[num_decode_tokens:num_tokens],
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table[num_decodes:],
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=seq_lens_list[num_decodes:],
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                head_size=self.head_size,
                scale=self.scale,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                current_key=key,
                current_value=value,
                attn_metadata=attn_metadata,
                is_prefill_no_cache=False,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            output[num_decode_tokens:num_tokens] = prefill_out.view(n_prefill, self.num_heads, self.head_size)
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        # Pad actual_seq_len with 0 when num_tokens > actual_seq_len in TND layout
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        if query.shape[0] > actual_seq_qlen[-1]:
            actual_seq_qlen = actual_seq_qlen + [0]
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY):
            return

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slot_mapping,
            use_bnsd=self.use_bnsd_kv_cache,
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if self.kv_sharing_target_layer_name is not None:
                # KV-sharing target layers consume another layer's cache.
                # Writing their dummy/current K/V would overwrite shared slots.
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event.record()
                return query, key, value, output
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            key_to_cache = key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key
            value_to_cache = value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value
            # quick fix to make sure slots is int32 for cross attention case.
            # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
            slots_to_cache = slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32)
            DeviceOperator.reshape_and_cache(
                key=key_to_cache,
                value=value_to_cache,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                slot_mapping=slots_to_cache,
                use_bnsd=self.use_bnsd_kv_cache,
            )
            notify_kv_cache_written()
        return query, key, value, output

    # Keep PCP gathering outside reshape_and_cache: derived implementations
    # such as C8 reuse that method but are outside this PCP feature matrix.
    # Gathered inputs still flow through the canonical cache writer below.
    def _reshape_and_cache_pcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) <= 1:
            return query, key, value, output

        expanded_slot_mapping = attn_metadata.slot_mapping
        local_num_input_tokens = attn_metadata.pcp_local_num_input_tokens
        assert local_num_input_tokens is not None, "PCP GQA metadata must be finalized before execution."
        if key.shape[0] < local_num_input_tokens:
            raise RuntimeError(
                f"PCP GQA input is shorter than the rank-local padded batch: {key.shape[0]} < {local_num_input_tokens}."
            )

        (cache_key, cache_value), cache_slot_mapping = _gather_prefill_cache_inputs(
            (
                key[:local_num_input_tokens],
                value[:local_num_input_tokens],
            ),
            expanded_slot_mapping,
            get_pcp_num_replicated_tokens(attn_metadata.num_decode_tokens, self.is_pcp_decode_sharded),
        )
        local_num_actual_tokens = attn_metadata.num_actual_tokens
        try:
            attn_metadata.slot_mapping = cache_slot_mapping
            attn_metadata.num_actual_tokens = cache_key.shape[0]
            self.reshape_and_cache(query, cache_key, cache_value, kv_cache, attn_metadata, output)
        finally:
            attn_metadata.slot_mapping = expanded_slot_mapping
            attn_metadata.num_actual_tokens = local_num_actual_tokens

        return query, cache_key, cache_value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        with attention_transfer_window():
            num_tokens = query.shape[0]

            if (
                attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and self.sliding_window is None
                and using_paged_attention(num_tokens, self.vllm_config, self.head_size)
            ):
                output = self.forward_paged_attention(query, attn_metadata, output)
            else:
                output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

            return output

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
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        self._layer_name = layer.layer_name

        # INT2 KV cache dispatch: if cache is uint8 (packed INT2), use INT2 forward
        if kv_cache is not None and len(kv_cache) >= 2:
            k_cache = kv_cache[0]
            if not getattr(type(self), "_r190_dtype_logged", False):
                type(self)._r190_dtype_logged = True
                _oscar_diag_write(
                    "r190_probe.log", f"[R190-DTYPE] k_cache.dtype={k_cache.dtype} kv_cache_len={len(kv_cache)}\n"
                )
            if isinstance(k_cache, torch.Tensor) and k_cache.dtype == torch.uint8:
                # R189 PROBE: run a small MSD_ON LSE before this layer's attention to check cube state
                if os.environ.get("R189_PROBE") == "1":
                    _oscar_diag_write("r190_probe.log", "[R189-ENTRY] forward INT2 path entered\n")
                if os.environ.get("R189_PROBE") == "1" and not getattr(type(self), "_r189_done", False):
                    type(self)._r189_done = True
                    try:
                        if not hasattr(type(self), "_r189_mod"):
                            import torch.utils.cpp_extension as _ce189

                            _R152 = os.environ.get("R152_LAUNCHER_DIR", "")
                            _TN = os.path.dirname(torch_npu.__file__)
                            _INC = [
                                os.path.join(_TN, "include"),
                                os.path.join(_TN, "include", "third_party", "op-plugin"),
                                "/usr/local/Ascend/cann-9.0.0/aarch64-linux/include",
                                "/usr/local/Ascend/cann-9.0.0/include",
                            ]
                            _VENDOR = os.environ.get("R183_COMPOSITE_OPP", "")
                            _INC.append(_VENDOR + "/op_api/include")
                            _LD = [
                                "-L/usr/local/Ascend/cann-9.0.0/lib64",
                                "-lascendcl",
                                "-lopapi",
                                "-L" + os.path.join(_TN, "lib"),
                                "-ltorch_npu",
                                "-L" + _VENDOR + "/op_api/lib",
                                "-lcust_opapi",
                                "-Wl,-rpath,/usr/local/Ascend/cann-9.0.0/lib64",
                                "-Wl,-rpath," + _VENDOR + "/op_api/lib",
                            ]
                            os.environ["TORCH_EXTENSIONS_DIR"] = f"/tmp/r189_{os.getpid()}"
                            os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)
                            type(self)._r189_mod = _ce189.load(
                                name="r189",
                                sources=[_R152 + "/int2_op_launch_lse_fast_r152.cpp"],
                                extra_include_paths=_INC,
                                extra_ldflags=_LD,
                                build_directory=os.environ["TORCH_EXTENSIONS_DIR"],
                                verbose=False,
                            )
                        _dev = query.device
                        _D, _nq, _nkv, _bs = 128, 32, 8, 128
                        _q = (torch.randn(1, _nq, 1, _D, device=_dev) * 0.5).bfloat16()
                        _k = torch.randint(0, 256, (4, _nkv, _bs, _D // 4), device=_dev, dtype=torch.uint8).view(
                            torch.int8
                        )
                        _v = torch.randint(0, 256, (4, _nkv, _bs, _D // 4), device=_dev, dtype=torch.uint8).view(
                            torch.int8
                        )
                        _bt = torch.arange(4, dtype=torch.int32, device=_dev).unsqueeze(0)
                        _ks = (torch.rand(1, _nkv, 4 * _bs, device=_dev) * 0.05 + 0.001).bfloat16()
                        _ko = (torch.randn(1, _nkv, 4 * _bs, device=_dev) * 0.02).bfloat16()
                        _vs = (torch.rand(1, _nkv, 4 * _bs, device=_dev) * 0.05 + 0.001).bfloat16()
                        _vo = (torch.randn(1, _nkv, 4 * _bs, device=_dev) * 0.02).bfloat16()
                        _o = torch.zeros(1, _nq, 1, _D, dtype=torch.bfloat16, device=_dev)
                        _l = torch.zeros(1, _nq, 1, dtype=torch.float32, device=_dev)
                        _oscar_diag_write("r190_probe.log", "[R189-PROBE] BEFORE-LSE-call\n")
                        _r = type(self)._r189_mod.launch_lse_fast(
                            _q,
                            _k,
                            _v,
                            _bt,
                            _ks,
                            _ko,
                            _vs,
                            _vo,
                            _o,
                            _l,
                            _nq,
                            1.0 / (_D**0.5),
                            _nkv,
                            "BNSD",
                            _bs,
                            1,
                            1,
                            [4 * _bs],
                            torch.zeros(0, dtype=torch.bfloat16, device=_dev),
                            0,
                            0,
                            0,
                            True,
                        )
                        torch.npu.synchronize()
                        _oscar_diag_write(
                            "r190_probe.log",
                            f"[R189-PROBE-L0-BEFORE-ATTN] rc={int(_r[0])} finite={torch.isfinite(_o).all().item()}\n",
                        )
                    except Exception as _e:
                        _oscar_diag_write("r190_probe.log", f"[R189-PROBE-L0-BEFORE-ATTN] CRASH: {str(_e)[:80]}\n")
                # Dispatch to INT2 forward method
                return self._forward_int2(layer, query, key, value, kv_cache, attn_metadata, output)

        # release-line kvcomp/hamming-sparse feature: main has neither the
        # feature module nor the attribute, so this branch is not ported.

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Initialize key_cache and value_cache from kv_cache if not already set.
        # This is needed for DecodeOnly mode where key/value are None but we still
        # need access to the cache for attention computation.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            # Only PCP-context metadata expands the slot mapping to every
            # rank's tokens; a replicated draft decode writes its local slots.
            if attn_metadata.uses_pcp_sharded_layout:
                assert attn_metadata.pcp_local_num_input_tokens is not None, "PCP metadata must be finalized."
                query, key, value, output_padded = self._reshape_and_cache_pcp(
                    query, key, value, kv_cache, attn_metadata, output
                )
            else:
                query, key, value, output_padded = self.reshape_and_cache(
                    query, key, value, kv_cache, attn_metadata, output
                )
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _forward_int2(
        self,
        layer: "AttentionLayer",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: "AscendMetadata",
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with INT2 KV cache (packed uint8).

        This method handles INT2 KV cache when the cache dtype is uint8.
        It quantizes K/V to INT2 before writing to cache, and dequantizes
        before attention computation.
        """
        from vllm_ascend.attention.attention_int2 import AscendINT2AttentionBackendImpl

        # Lazy initialization of INT2 backend instance
        if not hasattr(self, "_int2_backend"):
            self._int2_backend = AscendINT2AttentionBackendImpl.__new__(AscendINT2AttentionBackendImpl)
            # Copy necessary attributes from self
            self._int2_backend.num_heads = self.num_heads
            self._int2_backend.head_size = self.head_size
            self._int2_backend.scale = self.scale
            self._int2_backend.num_kv_heads = self.num_kv_heads
            self._int2_backend.vllm_config = self.vllm_config
            self._int2_backend.device = query.device  # Use query device instead of self.device
            self._int2_backend.kv_cache_dtype = self.kv_cache_dtype
            # R113: also copy the standard attrs queried on the graph init path
            # (class-level defaults exist as backstop; copy real values when present).
            for _a in (
                "sliding_window",
                "alibi_slopes",
                "attn_type",
                "logits_soft_cap",
                "hidden_size",
                "num_heads",
                "head_size",
            ):
                if getattr(self, _a, None) is not None:
                    setattr(self._int2_backend, _a, getattr(self, _a))
            self._int2_backend.key_cache = None
            self._int2_backend.value_cache = None
            # Set KV transfer attributes
            self._int2_backend.is_kv_producer = getattr(self, "is_kv_producer", False)
            self._int2_backend.is_kv_consumer = getattr(self, "is_kv_consumer", False)
            # Populate Oscar rotation matrices (this __new__ bypassed __init__).
            self._int2_backend._oscar_R_k = None
            self._int2_backend._oscar_R_v = None
            self._int2_backend._oscar_rot_loaded = False
            self._int2_backend._load_oscar_rotations()

        # Delegate to INT2 backend's forward method
        return self._int2_backend.forward(layer, query, key, value, kv_cache, attn_metadata, output)


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

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
        self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._prepare_c8_scales(layer, query.device)
        float_key, float_value = None, None
        if self.vllm_config.kv_transfer_config is None:
            if key is not None and value is not None:
                if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                    float_key, float_value = key, value
                key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
            # pooling model branch
            if attn_metadata.model_runner_type == "pooling":
                attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output

            # When `modelrunnerv2` compiles the graph, the value of `attn_metadata.attn_state` is `None`;
            # therefore, the graph-mode condition needs to be evaluated earlier.
            if _EXTRA_CTX.capturing:
                attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                return self._forward_c8_decode(query, attn_metadata, output, layer)
            elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
            else:
                return self._forward_c8_fused_infer_attention(
                    query,
                    float_key if float_key is not None else key,
                    float_value if float_value is not None else value,
                    attn_metadata,
                    output,
                    layer,
                )
        else:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly and self.is_kv_producer:
                output_padded = None
                if key is not None and value is not None:
                    output_padded = output
                    query, key, value, output_padded = self.reshape_and_cache(
                        query, key, value, kv_cache, attn_metadata, output
                    )
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if output_padded is not None:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
                else:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            elif not self.is_kv_producer:
                if key is not None and value is not None:
                    key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                    query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    return self._forward_c8_decode(query, attn_metadata, output, layer)

    def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -> torch.Tensor:
        """View a KV cache tensor in NZ 5D layout: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)."""
        NZ_FMT_LAST_DIM = 32
        return cache.view(-1, self.num_kv_heads, self.head_size // NZ_FMT_LAST_DIM, block_size, NZ_FMT_LAST_DIM)

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        layer._c8_k_inv_scale = 1.0 / layer._c8_k_scale
        layer._c8_v_inv_scale = 1.0 / layer._c8_v_scale

        nz_bnsd = (self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale_nz_bnsd = layer._c8_k_scale.view(nz_bnsd).contiguous()
        layer._c8_v_aq_scale_nz_bnsd = layer._c8_v_scale.view(nz_bnsd).contiguous()

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize."""
        batch_size = block_table.shape[0]
        max_blocks_per_seq = block_table.shape[1]

        # NZ 5D view: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)
        block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        key_nz = self._nz_5d_view(key, block_size)
        value_nz = self._nz_5d_view(value, block_size)

        # Gather: (batch*max_blocks, H, D//nz, S, nz)
        gathered_k = key_nz[flat_ids]
        gathered_v = value_nz[flat_ids]
        # NZ→ND conversion: permute (S, H, D//nz, nz) → reshape (S, H, D)
        gathered_k = (
            gathered_k.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )
        gathered_v = (
            gathered_v.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, self.num_kv_heads, self.head_size)[valid_mask]
        dense_v = gathered_v.view(-1, self.num_kv_heads, self.head_size)[valid_mask]

        # Scale-only dequant for NZ (symmetric)
        dense_k = dense_k.to(target_dtype) * layer._c8_k_scale
        dense_v = dense_v.to(target_dtype) * layer._c8_v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale + layer._c8_k_offset),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale + layer._c8_v_offset),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        batch_size = len(attn_metadata.seq_lens_list)

        key = self._nz_5d_view(self.key_cache, block_size)
        value = self._nz_5d_view(self.value_cache, block_size)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
            value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            antiquant_mode=0,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            inner_precise=1,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self._nz_5d_view(self.key_cache, block_size)
            kv_v = self._nz_5d_view(self.value_cache, block_size)

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
                value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                antiquant_mode=0,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                inner_precise=1,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self._nz_5d_view(self.key_cache, blk_size)
                paged_v = self._nz_5d_view(self.value_cache, blk_size)

                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k.contiguous(),
                value=prefill_v.contiguous(),
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - layer._c8_k_offset) * layer._c8_k_scale
                value = (value.to(query.dtype) - layer._c8_v_offset) * layer._c8_v_scale

        if block_table is None:
            key = key.contiguous()
            value = value.contiguous()

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if self.kv_sharing_target_layer_name is not None:
                # C8/NZ cache writes follow the same KV-sharing rule.
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event.record()
                return query, key, value, output
            slots = attn_metadata.slot_mapping

            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER

            # NZ write path: 5D view + npu_scatter_pa_kv_cache
            block_size = self.vllm_config.cache_config.block_size
            k_cache_layer = self._nz_5d_view(self.key_cache, block_size)
            v_cache_layer = self._nz_5d_view(self.value_cache, block_size)

            torch_npu.npu_scatter_pa_kv_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=k_cache_layer,
                value_cache=v_cache_layer,
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots,
            )

            notify_kv_cache_written()
        return query, key, value, output
