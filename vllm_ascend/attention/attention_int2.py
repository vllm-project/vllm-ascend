# SPDX-License-Identifier: Apache-2.0
"""OSCAR INT2 three-tier KV attention backend (extracted module).

Historical provenance: split verbatim out of the frozen RC53 overlay of
``vllm_ascend/attention/attention_v1.py`` (lines 4170..EOF at pin
aec44cdb716a5712f449f794) so the stock BF16 attention path stays upstream-
shaped while the INT2 backend lives in its own, explicitly-imported module.
The base class dispatch (uint8 KV cache -> ``self._forward_int2``) resolves
dynamically, so attention_v1.py does NOT import this module; it is imported
by the quantization method (kv_int2) when OSCAR INT2 is enabled.

Extraction commit: oscar-int2/integration. Behavior contract: identical to
the monolith under the same environment (validated by the RC53 needle /
GEN regression legs re-run against this branch).
"""

# main moved the CP metadata dataclasses from common_cp into attention_cp
# (same main-port adaptation as int2/ctx.py)
# main-port note: enable_cp was folded into enable_dcp on main; the
# assembly does not call it (dead re-export from the release tree).
# main-port note: release-only kvcomp_attn / kvcomp_utils /
# flashcomm2_oshard_manager imports dropped (same adaptation as
# int2/ctx.py); none were referenced in this module.
import os
import sys as _r144_ovl_sys
from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import (  # type: ignore
    AttentionLayer,
)

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.oscar.config import get_oscar_config

# ---- shared module state from the base attention module ----
from vllm_ascend.oscar.diag import (  # noqa: F401
    write as _odiag_w,
)

# Semantic configuration (single source of truth; see vllm_ascend/oscar/config.py).
# Product activation is the INT2 KV quantization method; the options below
# steer geometry/executor/numerics once active.
_OSCAR_CFG = get_oscar_config()
from vllm.logger import logger  # noqa: E402 - sectioned/lazy import (port layout)

from vllm_ascend.attention.attention_v1 import (  # noqa: F401,E402 - sectioned/lazy import (port layout)
    SWA_INT_MAX,
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
    AscendMetadata,
)
from vllm_ascend.attention.int2.batchmeta_resolve import (  # noqa: E402 - sectioned/lazy import (port layout)
    Int2BatchMetaResolveMixin,  # noqa: E402 - sectioned/lazy import (port layout)
)
from vllm_ascend.attention.int2.batchmeta_update import (  # noqa: E402 - sectioned/lazy import (port layout)
    Int2BatchMetaUpdateMixin,  # noqa: E402 - sectioned/lazy import (port layout)
)
from vllm_ascend.attention.int2.bridge import Int2BridgeMixin  # noqa: E402 - sectioned/lazy import (port layout)
from vllm_ascend.attention.int2.decode import Int2DecodeMixin  # noqa: E402 - sectioned/lazy import (port layout)
from vllm_ascend.attention.int2.helpers import (  # noqa: F401,E402 - sectioned/lazy import (port layout)
    _R144_TT_DYN,
    _R170_EDGEFENCE_MODE,
    _R170_EDGEFENCE_N,
    _R170_EDGEFENCE_RAW,
    _R170_MID2_ON,
    _R170_ROWSLOT_ON,
    _oscar_dbg,
    _oscar_diag_write,
    _r114_ts_write,
    _r136_p0_dump,
    _r150_tr,
    _r156_shared_askv,
    _r156_shared_asq,
    _r166_tr,
    _R166SlotAlloc,
    _r199_pool_rowsum,
    _r199_sc_mismatch,
    _r203_d2h_install,
    _r203_qsl_host,
    _r216_merge,
    _r216_mid,
    _r218_diag,
    load_oscar_rotations,
)
from vllm_ascend.attention.int2.middle_batched import (  # noqa: E402 - sectioned/lazy import (port layout)
    Int2MiddleBatchedMixin,  # noqa: E402 - sectioned/lazy import (port layout)
)
from vllm_ascend.attention.int2.middle_candidate import (  # noqa: E402 - sectioned/lazy import (port layout)
    Int2MiddleCandidateMixin,  # noqa: E402 - sectioned/lazy import (port layout)
)
from vllm_ascend.attention.int2.middle_exec import (  # noqa: E402 - sectioned/lazy import (port layout)
    Int2MiddleExecMixin,  # noqa: E402 - sectioned/lazy import (port layout)
)
from vllm_ascend.attention.int2.prefill import Int2PrefillMixin  # noqa: E402 - sectioned/lazy import (port layout)
from vllm_ascend.attention.int2.quantcache import (  # noqa: E402 - sectioned/lazy import (port layout)
    Int2QuantCacheMixin,  # noqa: E402 - sectioned/lazy import (port layout)
)
from vllm_ascend.attention.int2.ring import Int2RingMixin  # noqa: E402 - sectioned/lazy import (port layout)
from vllm_ascend.attention.int2.slotstate import Int2SlotStateMixin  # noqa: E402 - sectioned/lazy import (port layout)
from vllm_ascend.attention.int2.staging import Int2StagingMixin  # noqa: E402 - sectioned/lazy import (port layout)

# R116 quiet mode replica (same gate as attention_v1; this module's debug
# prints are silenced unless allowlisted). Kept so log volume matches the
# frozen monolith under R116_QUIET=1.
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


# ---------------- verbatim INT2 backend (see module docstring) ----------------
# (_oscar_dbg comes from int2.helpers; module-local copy removed)
class AscendINT2AttentionBackendImpl(
    Int2QuantCacheMixin,
    Int2BridgeMixin,
    Int2SlotStateMixin,
    Int2BatchMetaResolveMixin,
    Int2BatchMetaUpdateMixin,
    Int2RingMixin,
    Int2StagingMixin,
    Int2MiddleCandidateMixin,
    Int2MiddleBatchedMixin,
    Int2MiddleExecMixin,
    Int2DecodeMixin,
    Int2PrefillMixin,
    AscendAttentionBackendImpl,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_int2_quant = True
        # Oscar rotation matrices. NOTE: this __init__ is NOT always called --
        # the INT2 backend is also instantiated via __new__ in _forward_int2
        # (lazy path) and via class surgery (layer.impl.__class__ rebind). Both
        # bypass __init__. So callers MUST use _load_oscar_rotations() to
        # populate these, and read sites MUST use getattr defensively.
        self._oscar_R_k = None
        self._oscar_R_v = None
        self._oscar_rot_loaded = False

    _INT2_MIN = -2

    _INT2_MAX = 1

    _INT2_PACK = 4  # values per byte

    _r91_mod = None

    _r91_loaded = False

    _r91_getws_count = 0

    _r91_launch_count = 0

    _r91_sync_count = 0

    _r91_output_consumed_count = 0

    _r91_fallback_count = 0

    _r91_entry_count = 0

    _r112_mod = None

    _r112_loaded = False

    _r135_mod = None

    _r135_loaded = False

    _TT_PREFIX = _OSCAR_CFG.prefix_window

    _TT_RECENT = _OSCAR_CFG.recent_window

    _TT_MAX_SEQS = int(os.environ.get("R109_TT_MAX_SEQS", "4"))  # max concurrent sequences with HP state (B<=N)

    _TT_MAX_HP = (
        _TT_PREFIX + _TT_RECENT + (0 if os.environ.get("R199_PRREAD", "0") == "1" else 4)
    )  # HP buffer tokens per seq (derived from configured tiers)

    _tt_ready = False

    _r109_middle_calls = 0

    _r109_middle_ok = 0

    _r109_full_calls = 0  # full-history candidate calls: MUST stay 0

    _r109_merge_count = 0

    def _load_oscar_rotations(self):
        """Idempotently load Oscar R_k/R_v for ALL layers (indexed by layer_idx).

        Loads the full [num_hidden_layers, head_dim, head_dim] buffer so that
        forward() can index by ``layer.layer_idx`` regardless of which layer
        owns this impl. Safe to call on lazily-created / class-surgeried
        instances that never ran __init__.
        """
        if getattr(self, "_oscar_rot_loaded", False):
            return
        self._oscar_rot_loaded = True
        self._oscar_R_k = None
        self._oscar_R_v = None
        # R218: adopt the import-time CLASS-level rotations when present.
        # __init__ / the __new__-lazy path reset INSTANCE attrs to None,
        # which shadow class attrs -- without this adoption the engine would
        # silently quantize unrotated even though import-time load succeeded.
        _cls_k = getattr(type(self), "_oscar_R_k", None)
        _cls_v = getattr(type(self), "_oscar_R_v", None)
        if _cls_k is not None and _cls_v is not None:
            self._oscar_R_k = _cls_k
            self._oscar_R_v = _cls_v
            _r218_diag("LOADER-ADOPTED import-time class rotations cls={}".format(type(self).__name__))
            print(
                "[R218ROT] adopted import-time class rotations cls={}".format(type(self).__name__),
                file=_r144_ovl_sys.stderr,
                flush=True,
            )
            return
        k_rot_path = _OSCAR_CFG.k_rotation_path
        v_rot_path = _OSCAR_CFG.v_rotation_path
        _r218_diag(
            "LOADER-ENTRY cls={} k={!r} v={!r} required={!r}".format(
                type(self).__name__, k_rot_path[:60], v_rot_path[:60], os.environ.get("R218_ROT_REQUIRED", "<unset>")
            )
        )
        print(
            "[R218ROT] load entry k={!r} v={!r} cls={}".format(k_rot_path[:40], v_rot_path[:40], type(self).__name__),
            file=_r144_ovl_sys.stderr,
            flush=True,
        )
        if not (k_rot_path and v_rot_path):
            if os.environ.get("R218_ROT_REQUIRED", "0") == "1":
                raise RuntimeError(
                    "R218_ROT_REQUIRED=1 but SGLANG_OSCAR_{K,V}_ROTATION_PATH unset: "
                    "rotation is part of the OSCAR algorithm (user ruling "
                    "2026-10-17); refusing to quantize unrotated"
                )
            return
        try:
            # NOTE: do NOT use get_current_vllm_config() here -- at model
            # forward time the vllm config context is NOT set, so it returns
            # None and rotation silently fails to load. The impl already holds
            # self.vllm_config (copied in the lazy __new__ path / set at
            # construction); fall back to the global getter only as a last resort.
            vllm_config = getattr(self, "vllm_config", None)
            if vllm_config is None:
                vllm_config = get_current_vllm_config()
            if not (vllm_config and hasattr(vllm_config, "model_config")):
                _r218_diag(
                    "LOADER-SKIP vllm_config unavailable self.vllm_config={!r}".format(
                        getattr(self, "vllm_config", None)
                    )
                )
                print(
                    "[R218ROT] SKIP: vllm_config unavailable (self.vllm_config={!r})".format(
                        getattr(self, "vllm_config", None)
                    ),
                    file=_r144_ovl_sys.stderr,
                    flush=True,
                )
                if os.environ.get("R218_ROT_REQUIRED", "0") == "1":
                    raise RuntimeError(
                        "R218_ROT_REQUIRED=1 but vllm_config unavailable at "
                        "rotation load: refusing to quantize unrotated"
                    )
                return
            model_config = vllm_config.model_config
            num_layers = getattr(model_config, "num_hidden_layers", 36)
            head_dim = getattr(model_config, "head_dim", 128)
            # start_layer=0 -> load every layer; forward indexes by layer.layer_idx
            device = torch.device("npu")
            dtype = torch.bfloat16
            print(
                "[R218ROT] loading ckpt layers={} head_dim={}".format(num_layers, head_dim),
                file=_r144_ovl_sys.stderr,
                flush=True,
            )
            self._oscar_R_k = load_oscar_rotations(k_rot_path, num_layers, 0, head_dim, device, dtype)
            self._oscar_R_v = load_oscar_rotations(v_rot_path, num_layers, 0, head_dim, device, dtype)
            # Publish at class level: every layer owns its own impl instance,
            # and the import-time adoption block above reads type(self) attrs
            # -- without this, each layer re-reads the checkpoint and holds a
            # private copy (36 loads + 36x GPU memory).
            type(self)._oscar_R_k = self._oscar_R_k
            type(self)._oscar_R_v = self._oscar_R_v
            logger.info(
                "Loaded Oscar rotation matrices for INT2 attention: R_k=%s R_v=%s",
                tuple(self._oscar_R_k.shape),
                tuple(self._oscar_R_v.shape),
            )
            _r218_diag(
                "LOADER-LOADED R_k={} R_v={} dev={}".format(
                    tuple(self._oscar_R_k.shape), tuple(self._oscar_R_v.shape), self._oscar_R_k.device
                )
            )
            print(
                "[R218ROT] LOADED R_k={} R_v={} dev={}".format(
                    tuple(self._oscar_R_k.shape), tuple(self._oscar_R_v.shape), self._oscar_R_k.device
                ),
                file=_r144_ovl_sys.stderr,
                flush=True,
            )
        except Exception as e:
            _r218_diag("LOADER-FAILED {!r}".format(e))
            print("[R218ROT] LOAD FAILED: {!r}".format(e), file=_r144_ovl_sys.stderr, flush=True)
            logger.warning("Failed to load Oscar rotation matrices: %r", e)
            self._oscar_R_k = None
            self._oscar_R_v = None
            if os.environ.get("R218_ROT_REQUIRED", "0") == "1":
                raise

    def _forward_int2_decode(self, query, attn_metadata, output, layer):
        """INT2 decode: only dequantize blocks needed for current batch."""
        num_block, block_size, num_kv_heads, packed_head = self.key_cache.shape
        batch_size = len(attn_metadata.seq_lens_list)

        # Get block_table: [batch_size, max_blocks_per_seq]
        block_table = attn_metadata.block_tables
        max_blocks_per_seq = block_table.shape[1]

        # Calculate actual number of blocks needed based on sequence lengths
        # For decode, each sequence has seq_len tokens, which requires ceil(seq_len / block_size) blocks
        actual_seq_kvlen = attn_metadata.seq_lens_list
        num_blocks_needed = [(seq_len + block_size - 1) // block_size for seq_len in actual_seq_kvlen]
        max_blocks_needed = max(num_blocks_needed)

        _oscar_dbg(
            f"[INT2-DECODE] batch_size={batch_size}, max_blocks_per_seq={max_blocks_per_seq}, "
            f"max_blocks_needed={max_blocks_needed}, block_size={block_size}",
            flush=True,
        )
        _oscar_dbg(
            f"[INT2-DECODE] num_kv_heads={num_kv_heads}, packed_head={packed_head}, head_size={self.head_size}",
            flush=True,
        )
        _oscar_dbg(
            f"[INT2-DECODE] actual_seq_kvlen={actual_seq_kvlen}, num_blocks_needed={num_blocks_needed}", flush=True
        )

        # Only use the blocks we actually need
        block_table_trimmed = block_table[:, :max_blocks_needed]

        # Gather only the blocks we need: [batch_size, max_blocks_needed, block_size, num_kv_heads, packed_head]
        key_blocks = self.key_cache[block_table_trimmed]
        value_blocks = self.value_cache[block_table_trimmed]

        _oscar_dbg(
            f"[INT2-DECODE] key_blocks shape={key_blocks.shape}, value_blocks shape={value_blocks.shape}", flush=True
        )

        # Reshape to [batch_size * max_blocks_needed * block_size, num_kv_heads, packed_head]
        total_tokens = batch_size * max_blocks_needed * block_size
        key_blocks_flat = key_blocks.view(total_tokens, num_kv_heads, packed_head)
        value_blocks_flat = value_blocks.view(total_tokens, num_kv_heads, packed_head)

        _oscar_dbg(f"[INT2-DECODE] total_tokens={total_tokens}", flush=True)

        # Gather per-token-head scales/zeros matching the data blocks.
        k_scale_blocks = self.k_scale_cache[block_table_trimmed]
        k_zero_blocks = self.k_zero_cache[block_table_trimmed]
        v_scale_blocks = self.v_scale_cache[block_table_trimmed]
        v_zero_blocks = self.v_zero_cache[block_table_trimmed]

        # Dequantize only the blocks we need (dynamic per-token-head affine)
        key_blocks_bf16 = self._dequantize_packed_int2(
            key_blocks_flat,
            k_scale_blocks.view(total_tokens, num_kv_heads),
            k_zero_blocks.view(total_tokens, num_kv_heads),
            query.dtype,
        )
        value_blocks_bf16 = self._dequantize_packed_int2(
            value_blocks_flat,
            v_scale_blocks.view(total_tokens, num_kv_heads),
            v_zero_blocks.view(total_tokens, num_kv_heads),
            query.dtype,
        )

        _oscar_dbg(
            f"[INT2-DECODE] After dequant: key_blocks_bf16 shape={key_blocks_bf16.shape}, "
            f"dtype={key_blocks_bf16.dtype}",
            flush=True,
        )
        _oscar_dbg(
            f"[INT2-DECODE] After dequant: value_blocks_bf16 shape={value_blocks_bf16.shape}, "
            f"dtype={value_blocks_bf16.dtype}",
            flush=True,
        )

        # Reshape back to [batch_size * max_blocks_needed * block_size, num_kv_heads, head_dim]
        key_cache_tnd = key_blocks_bf16.view(total_tokens, num_kv_heads, self.head_size)
        value_cache_tnd = value_blocks_bf16.view(total_tokens, num_kv_heads, self.head_size)

        _oscar_dbg(
            f"[INT2-DECODE] key_cache_tnd shape={key_cache_tnd.shape}, value_cache_tnd shape={value_cache_tnd.shape}",
            flush=True,
        )

        # Query: [batch_size, num_heads, head_dim] for TND layout
        query_tnd = query[:batch_size]

        _oscar_dbg(f"[INT2-DECODE] query_tnd shape={query_tnd.shape}", flush=True)

        # Get sequence lengths
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        actual_seq_kvlen = attn_metadata.seq_lens_list

        _oscar_dbg(f"[INT2-DECODE] actual_seq_qlen={actual_seq_qlen}, actual_seq_kvlen={actual_seq_kvlen}", flush=True)

        # Only use the first seq_len tokens from gathered blocks (not the entire block)
        # For batch_size=1, we only need the first actual_seq_kvlen[0] tokens
        if batch_size == 1:
            seq_len = actual_seq_kvlen[0]
            key_cache_tnd = key_cache_tnd[:seq_len]
            value_cache_tnd = value_cache_tnd[:seq_len]
        elif batch_size > 1:
            # TND is packed-varlen: request i must start at the cumulative
            # offset of actual_seq_kvlen[:i]. The block gather above is
            # per-request strided (max_blocks_needed * block_size rows per
            # request), so repack to contiguous per-request rows -- otherwise
            # the kernel reads the padded gap rows for every i > 0.
            _stride = max_blocks_needed * block_size
            _kparts, _vparts = [], []
            for _b in range(batch_size):
                _s0 = _b * _stride
                _n = int(actual_seq_kvlen[_b])
                _kparts.append(key_cache_tnd[_s0 : _s0 + _n])
                _vparts.append(value_cache_tnd[_s0 : _s0 + _n])
            key_cache_tnd = torch.cat(_kparts, dim=0)
            value_cache_tnd = torch.cat(_vparts, dim=0)
            _oscar_dbg(
                f"[INT2-DECODE] Repacked varlen rows: total={int(key_cache_tnd.shape[0])}, "
                f"key_cache_tnd shape={key_cache_tnd.shape}",
                flush=True,
            )

        # Check tensor properties (Gemini suggestion)
        _oscar_dbg(
            f"[INT2-DECODE] key_cache_tnd: shape={key_cache_tnd.shape}, stride={key_cache_tnd.stride()}, "
            f"is_contiguous={key_cache_tnd.is_contiguous()}",
            flush=True,
        )
        _oscar_dbg(
            f"[INT2-DECODE] value_cache_tnd: shape={value_cache_tnd.shape}, stride={value_cache_tnd.stride()}, "
            f"is_contiguous={value_cache_tnd.is_contiguous()}",
            flush=True,
        )
        _oscar_dbg(
            f"[INT2-DECODE] query_tnd: shape={query_tnd.shape}, stride={query_tnd.stride()}, "
            f"is_contiguous={query_tnd.is_contiguous()}",
            flush=True,
        )

        # Force contiguous if needed
        if not key_cache_tnd.is_contiguous():
            key_cache_tnd = key_cache_tnd.contiguous()
            _oscar_dbg("[INT2-DECODE] Made key_cache_tnd contiguous", flush=True)
        if not value_cache_tnd.is_contiguous():
            value_cache_tnd = value_cache_tnd.contiguous()
            _oscar_dbg("[INT2-DECODE] Made value_cache_tnd contiguous", flush=True)
        if not query_tnd.is_contiguous():
            query_tnd = query_tnd.contiguous()
            _oscar_dbg("[INT2-DECODE] Made query_tnd contiguous", flush=True)

        _oscar_dbg("[INT2-DECODE] Calling NPU attention WITHOUT block_table (TND layout, T=seq_len)", flush=True)

        # Apply Oscar rotation to Q if rotation matrices are loaded
        if self._oscar_R_k is not None:
            if self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
                R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
                # query_tnd shape: [batch_size, num_heads, head_dim]
                # R_k shape: [head_dim, head_dim]
                # Result: [batch_size, num_heads, head_dim]
                query_tnd = torch.matmul(query_tnd, R_k)
                _r218q = globals().setdefault("_R218_ROT_SEEN", set())
                if "Q" not in _r218q:
                    _r218q.add("Q")
                    _r218_diag("FIRE-Q")
                    print(
                        "[R218ROT-Q] decode Q-rotate FIRES tnd={} lay={}".format(
                            tuple(query_tnd.shape), self._oscar_layer_ord(layer)
                        ),
                        file=_r144_ovl_sys.stderr,
                        flush=True,
                    )
                _oscar_dbg("[INT2-DECODE] Applied R_k rotation to query", flush=True)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query_tnd,
            key=key_cache_tnd,
            value=value_cache_tnd,
            input_layout="TND",
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            scale=self.scale,
            sparse_mode=3,
            atten_mask=attn_metadata.attn_mask,
        )

        _oscar_dbg(f"[INT2-DECODE] attn_output shape={attn_output.shape}", flush=True)

        # Apply inverse Oscar rotation to output if rotation matrices are loaded
        if self._oscar_R_v is not None:
            if self._oscar_layer_ord(layer) < self._oscar_R_v.shape[0]:
                R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]
                # attn_output shape: [batch_size, num_heads, head_dim]
                # R_v shape: [head_dim, head_dim]
                # Inverse rotation: output @ R_v.T
                attn_output = torch.matmul(attn_output, R_v.T)
                _r218o = globals().setdefault("_R218_ROT_SEEN", set())
                if "O" not in _r218o:
                    _r218o.add("O")
                    _r218_diag("FIRE-O")
                    print(
                        "[R218ROT-O] decode output UN-rotate (x R_v.T) FIRES out={} lay={}".format(
                            tuple(attn_output.shape), self._oscar_layer_ord(layer)
                        ),
                        file=_r144_ovl_sys.stderr,
                        flush=True,
                    )
                _oscar_dbg("[INT2-DECODE] Applied R_v.T inverse rotation to output", flush=True)

        output[:batch_size] = attn_output
        return output

    def _forward_int2_prefill(self, query, key_float, value_float, attn_metadata, output, layer):
        """INT2 prefill: use float KV. Applies Oscar rotation, then attn core."""

        _oscar_dbg(f"[INT2-PREFILL] query shape: {query.shape}, dtype: {query.dtype}", flush=True)
        _oscar_dbg(f"[INT2-PREFILL] key_float shape: {key_float.shape}, dtype: {key_float.dtype}", flush=True)
        _oscar_dbg(f"[INT2-PREFILL] value_float shape: {value_float.shape}, dtype: {value_float.dtype}", flush=True)
        _oscar_dbg(
            f"[INT2-PREFILL] num_kv_heads: {self.num_kv_heads}, num_heads: {self.num_heads}, "
            f"head_size: {self.head_size}",
            flush=True,
        )

        query = query.contiguous()
        key_float = key_float.contiguous()
        value_float = value_float.contiguous()

        # Apply Oscar rotation to Q/K/V
        if self._oscar_R_k is not None:
            if self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
                R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
                R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]
                _r218_p1 = globals().setdefault("_R218_FIRES", set())
                if "P1" not in _r218_p1:
                    _r218_p1.add("P1")
                    _p1vals = not torch.npu.is_current_stream_capturing()
                    _q0 = query.clone() if _p1vals else None
                    _k0 = key_float.clone() if _p1vals else None
                    _v0 = value_float.clone() if _p1vals else None
                    query = torch.matmul(query, R_k)
                    key_float = torch.matmul(key_float, R_k)
                    value_float = torch.matmul(value_float, R_v)
                    _r218_diag(
                        "FIRE-P1 prefill rot qd={:.4e} kd={:.4e} vd={:.4e} lay={}".format(
                            float((query - _q0).abs().max()) if _q0 is not None else -1.0,
                            float((key_float - _k0).abs().max()) if _k0 is not None else -1.0,
                            float((value_float - _v0).abs().max()) if _v0 is not None else -1.0,
                            self._oscar_layer_ord(layer),
                        )
                    )
                else:
                    query = torch.matmul(query, R_k)
                    key_float = torch.matmul(key_float, R_k)
                    value_float = torch.matmul(value_float, R_v)
                _oscar_dbg("[INT2-PREFILL] Applied R_k/R_v rotation to q/k/v", flush=True)

        return self._attn_core_prefill(query, key_float, value_float, attn_metadata, output, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with INT2 KV cache quantization."""
        _r150_tr(
            "fwd_entry",
            cls=type(self).__qualname__,
            mod=type(self).__module__,
            file=__file__,
            T=int(query.shape[0]),
            nkvh=int(query.shape[1]) if query.dim() > 1 else -1,
            hd=int(query.shape[-1]),
            nat=int(attn_metadata.num_actual_tokens) if attn_metadata is not None else -1,
            state=str(getattr(attn_metadata, "attn_state", None)),
            kv0=str(kv_cache[0].dtype) if kv_cache else None,
            nkv=len(kv_cache) if kv_cache else 0,
        )
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization not supported for INT2")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Ensure Oscar rotation matrices are loaded. Covers BOTH entry paths:
        # the lazy __new__ instance from _forward_int2 and any class-surgery
        # direct-forward path, neither of which ran __init__. Idempotent.
        self._load_oscar_rotations()
        # R150 P2-side: prefill chunks currently quantize via ~58 ATen ops
        # (_r112_mod only ensured on decode paths). Ensure it at entry so the
        # bit-exact R113 C++ quant serves prefill too. Default OFF.
        if os.environ.get("R150_PREFILL_CPP", "0") == "1" and getattr(type(self), "_r112_mod", None) is None:
            try:
                self._r112_ensure_fast(query.device)
            except Exception as _e150p:
                _r150_tr("prefill_cpp_ensure_failed", err=str(_e150p)[:200])

        self._prepare_int2_scales(layer, query.device)

        # --- OSCAR three-tier init (idempotent) ---
        self._ensure_tt(query.device)
        # Compute rotated KV once -- used by both quant (INT2 middle) and
        # the three-tier HP buffers (BF16 prefix/recent). This avoids
        # double-rotation when the three-tier path is active.
        layer_idx = getattr(layer, "layer_idx", 0)
        # R114 steady hook: FIRST-created attention module == layer 0 (layer
        # objects carry no reliable layer id in this vllm build)
        if not hasattr(type(self), "_r114_l0"):
            type(self)._r114_l0 = id(self)
        if id(self) == type(self)._r114_l0 and os.environ.get("R114_STEADY_HOOK", "0") == "1":
            try:  # noqa: SIM105 - narrow guard, keep plain try/except
                _r114_ts_write()
            except Exception:
                pass
        R_k = R_v = None
        if self._oscar_R_k is not None and self._oscar_layer_ord(layer) < self._oscar_R_k.shape[0]:
            R_k = self._oscar_R_k[self._oscar_layer_ord(layer)]
            R_v = self._oscar_R_v[self._oscar_layer_ord(layer)]

        pre_rotated_kv = None
        key_rot: Any = None
        value_rot: Any = None
        if key is not None and value is not None:
            actual_k = key[: attn_metadata.num_actual_tokens].contiguous()
            actual_v = value[: attn_metadata.num_actual_tokens].contiguous()
            if R_k is not None and os.environ.get("R155_T_NOROT", "0") != "1":
                key_rot = torch.matmul(actual_k, R_k)
                value_rot = torch.matmul(actual_v, R_v)
                _r218w = globals().setdefault("_R218_ROT_SEEN", set())
                if "W" not in _r218w:
                    _r218w.add("W")
                    _r218_diag("FIRE-W")
                    print(
                        "[R218ROT-W] write-rotate FIRES nat={} lay={} krot={}".format(
                            int(attn_metadata.num_actual_tokens), layer_idx, tuple(key_rot.shape)
                        ),
                        file=_r144_ovl_sys.stderr,
                        flush=True,
                    )
            else:
                key_rot = actual_k
                value_rot = actual_v
            pre_rotated_kv = (key_rot, value_rot)
            # R118 trace carriers (default OFF): q/k/v at attention entry and
            # key_rot/value_rot after R_k. Plain copy_ ops — under ACLGraph
            # capture they become graph nodes, so each replay refreshes the
            # carrier with the content the graph actually consumed. Dump is
            # Python-side per step (eager: below at the F1 call; graph: in
            # _r113_update_oscar_mid) — never inside a capture region.
            if os.environ.get("R118_TRACE", "0") == "1":
                try:
                    import importlib as _il118

                    _rt118 = _il118.import_module("r118_trace")
                except ImportError:
                    _rt118 = None
                if _rt118 is not None:
                    _rt118.snap("q", query)
                    _rt118.snap("k", key)
                    _rt118.snap("v", value)
                    _rt118.snap("kr", key_rot)
                    _rt118.snap("vr", value_rot)
                    _rt118.note_step("f1", layer_idx, int(attn_metadata.num_actual_tokens))

        # R114 F1: TRUE device-fused cache update — quant(kth clip)+pack+6-field
        # scatter in ONE AscendC kernel (grid=1, no ATen/index_put_/barrier
        # inside; byte-exact vs R113 path over 1440-case matrix + 100x race
        # stress). Default OFF; eager only (fail-closed for capture: excluded);
        # T<=16 (decode hot path; larger T falls back to the R113 path).
        _r114_fused = (
            (
                os.environ.get("R114_FUSED_CACHE_UPDATE", "0") == "1"
                or os.environ.get("R115_FUSED_CACHE_UPDATE", "0") == "1"
                or os.environ.get("R116_FUSED_CACHE_UPDATE", "0") == "1"
            )
            and pre_rotated_kv is not None
            # R116: F1 may enter ACLGraph capture ONLY by explicit
            # opt-in, after G0-G4 (real valid-tail) pass — the bare
            # pybind path is capture-proven byte-exact standalone.
            and (not _EXTRA_CTX.capturing or os.environ.get("R116_FUSED_CAPTURE", "0") == "1")
            and attn_metadata.num_actual_tokens <= 16
            and len(kv_cache) > 1
        )
        _r150_tr(
            "gate",
            fused=bool(_r114_fused),
            capturing=bool(_EXTRA_CTX.capturing),
            nat16=bool(attn_metadata.num_actual_tokens <= 16),
            prerot=bool(pre_rotated_kv is not None),
            f114=os.environ.get("R114_FUSED_CACHE_UPDATE", "0"),
            f115=os.environ.get("R115_FUSED_CACHE_UPDATE", "0"),
            f116=os.environ.get("R116_FUSED_CACHE_UPDATE", "0"),
            nkv=len(kv_cache) if kv_cache else 0,
        )
        _cueager_skip = os.environ.get("R155_T_NOCU", "0") == "1" or (
            os.environ.get("R155_CUEAGER", "0") == "1"
            and not (
                os.environ.get("R165_PFQUANT", "1") == "1"
                and attn_metadata.attn_state != AscendAttentionState.DecodeOnly
            )
        )
        _envn = globals().setdefault("_R166_ENVLOG", [False])
        if not _envn[0]:
            _envn[0] = True
            _r166_tr(
                "envdump",
                pid_flag_vals={
                    k: os.environ.get(k, "<ABSENT>")
                    for k in (
                        "R166_WVERIFY",
                        "R166_STREAMFIX",
                        "R166_BT_HOST",
                        "R166_SLOTALGO",
                        "R166_REFTRACE",
                        "R165_PFQUANT",
                        "R113_SCATTER_CPP",
                    )
                },
            )
        _r166_tr(
            "cu_gate",
            skip=bool(_cueager_skip),
            st=str(attn_metadata.attn_state),
            T=int(attn_metadata.num_actual_tokens),
            pfq=os.environ.get("R165_PFQUANT", "1"),
            cue=os.environ.get("R155_CUEAGER", "0"),
            cap=bool(_EXTRA_CTX.capturing),
        )
        if _cueager_skip:
            # R155 A2-lite (R155_CUEAGER): skip the cache-update for DECODE
            # steps (the decode-time ring-row CU covers the middle's needs).
            # R165_PFQUANT (default ON): PREFILL/ChunkedPrefill steps DO
            # quantize+write the paged INT2 cache — under the old skip, the
            # middle tier stayed EMPTY until a decode-time CU landed mid-run,
            # activating the middle at a timing-dependent step (rows/legs
            # flipped tokens together at that transition; the "non-canonical
            # rows" family). With prefill-time quantize the middle is
            # populated from step 0, consistently for every row/leg/B.
            # NOTE: this changes the canonical token streams (the old
            # canonical had an empty middle — a semantic hole, not a
            # contract).
            _kc0, _vc0 = kv_cache[0], kv_cache[1]
            if getattr(self, "k_scale_cache", None) is None or self.k_scale_cache.shape[0] != _kc0.shape[0]:
                self.k_scale_cache = None
                self._ensure_int2_scale_buffers(_kc0, _kc0.shape[1], _kc0.shape[2], torch.bfloat16, _kc0.device)
            self.key_cache, self.value_cache = _kc0, _vc0
            key_packed = value_packed = None
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
                attn_metadata.reshape_cache_event.record()
        elif _r114_fused:
            kc, vc = kv_cache[0], kv_cache[1]
            # Both fused sub-paths below read self.key_cache/value_cache at
            # decode time; publish them here so the R114-only route (without
            # R115/R116) leaves them defined instead of AttributeError.
            self.key_cache, self.value_cache = kc, vc
            # R116 layout contract: scale/zero caches are PAGE-PARALLEL to the
            # packed cache — (kc.shape[0], kc.shape[1], kc.shape[2]) — all dims
            # derived from kc itself (never vllm block_size, never a constant).
            # A profile-run DUMMY cache (few pages) may have created stale
            # buffers earlier in the process; realign OUTSIDE capture (the eager
            # warmup pass), fail-closed inside capture.
            _need_scale = (
                getattr(self, "k_scale_cache", None) is None
                or self.k_scale_cache.shape[0] != kc.shape[0]
                or self.k_scale_cache.shape[1] != kc.shape[1]
                or self.k_scale_cache.shape[2] != kc.shape[2]
            )
            if _need_scale and _EXTRA_CTX.capturing:
                raise RuntimeError(
                    f"[R116-GA] scale-cache layout mismatch under capture: "
                    f"ksc={tuple(self.k_scale_cache.shape) if self.k_scale_cache is not None else None} "
                    f"kc={tuple(kc.shape)} (must realign during eager warmup)"
                )
            if _need_scale:
                self.k_scale_cache = None
                self._ensure_int2_scale_buffers(kc, kc.shape[1], kc.shape[2], torch.bfloat16, kc.device)
            if os.environ.get("R116_FUSED_CACHE_UPDATE", os.environ.get("R115_FUSED_CACHE_UPDATE", "0")) == "1":
                # R115: NATIVE registry aclnn path (async enqueue, NO taskQ
                # drain; ACLGraph G0-G4 PASS — R116 REDONE with real
                # valid-tail G3, ALL PASS x2 runs).
                # R116 Phase-4 capture-safety:
                #  (a) FAIL-CLOSED: inside a capture region the binding must
                #      already be loaded (built during eager warmup) — never
                #      JIT inside torch.npu.graph;
                #  (b) R116_TORCHOP=1 routes through the formal
                #      torch.library mutating schema (oscar.int2_cache_update_v2);
                #  (c) slot_mapping must ALREADY be int64 (static address under
                #      capture; .to() no-op asserted).
                if _EXTRA_CTX.capturing:
                    if getattr(type(self), "_r115_aclnn_mod", None) is None:
                        raise RuntimeError(
                            "[R116-GA] F1 under capture requires the binding to be "
                            "preloaded during eager warmup (fail-closed, no capture-time JIT)"
                        )
                    # NOTE: slot_mapping may arrive as int32 in this build; the
                    # .to(torch.long) cast below is an ordinary ATen op — it is
                    # CAPTURED as a graph node and re-executed on every replay,
                    # reading the static int32 buffer and writing a static pool
                    # tensor the kernel then reads. No host value access.
                mod: Any = getattr(type(self), "_r115_aclnn_mod", None)
                if mod is None:
                    from vllm_ascend.oscar import r116_torch_op as _r116op

                    mod = _r116op.preload(query.device)
                    type(self)._r115_aclnn_mod = mod
                    type(self)._r115_cu_calls = 0
                    if os.environ.get("R116_TORCHOP", "0") == "1":
                        type(self)._r116_torchop = _r116op.register()
                if os.environ.get("R168_CUIN", "0") == "1" and not getattr(mod, "_r168_wrapped", False):
                    # R168 P1: record every CU call's T + per-128-row input
                    # K abs-sums (path-agnostic; same model+prompt => same K
                    # values across runs => split-vs-whole input comparison).
                    def _r168_wrap(k, v, slots, kc, vc, ks, kz, vs, vz, _o=mod.cache_update_aclnn):
                        try:
                            import json as _jw168

                            _T = int(k.shape[0])
                            _rows = k.detach().float().abs().reshape(_T, -1).sum(dim=1)
                            _nseg = min(40, -(-_T // 128))
                            _seg = [round(float(_rows[i * 128 : (i + 1) * 128].sum()), 2) for i in range(_nseg)]
                            with open("/tmp/r168_cuin.jsonl", "a") as _fw168:
                                _fw168.write(
                                    _jw168.dumps({"T": _T, "seg": _seg, "sl0": int(slots[0]), "sln": int(slots[-1])})
                                    + "\n"
                                )
                        except Exception:
                            pass
                        return _o(k, v, slots, kc, vc, ks, kz, vs, vz)

                    mod.cache_update_aclnn = _r168_wrap
                    mod._r168_wrapped = True
                _slotmap = attn_metadata.slot_mapping
                _slotmap = _slotmap if _slotmap.dtype == torch.long else _slotmap.to(torch.long)
                if (
                    getattr(type(self), "_r116_torchop", None) is not None
                    and os.environ.get("R116_TORCHOP", "0") == "1"
                ):
                    _nt166b = int(attn_metadata.num_actual_tokens)
                    if os.environ.get("R166_CUFIX", "0") == "1" and key_rot.shape[0] > _nt166b:
                        _kr166b = key_rot[:_nt166b].contiguous()
                        _vr166b = value_rot[:_nt166b].contiguous()
                        _sm166b = _slotmap[:_nt166b].contiguous()
                    else:
                        _kr166b = key_rot.contiguous()
                        _vr166b = value_rot.contiguous()
                        _sm166b = _slotmap
                    type(self)._r116_torchop(
                        _kr166b,
                        _vr166b,
                        _sm166b,
                        kc,
                        vc,
                        self.k_scale_cache,
                        self.k_zero_cache,
                        self.v_scale_cache,
                        self.v_zero_cache,
                    )
                    rc, lr = 0, 0
                else:
                    # R166_CUFIX (default OFF): slice to THIS STEP's tokens.
                    # Root cause (R166 E15/E17): under FULL graph mode the
                    # eager fused CU passed the WHOLE static key_rot/value_rot
                    # buffers (persisting the last PREFILL's T=1047 rows)
                    # with the per-step refreshed slotmap — rewriting the
                    # prefill-era KV onto the CURRENT request's slots at every
                    # decode step (probe: blocks converge to the first
                    # request's content; WVERIFY real -> REVERIFY stomped).
                    _nt166 = int(attn_metadata.num_actual_tokens)
                    if os.environ.get("R166_CUFIX", "0") == "1" and key_rot.shape[0] > _nt166:
                        _kr166 = key_rot[:_nt166].contiguous()
                        _vr166 = value_rot[:_nt166].contiguous()
                        _sm166 = _slotmap[:_nt166].contiguous()
                    else:
                        _kr166 = key_rot.contiguous()
                        _vr166 = value_rot.contiguous()
                        _sm166 = _slotmap
                    rc, _ws, lr, _c, _h, _ge, _re = mod.cache_update_aclnn(
                        _kr166,
                        _vr166,
                        _sm166,
                        kc,
                        vc,
                        self.k_scale_cache,
                        self.k_zero_cache,
                        self.v_scale_cache,
                        self.v_zero_cache,
                    )
                if rc != 0 or lr != 0:
                    raise RuntimeError(
                        f"R115 aclnn cache-update failed rc={rc} lr={lr} shapes: "
                        f"k={tuple(key_rot.shape)} v={tuple(value_rot.shape)} "
                        f"slotmap={tuple(_slotmap.shape)} {_slotmap.dtype} "
                        f"kc={tuple(kc.shape)} vc={tuple(vc.shape)} "
                        f"ksc={tuple(self.k_scale_cache.shape)} "
                        f"kzc={tuple(self.k_zero_cache.shape)} "
                        f"vsc={tuple(self.v_scale_cache.shape)} "
                        f"vzc={tuple(self.v_zero_cache.shape)}"
                    )
                type(self)._r115_cu_calls = getattr(type(self), "_r115_cu_calls", 0) + 1
                if (
                    os.environ.get("R168_CUSPLIT", "0") == "1"
                    and not _EXTRA_CTX.capturing
                    and int(attn_metadata.num_actual_tokens) > 256
                ):
                    # R168 P1: concat-shadow split test. Stash this chunk's
                    # rows; when the NEXT chunk of the same layer arrives,
                    # re-quantize BOTH as one piece into scratch and compare
                    # the FIRST chunk's slots: equal => kernel piece-invariant;
                    # different => chunk-split bakes different INT2 content.
                    try:
                        _st168 = getattr(type(self), "_r168_cusplit", None)
                        if _st168 is None:
                            _st168 = type(self)._r168_cusplit = {"pend": {}}
                        _lid168 = int(self._oscar_layer_ord(layer))
                        _pend = _st168["pend"].get(_lid168)
                        if _pend is not None:
                            import torch as _t168s

                            _kr0, _vr0, _sm0 = _pend
                            _bs168 = int(kc.shape[1])
                            _sc = _st168.setdefault(
                                "scratch",
                                (
                                    torch.zeros_like(kc),
                                    torch.zeros_like(vc),
                                    torch.zeros_like(self.k_scale_cache),
                                    torch.zeros_like(self.k_zero_cache),
                                    torch.zeros_like(self.v_scale_cache),
                                    torch.zeros_like(self.v_zero_cache),
                                ),
                            )
                            mod.cache_update_aclnn(
                                _t168s.cat([_kr0, _kr166], 0).contiguous(),
                                _t168s.cat([_vr0, _vr166], 0).contiguous(),
                                _t168s.cat([_sm0, _slotmap], 0).contiguous(),
                                _sc[0],
                                _sc[1],
                                _sc[2],
                                _sc[3],
                                _sc[4],
                                _sc[5],
                            )
                            _t168s.npu.synchronize()
                            _blk0 = _t168s.unique(_sm0 // _bs168)
                            _blk0l = _blk0.tolist()
                            _rs = kc[_blk0].view(len(_blk0l), -1).sum(dim=1).tolist()
                            _ss = _sc[0][_blk0].view(len(_blk0l), -1).sum(dim=1).tolist()
                            _rss = self.k_scale_cache[_blk0].float().sum(dim=(1, 2)).tolist()
                            _sss = _sc[2][_blk0].float().sum(dim=(1, 2)).tolist()
                            import json as _j168s

                            with open("/tmp/r168_cusplit.jsonl", "a") as _f168s:
                                _f168s.write(
                                    _j168s.dumps(
                                        {
                                            "lid": _lid168,
                                            "T1": int(_kr0.shape[0]),
                                            "T2": int(_kr166.shape[0]),
                                            "blk": [int(x) for x in _blk0l],
                                            "real_k": [round(x, 1) for x in _rs],
                                            "shadow_k": [round(x, 1) for x in _ss],
                                            "real_ks": [round(x, 3) for x in _rss],
                                            "shadow_ks": [round(x, 3) for x in _sss],
                                            "k_eq": _rs == _ss,
                                            "ks_eq": _rss == _sss,
                                        }
                                    )
                                    + "\n"
                                )
                            _st168["pend"][_lid168] = None
                        else:
                            _st168["pend"][_lid168] = (_kr166.clone(), _vr166.clone(), _slotmap.clone())
                    except Exception as _e168s:
                        print("[R168-CUSPLIT] fail: {}".format(_e168s), flush=True)
                # R118 eager-mode ladder dump (default OFF; never inside capture)
                if (
                    os.environ.get("R118_TRACE", "0") == "1"
                    and not _EXTRA_CTX.capturing
                    and attn_metadata.num_actual_tokens <= 16
                ):
                    try:
                        import importlib as _il118e

                        _rt118e = _il118e.import_module("r118_trace")
                    except ImportError:
                        _rt118e = None
                    if _rt118e is not None:
                        _en = globals().setdefault("_R118_EAGER_N", [0])
                        _rt118e.dump_row("eager", _en[0], {"L": layer_idx, "nat": int(attn_metadata.num_actual_tokens)})
                        _en[0] += 1
                # R116: the fused branch skips _reshape_and_cache_int2, which is
                # where self.key_cache/value_cache got bound. Bind them here —
                # REQUIRED for the three-tier middle candidate (reads
                # self.key_cache.shape). Eager legs previously got them from the
                # T>16 prefill pass; graph dummy runs are all T<=16 -> NoneType.
                self.key_cache, self.value_cache = kc, vc
                if os.environ.get("R115_CU_STATS", "0") == "1":
                    import json as _json115

                    with open(os.environ.get("R115_CU_STATS_OUT", "/tmp/r115_cu_calls.jsonl"), "a") as _f:
                        _f.write(
                            _json115.dumps(
                                {
                                    "pid": os.getpid(),
                                    "L": layer_idx,
                                    "T": int(attn_metadata.num_actual_tokens),
                                    "n": type(self)._r115_cu_calls,
                                }
                            )
                            + "\n"
                        )
            else:
                if getattr(type(self), "_r114_ops_lib", None) is None:
                    torch.ops.load_library(
                        os.path.join(os.environ.get("R114_RUNNER_VENDOR", ""), "oscar_cu/build/liboscar_cu_ops.so")
                    )
                    type(self)._r114_ops_lib = True
                _nt166c = int(attn_metadata.num_actual_tokens)
                _kr166c = key_rot.contiguous()
                _vr166c = value_rot.contiguous()
                _sm166c = attn_metadata.slot_mapping.to(torch.long)
                if os.environ.get("R166_CUFIX", "0") == "1" and key_rot.shape[0] > _nt166c:
                    _kr166c = key_rot[:_nt166c].contiguous()
                    _vr166c = value_rot[:_nt166c].contiguous()
                    _sm166c = _sm166c[:_nt166c].contiguous()
                torch.ops.oscar.int2_cache_update(
                    _kr166c,
                    _vr166c,
                    _sm166c,
                    kc,
                    vc,
                    self.k_scale_cache,
                    self.k_zero_cache,
                    self.v_scale_cache,
                    self.v_zero_cache,
                )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
                attn_metadata.reshape_cache_event.record()
            key_packed = value_packed = None  # fused: cache already written
        else:
            _r149n_T = int(attn_metadata.num_actual_tokens)
            _r149dbg = globals().setdefault("_R149_DBG_N", [0])
            _r149dbg[0] += 1
            _r150_tr("else_enter", T=int(_r149n_T), dbg=_r149dbg[0])
            if _r149dbg[0] <= 6:
                print(
                    f"[R149-SITE] #{_r149dbg[0]} T={_r149n_T} "
                    f"prerot={'yes' if pre_rotated_kv is not None else 'no'} "
                    f"flag={os.environ.get('R149_PREFILL_NATIVE', '0')} "
                    f"Rk={'yes' if getattr(self, '_oscar_R_k', None) is not None else 'no'}",
                    flush=True,
                )
            _r149n_on = (
                os.environ.get("R149_PREFILL_NATIVE", "0") == "1"
                and _r149n_T >= 1
                and _r149n_T <= 64
                and (
                    pre_rotated_kv is not None
                    or (getattr(self, "_oscar_R_k", None) is not None and getattr(self, "_oscar_R_v", None) is not None)
                )
            )
            if _r149n_on and getattr(type(self), "_r149_native_mod", None) is None:
                try:
                    import ctypes as _ct149

                    R149V = os.environ.get("R149_VENDOR", "")
                    _ct149.CDLL(R149V + "/op_api/lib/libcust_opapi.so", mode=_ct149.RTLD_GLOBAL)
                    import torch_npu as _tn149
                    from torch.utils import cpp_extension as _ce149

                    _TN149 = os.path.dirname(_tn149.__file__)
                    # R150 P0 fix: generated aclnn headers live in the build
                    # tree op_api/ — without it the extension build fails with
                    # 'aclnn_oscar_int2_prefill_pack_v1.h: No such file' and the
                    # native path silently degrades to legacy (R149 FIRST_FAIL).
                    _r150_inc = os.environ.get("R150_LAUNCH_INC", "")
                    type(self)._r149_native_mod = _ce149.load(
                        name="r149_prefill_launch_ext",
                        sources=[os.environ.get("R149_LAUNCH_SRC", "")],
                        extra_include_paths=[
                            _r150_inc,
                            os.path.join(_TN149, "include"),
                            os.path.join(_TN149, "include", "third_party", "op-plugin"),
                            "/usr/local/Ascend/cann-9.0.0/aarch64-linux/include",
                            "/usr/local/Ascend/cann-9.0.0/include",
                        ],
                        extra_ldflags=[
                            "-L" + R149V + "/op_api/lib",
                            "-lcust_opapi",
                            "-L/usr/local/Ascend/cann-9.0.0/lib64",
                            "-lascendcl",
                            "-L" + os.path.join(_TN149, "lib"),
                            "-ltorch_npu",
                        ],
                        verbose=False,
                    )
                except Exception as _e149:
                    print(f"[R149-NATIVE] launcher build FAILED: {_e149}", flush=True)
                    _r150_tr("launcher_build_failed", err=str(_e149)[:300])
                    type(self)._r149_native_mod = False
            _r149n_mod = getattr(type(self), "_r149_native_mod", None)
            if _r149n_on and _r149n_mod:
                # R149 P2: ONE native op call = quant+pack+scatter (T<=64 fast
                # regime: 0.67ms vs 2.7ms ATen at T=8; bit-exact 231/231+100x).
                if pre_rotated_kv is not None:
                    _rk, _rv = pre_rotated_kv
                else:
                    # same rotation as _quantize_kv_to_int2 (math order verbatim)
                    _rk = torch.matmul(key[:_r149n_T].contiguous(), self._oscar_R_k[self._oscar_layer_ord(layer)])
                    _rv = torch.matmul(value[:_r149n_T].contiguous(), self._oscar_R_v[self._oscar_layer_ord(layer)])
                _kc, _vc = kv_cache[0], kv_cache[1]
                _rc149 = type(self)._r149_native_mod.r149_prefill_quant_pack(
                    _rk[:_r149n_T].contiguous(),
                    _rv[:_r149n_T].contiguous(),
                    attn_metadata.slot_mapping[:_r149n_T].to(torch.long),
                    _kc,
                    _vc,
                    self.k_scale_cache,
                    self.k_zero_cache,
                    self.v_scale_cache,
                    self.v_zero_cache,
                    0,
                )
                if int(_rc149[0]) != 0 or int(_rc149[1]) != 0 or int(_rc149[2]) != 0:
                    raise RuntimeError(f"[R149-NATIVE] rc={tuple(int(x) for x in _rc149)}")
                _r149n_n = globals().setdefault("_R149_NATIVE_N", [0])
                _r149n_n[0] += 1
                _r150_tr(
                    "native_call",
                    T=int(_r149n_T),
                    rc=[int(x) for x in _rc149],
                    slot_n=_r149n_n[0],
                    layer_ord=int(self._oscar_layer_ord(layer)) if hasattr(self, "_oscar_layer_ord") else -1,
                )
                if _r149n_n[0] <= 3 or _r149n_n[0] % 1000 == 0:
                    print(
                        f"[R149-NATIVE] call#{_r149n_n[0]} T={_r149n_T} rc={tuple(int(x) for x in _rc149)}", flush=True
                    )
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event = torch.npu.Event()
                    attn_metadata.reshape_cache_event.record()
                # R116 contract: bind caches for the three-tier middle reader
                self.key_cache, self.value_cache = _kc, _vc
                key_packed = value_packed = None
            else:
                _r150_tr(
                    "quant_call", T=int(attn_metadata.num_actual_tokens), cpp=os.environ.get("R113_QUANT_CPP", "0")
                )
                k_packed, k_scale, k_zero, v_packed, v_scale, v_zero = self._quantize_kv_to_int2(
                    key,
                    value,
                    layer,
                    attn_metadata.num_actual_tokens,
                    pre_rotated_kv=pre_rotated_kv,
                )
                query, key_packed, value_packed, _ = self._reshape_and_cache_int2(
                    query,
                    k_packed,
                    k_scale,
                    k_zero,
                    v_packed,
                    v_scale,
                    v_zero,
                    kv_cache,
                    attn_metadata,
                    output,
                    layer_idx=layer_idx,
                )

        # R144 DIAG: capture dispatch decision — file-based (stderr suppressed)
        _r144dsp_n = globals().setdefault("_R144_DISP_N", [0])
        if _r144dsp_n[0] < 8:
            _r144dsp_n[0] += 1
            try:
                with open("/tmp/r144_disp_probe.log", "a") as _pf:
                    _pf.write(
                        f"[R144-DISP] call#{_r144dsp_n[0]} L={layer_idx} "
                        f"state={attn_metadata.attn_state} num_tok={num_tokens} "
                        f"TT_MID={os.environ.get('R109_THREETIER_MIDDLE', '?')} "
                        f"BATCHED={os.environ.get('R144_BATCHED_MIDDLE', '?')} "
                        f"cap={_EXTRA_CTX.capturing}\n"
                    )
            except Exception:
                pass
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            # EAGER-ONLY delivery: the full-graph capture variants are not
            # shipped; fail closed instead of silently routing to a path
            # this build does not carry (upstream BF16 graph behavior is
            # unaffected -- this class is only installed for INT2 models).
            if _EXTRA_CTX.capturing:
                raise RuntimeError(
                    "[OSCAR-INT2] graph capture is not supported by the "
                    "eager-only INT2 backend; run with enforce_eager "
                    "(fail-closed, no silent fallback)"
                )
            # R96 R91 bridge: env-gated, default OFF
            if os.environ.get("R95_R91_BRIDGE", "0") == "1":
                if self._r91_bridge_decode(query, key_rot, value_rot, attn_metadata, output, layer):
                    return output
            # R109 Track B: candidate replaces ONLY the INT2 middle tier (default OFF)
            if os.environ.get("R109_THREETIER_MIDDLE", "0") == "1":
                return self._forward_int2_decode_threetier_middle(
                    query, key_rot, value_rot, attn_metadata, output, layer
                )
            return self._forward_int2_decode_threetier(query, key_rot, value_rot, attn_metadata, output, layer)
        else:
            float_key, float_value = key, value
            return self._forward_int2_prefill_threetier(query, float_key, float_value, attn_metadata, output, layer)
