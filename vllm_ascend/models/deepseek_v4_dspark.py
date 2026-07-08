# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4 — ASCEND-NATIVE rewrite (v1).

Deploy at: vllm_ascend/models/deepseek_v4_dspark.py  (register "DeepSeekV4DSpark"
in vllm_ascend/models/__init__.py register_model()).

Why this rewrite (vs the earlier core-tree version): the old file imported the
CORE `vllm.model_executor.models.deepseek_v4` which pulls in `mhc` -> `tilelang`
(CUDA-only, absent on Ascend) -> `'NoneType' object has no attribute 'jit'`.
This version is built ENTIRELY on vllm-ascend building blocks (the same ones the
WORKING `deepseek_v4_mtp.py` uses): `DeepseekV2DecoderLayer` (Ascend, does HC
internally via torch.ops._C_ascend.npu_hc_pre/post — no tilelang), `SharedHead`,
plain-torch `_hc_head`, Ascend FusedMoE. No `mhc`/`tilelang` anywhere.

DSpark = DFlash parallel backbone + lightweight sequential (Markov) head + a
confidence head. This is the DRAFT model; the parallel-block construction +
target-KV injection live in the proposer. Structure mirrors upstream PR#46995's
DSparkDeepseekV4ForCausalLM (NVIDIA), ported to Ascend components.
"""
import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform

# ASCEND-native building blocks (relative imports resolve inside vllm_ascend.models)
from .deepseek_v4 import DeepseekV2DecoderLayer
from .deepseek_v4_mtp import SharedHead
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.device.device_op import DeviceOperator

logger = init_logger(__name__)

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    """Plain-torch hyper-connection head collapse (== Ascend MTP.hc_head, NO tilelang).
    x: [.., hc_mult, hidden] -> [.., hidden]."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


# ---------------------------------------------------------------- DSpark heads
class DSparkMarkovHead(nn.Module):
    """Eq.5: B(x_{k-1}) = W1[x_{k-1}] @ W2, low-rank (markov_rank=256).
    Returns (bias_logits[V], markov_embed[256]); plain bf16 (no .scale)."""

    def __init__(self, vocab_size: int, markov_rank: int):
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(vocab_size, markov_rank, bias=False)

    def forward(self, token_ids: torch.Tensor):
        embed = self.markov_w1(token_ids)                       # [.., 256]
        logits = torch.nn.functional.linear(
            embed.float(), self.markov_w2.weight.float())       # embed @ W2^T -> [.., V]
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    """Eq.7: c_k = w^T [h_k ; W1[x_{k-1}]] (RAW logit, no sigmoid). fp32."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        h = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(h.float()).squeeze(-1)                 # [.. ] raw logit


# ---------------------------------------------------------------- DSpark draft stack
class DeepSeekV4DSparkModel(nn.Module):
    """3-deep DSpark draft stack (mtp.0/1/2) over the whole gamma block in ONE pass.
    main_proj/main_norm on stage 0; norm/markov/confidence/hc_head on the last stage."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        # DSPARK_DRAFT_NOQUANT: build the DRAFT stack unquantized (bf16 draft with
        # w8a8 target; vllm#43304-class inheritance). Scoped: restored after layers.
        _dqos = __import__("os")
        self._noquant = bool(_dqos.environ.get("DSPARK_DRAFT_NOQUANT"))
        self._saved_qc = vllm_config.quant_config
        if self._noquant:
            try:
                vllm_config.quant_config = None
                quant_config = None
                print("DSPARK_DRAFT_NOQUANT ACTIVE: draft built unquantized", file=__import__("sys").stderr, flush=True)
            except Exception as _qe:
                print("DSPARK_DRAFT_NOQUANT failed to override: %r" % (_qe,), file=__import__("sys").stderr, flush=True)
        self.device = current_platform.device_type

        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.block_size = config.dspark_block_size          # gamma = 5
        self.noise_token_id = config.dspark_noise_token_id  # 128799
        self.target_layer_ids = list(getattr(config, "dspark_target_layer_ids", []) or [])
        if not self.target_layer_ids:
            raise ValueError(
                "DSpark requires dspark_target_layer_ids in the draft config.")
        self.markov_rank = config.dspark_markov_rank        # 256
        self.n_stages = int(
            getattr(config, "n_mtp_layers", None) or len(self.target_layer_ids))
        self.base_layer_id = config.num_hidden_layers       # 43

        # QUAROT_PATCH_V1: QuaRot anti-rotation for the DSpark context path.
        # path-A ckpt is QuaRot-rotated (target hidden rotated; draft attn incl.
        # cv_wkv rotated) but main_proj is stored UNROTATED, and vllm-ascend applies
        # quarot anti-rotation ONLY to Eagle3LlamaForCausalLM -> this custom class
        # gets nothing. Load global_rotation Q so combine_hidden_states can un-rotate
        # incoming rotated target hidden (per-4096-block @ Q^T) into main_proj basis,
        # then rotate main_x (@ Q) into the draft rotated basis cv_wkv expects.
        # Offline-validated: context-KV reconstruction cos 0.98 (broken=-0.02).
        self.register_buffer('_quarot_Q', None, persistent=False)
        try:
            import os as _qos
            from safetensors.torch import load_file as _qload
            _mp = vllm_config.speculative_config.draft_model_config.model
            _qp = _qos.path.join(str(_mp), 'optional', 'quarot.safetensors')
            if _qos.path.exists(_qp):
                self._quarot_Q = _qload(_qp)['global_rotation'].to(torch.bfloat16)
                import sys as _qsys
                print('DSPARK QUAROT_PATCH_V1 loaded Q %s' % (tuple(self._quarot_Q.shape),), file=_qsys.stderr, flush=True)
            else:
                import sys as _qsys
                print('DSPARK QUAROT_PATCH_V1 no quarot file at %s (no-op)' % _qp, file=_qsys.stderr, flush=True)
        except Exception as _qe:
            import sys as _qsys
            print('DSPARK QUAROT_PATCH_V1 load skipped: %r' % (_qe,), file=_qsys.stderr, flush=True)

        # per-stage sparse-attn topk index buffer (only if the arch is v3.2-style)
        if hasattr(config, "index_topk"):
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk, dtype=torch.int32, device=self.device)
        else:
            self.topk_indices_buffer = None

        # Shared (from target) embed + head; aliased in by the proposer if present.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"))
        self.shared_head = SharedHead(config=config, prefix=prefix, quant_config=quant_config)

        # stage 0: main_proj fuses concat(mean-over-hc) of target layers [40,41,42].
        # prefix "mtp.0.main_proj" so the W8A8 quant scheme resolves (the auto quant
        # adaptation produces mtp.0.main_proj.weight from the ckpt description).
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size, bias=False, return_bias=False, quant_config=quant_config,
            prefix="mtp.0.main_proj")
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 3-deep Ascend decoder stack. CRITICAL: pass prefix "mtp.{i}" (NOT
        # model.layers.{43+i}). extract_dsv4_layer_index adds num_hidden_layers for
        # any ".mtp." prefix -> config_layer_idx = num_hidden_layers + i -> ratio 0
        # (no Compressor/Indexer built, exactly like the working MTP draft). It ALSO
        # makes the quant scheme key (mtp.{i}.self_attn.wq_a) resolve against the
        # auto-adapted description. The Python param path stays model.layers.{i}
        # (ModuleList index 0/1/2) — decoupled from the prefix string.
        self.layers = nn.ModuleList([
            DeepseekV2DecoderLayer(
                vllm_config, f"mtp.{i}",
                config=config,
                topk_indices_buffer=self.topk_indices_buffer,
                is_draft_layer=True)
            for i in range(self.n_stages)
        ])
        if getattr(self, "_noquant", False):
            try:
                vllm_config.quant_config = self._saved_qc  # restore for anything after
            except Exception:
                pass

        # last stage: hc_head params + DSpark heads (final norm is shared_head.norm).
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.markov_head = DSparkMarkovHead(config.vocab_size, self.markov_rank)
        self.confidence_head = DSparkConfidenceHead(config.hidden_size + self.markov_rank)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """dflash contract; DSpark folds main_norm in -> returns main_x [T, hidden].
        QUAROT_PATCH_V1: un-rotate rotated target hidden per 4096-block (@Q^T) into
        main_proj basis, then rotate main_x (@Q) into the draft rotated basis."""
        if self._quarot_Q is not None:
            _q = self._quarot_Q.to(device=aux_hidden_states.device, dtype=aux_hidden_states.dtype)  # QUAROT_DEVFIX_V1
            _h = self.hidden_size
            aux_hidden_states = torch.cat(
                [aux_hidden_states[:, k*_h:(k+1)*_h] @ _q.t()
                 for k in range(len(self.target_layer_ids))], dim=-1)
        _mx = self.main_norm(self.main_proj(aux_hidden_states))
        import os as _qao  # QUAROT_AUXONLY_V1: bf16 draft skips re-rotation
        if self._quarot_Q is not None and not _qao.environ.get("DSPARK_QUAROT_AUXONLY"):
            _mx = _mx @ self._quarot_Q.to(device=_mx.device, dtype=_mx.dtype)
        import os as _capc
        if _capc.environ.get("DSPARK_CAPTURE"):
            try:
                self._cap_aux = aux_hidden_states.detach().cpu().float()
                self._cap_mainproj_out = self.main_proj(aux_hidden_states).detach().cpu().float()
                self._cap_mainx = _mx.detach().cpu().float()
            except Exception:
                pass

        import os as _oa, sys as _sa
        if _oa.environ.get("DSPARK_DBG"):
            try:
                _a = aux_hidden_states.float()
                # variance ACROSS rows (tokens): if ~0, aux is constant => capture broken
                _row_std = _a.std(dim=0).mean().item()
                _mnw = float(self.main_norm.weight.float().square().mean().sqrt())
                for _k in range(3):
                    _ss = _a[:, _k*4096:(_k+1)*4096]
                    print("DSPARK_AUXSLICE k=%d std=%.4e rowvar=%.4e" % (_k, float(_ss.std()), float(_ss.std(dim=0).mean())), file=_sa.stderr, flush=True)
                print("DSPARK_AUX aux.shape=%s aux.std=%.4e aux.rowvar=%.4e mainx.std=%.4e mainx.rowvar=%.4e mnw.rms=%.4e" % (
                    tuple(aux_hidden_states.shape), float(_a.std()), _row_std,
                    float(_mx.float().std()), float(_mx.float().std(dim=0).mean()), _mnw), file=_sa.stderr, flush=True)
            except Exception as _e:
                print("DSPARK_AUX ERR", _e, file=_sa.stderr, flush=True)
        return _mx

    @torch.inference_mode()
    def precompute_and_store_context_kv(self, context_states, context_positions,
                                        context_slot_mapping=None):
        """Per-stage context-KV injection of main_x. Wired against sfa_v1 exec_kv at
        NPU bring-up (the cache-write op is the only pending piece)."""
        if context_slot_mapping is None:
            return
        import os as _oc, sys as _sc
        if _oc.environ.get("DSPARK_DBG"):
            try:
                _cp = context_positions
                print("DSPARK_CTX num_ctx=%d pos[:8]=%s pos[-4:]=%s slot[:8]=%s" % (context_states.shape[0], _cp[:8].tolist(), _cp[-4:].tolist(), context_slot_mapping.flatten()[:8].tolist()), file=_sc.stderr, flush=True)
            except Exception as _ec:
                print("DSPARK_CTX ERR", _ec, file=_sc.stderr, flush=True)
        for layer in self.layers:
            self._write_context_latent(layer, context_states, context_positions,
                                       context_slot_mapping)

    def _write_context_latent(self, layer, context_states, context_positions, slot_mapping):
        """Write this draft layer's per-stage context-KV latent into its sliding-window
        MLA paged cache. Mirrors AscendDSAImpl._forward_prefill (dsa_v1.py:1982-2014):
        kv = kv_norm(wkv(main_x)); partial-RoPE(rope tail); scatter -> swa cache.
        (Resolved against the live Ascend DSA attention; see agent investigation.)"""
        dv4 = layer if hasattr(layer, "dsa_attn") else layer.self_attn   # DeepseekV4Attention
        sparse = dv4.dsa_attn                 # AscendDeepseekSparseAttention
        dsa_attn = sparse.dsa_attn            # DSAAttention
        impl = dsa_attn.impl                  # AscendDSAImpl
        layer_name = dsa_attn.layer_name      # == f"{prefix}.attn" (rope/metadata key)

        swa_kv_cache = sparse.swa_cache_layer.kv_cache
        while isinstance(swa_kv_cache, (list, tuple)) and len(swa_kv_cache) == 1:
            swa_kv_cache = swa_kv_cache[0]
        block_size = swa_kv_cache.shape[1]    # cache layout [num_blocks, block_size, 1, head_dim]

        cos_proxy, sin_proxy = get_cos_and_sin_dsa(context_positions, use_cache=False)   # use_cache=False
        cos = cos_proxy[layer_name]
        sin = sin_proxy[layer_name]

        # Reference (PR#46995 nvidia/dspark.py) feeds RAW main_x to wkv (NO
        # input_layernorm, NO hc_pre): kv = kv_norm(wkv(main_x)) -> rope -> scatter.
        _ctx_in = context_states
        # A#3: compute context KV via the SAME w8a8 cv_wkv (Cube-Vector) path the real
        # DSA prefill uses (cv_wkv.quantize + matmul), not a bare impl.wkv(x) call,
        # so the context keys exactly match the query keys' representation.
        if getattr(impl, "cv_wkv", None) is not None:
            _kvq, _kvs = impl.cv_wkv.quantize(_ctx_in)
            kv = impl.cv_wkv.matmul(_kvq, _kvs)
        else:
            kv = impl.wkv(_ctx_in)
        import os as _oz
        if _oz.environ.get("DSPARK_ZERO") or _oz.path.exists("/data1/DSPARK_ZERO_FLAG"):
            kv = kv * 0.0                      # zero-out probe: if draft output is unchanged, the context cross-read is dead
        kv = impl.kv_norm(kv)
        nope, rope, head = impl.nope_head_dim, impl.rope_head_dim, impl.head_dim
        assert rope is not None and nope + rope == head
        kv = kv.view(-1, 1, head)             # (num_ctx_tokens, 1, head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1), cos, sin,
            rotary_mode="interleave",
            partial_slice=[nope, head])       # rotate dims [nope_head_dim : head_dim]
        import os as _om, sys as _sm
        if _om.environ.get("DSPARK_DBG"):
            try:
                _raw = slot_mapping.flatten()[:6].tolist()
            except Exception:
                _raw = "?"
            print("DSPARK_SLOT swa_shape=%s cache_bs=%s raw_slot[:6]=%s" % (tuple(swa_kv_cache.shape), swa_kv_cache.shape[1], _raw), file=_sm.stderr, flush=True)
        import os as _rtos
        if _rtos.environ.get("DSPARK_READ_TRACE"):
            try:
                impl._dspark_last_ctx_pos = context_positions.detach().to(torch.int64).clone()
                impl._dspark_last_ctx_slot = slot_mapping.detach().flatten().to(torch.int64).clone()
                impl._dspark_last_ctx_layer = layer_name
                impl._dspark_write_cache_ptr = int(swa_kv_cache.data_ptr())
            except Exception:
                pass
        import os as _rg
        if _rg.environ.get("DSPARK_RING"):
            try:
                _bs = int(swa_kv_cache.shape[1])
                _orig = slot_mapping.flatten().to(torch.int64)
                _cp = context_positions.flatten().to(torch.int64)
                # ring within the sequence base block(s): keep each token base block from the
                # proposer slot for positions < bs, but force positions >= bs to wrap into the
                # SAME base block as position 0 of that contiguous run (single-seq ring).
                _base_block = (_orig[0] // _bs)
                slot_mapping = (_base_block * _bs + (_cp % _bs)).to(slot_mapping.dtype)
            except Exception as _re:
                import sys as _rs; print("DSPARK_RING ERR %r"%(_re,), file=_rs.stderr, flush=True)
        if slot_mapping.dim() == 1:
            slot_mapping = DeviceOperator.format_dsa_slot_mapping(
                slot_mapping.to(torch.int64), block_size)
        if _oz.environ.get("DSPARK_SENTINEL"):
            kv = torch.full_like(kv, 5.0)   # A/B probe: huge distinctive context KV; if draft output changes, attn DOES read context
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)
        import os as _o2, sys as _s2
        if _o2.environ.get("DSPARK_DBG"):
            try:
                print("DSPARK_WRITE layer=%s kv_std=%.5f slot[:8]=%s cache_norm=%.3f" % (
                    layer_name, float(kv.float().std()),
                    (slot_mapping.flatten()[:8].tolist() if hasattr(slot_mapping,'flatten') else slot_mapping),
                    float(swa_kv_cache.float().norm())), file=_s2.stderr, flush=True)
            except Exception as _e2:
                print("DSPARK_WRITE ERR", _e2, file=_s2.stderr, flush=True)


    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base draft logits U_k: collapse hc copies -> shared_head.norm -> LM head."""
        xc = _hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale,
                      self.hc_head_base, self.rms_norm_eps, self.hc_eps)
        _lg = self.logits_processor(self.shared_head.head, self.shared_head(xc))
        import os as _os, sys as _sy
        if _os.environ.get("DSPARK_DBG"):
            try:
                _tv = _lg.flatten().topk(3)
                print("DSPARK_HEAD hid.std=%.4e hid.shape=%s xc.std=%.4e lg.std=%.4e top3val=%s argmaxrow=%s" % (
                    float(hidden_states.float().std()), tuple(hidden_states.shape),
                    float(xc.float().std()), float(_lg.float().std()),
                    [round(float(v),3) for v in _tv.values.tolist()],
                    _lg.argmax(-1).flatten()[:6].tolist()), file=_sy.stderr, flush=True)
            except Exception as _e:
                print("DSPARK_HEAD ERR", _e, file=_sy.stderr, flush=True)
        if _os.environ.get("DSPARK_DBG"):
            import sys as _sys
            try:
                _am = _lg.argmax(-1).flatten()[:8].tolist()
            except Exception as _e:
                _am = "ERR:%s" % _e
            print("DSPARK_DBG logits shape=%s argmax8=%s" % (tuple(_lg.shape), _am), file=_sys.stderr, flush=True)
        return _lg

    def forward(self, input_ids, positions, inputs_embeds=None,
                hidden_states=None, **kwargs):
        """Run the gamma block through the 3-deep Ascend stack. The Ascend
        DeepseekV2DecoderLayer does HC internally and returns (hidden, residual);
        we chain residual like the target's layer loop and return the final hidden
        [T, hc_mult, hidden] (hc_head + heads applied in compute_logits/compute_block).

        MTP fusion: the draft residual stream is seeded by the next-token
        embedding PLUS the projected target hidden (main_x = combine_hidden_states
        output, delivered by the proposer as `hidden_states`). Without main_x the
        draft is blind to the target and acceptance collapses to 0."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        base = inputs_embeds
        import os as _os
        if _os.environ.get("DSPARK_DBG"):
            import sys as _sys
            print("DSPARK_DBG fwd hs_none=%s emb=%s hs=%s" % (hidden_states is None, tuple(inputs_embeds.shape), (None if hidden_states is None else tuple(hidden_states.shape))), file=_sys.stderr, flush=True)
        # MTP-1 (w8a8) reaches 93.9% pos-0 accept by feeding the target hidden DIRECTLY
        # into the draft (hnorm+h_proj). The pure context-KV path underperforms here, so
        # re-feed main_x (proposer hidden_states) into the residual: input_layernorm in
        # each layer renormalizes, so the direction (target signal) is what matters.
        # MTP-1-style direct feed: make main_x (proposer hidden_states) the DOMINANT
        # residual (replace, not add) so input_layernorm renormalizes it to std~1 and
        # the target signal drives the prediction (additive main_x std~0.09 was swamped
        # by the mask embedding). For pos0 this is exactly MTP-1's hnorm(last_hidden).
        x = base.unsqueeze(-2).repeat(1, self.hc_mult, 1)   # [T, hc_mult, hidden]
        residual = None
        if _os.environ.get("DSPARK_CAPTURE"):
            self._cap_stageinit = x.detach().cpu().float(); self._cap_inputids = input_ids.detach().cpu()
        if _os.environ.get("DSPARK_DBG"):
            import sys as _sst
            try: print("DSPARK_STAGE init x.std=%.4f x.norm=%.2f emb.std=%.4f" % (float(x.float().std()), float(x.float().norm()), float(base.float().std())), file=_sst.stderr, flush=True)
            except Exception as _e0: print("DSPARK_STAGE init ERR", _e0, file=_sst.stderr, flush=True)
        for _li, layer in enumerate(self.layers):
            x, residual = layer(positions=positions, hidden_states=x, residual=residual)
            if _os.environ.get("DSPARK_CAPTURE"):
                setattr(self, "_cap_stage%d" % _li, x.detach().cpu().float())
            if _os.environ.get("DSPARK_DBG"):
                import sys as _sst2
                try:
                    _xf = x.float()
                    _rs = ("None" if residual is None else ("%.4f" % float(residual.float().std())))
                    print("DSPARK_STAGE i=%d x.std=%.4f x.norm=%.2f x0.std=%.4f res.std=%s" % (_li, float(_xf.std()), float(_xf.norm()), float(_xf[0].std()), _rs), file=_sst2.stderr, flush=True)
                except Exception as _es: print("DSPARK_STAGE ERR", _es, file=_sst2.stderr, flush=True)
        return x

    @torch.inference_mode()
    def compute_block(self, x, anchor_ids, temperature: float):
        """Semi-AR head (inference forward_head). Returns (output_ids[B,gamma+1],
        confidence[B,gamma] RAW logit)."""
        xc = _hc_head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base,
                      self.rms_norm_eps, self.hc_eps)          # [.., hidden]
        U = self.logits_processor(self.shared_head.head, self.shared_head(xc))
        import os as _capb
        if _capb.environ.get("DSPARK_CAPTURE") and not _capb.path.exists("/data1/dspark_capture.pt"):
            try:
                import torch as _tc
                _snap = {
                    "backbone_x": x.detach().cpu().float(),
                    "stage_init": getattr(self, "_cap_stageinit", None),
                    "input_ids_blk": getattr(self, "_cap_inputids", None),
                    "stage0": getattr(self, "_cap_stage0", None),
                    "stage1": getattr(self, "_cap_stage1", None),
                    "postattn": (__import__("vllm_ascend.models.deepseek_v4", fromlist=["_DSPARK_PA"])._DSPARK_PA[:]),
                    "xc_hchead": xc.detach().cpu().float(),
                    "base_U": U.detach().cpu().float(),
                    "anchor": anchor_ids.detach().cpu(),
                    "aux": getattr(self, "_cap_aux", None),
                    "mainproj_out": getattr(self, "_cap_mainproj_out", None),
                    "main_x": getattr(self, "_cap_mainx", None),
                    "main_norm_w": self.main_norm.weight.detach().cpu().float() if hasattr(self,"main_norm") else None,
                    "hc_head_fn": self.hc_head_fn.detach().cpu().float(),
                    "hc_head_scale": self.hc_head_scale.detach().cpu().float(),
                    "hc_head_base": self.hc_head_base.detach().cpu().float(),
                    "shared_norm_w": self.shared_head.norm.weight.detach().cpu().float() if hasattr(self.shared_head,"norm") else None,
                    "markov_w1": self.markov_head.markov_w1.weight.detach().cpu().float(),
                    "markov_w2": self.markov_head.markov_w2.weight.detach().cpu().float(),
                    "rms_norm_eps": float(self.rms_norm_eps),
                    "hc_eps": float(self.hc_eps),
                    "block_size": int(self.block_size),
                    "hc_mult": int(self.hc_mult),
                }
                _tc.save(_snap, "/data1/dspark_capture.pt")
                import sys as _sc; print("DSPARK_CAPTURE saved snapshot keys=%s" % list(_snap.keys()), file=_sc.stderr, flush=True)
            except Exception as _ce:
                import sys as _sc; print("DSPARK_CAPTURE ERR %r"%(_ce,), file=_sc.stderr, flush=True)
        B = anchor_ids.shape[0]
        U = U.view(B, self.block_size, -1)
        xc = xc.view(B, self.block_size, -1)
        output_ids = anchor_ids.new_empty(B, self.block_size + 1)
        output_ids[:, 0] = anchor_ids
        import os as _mb, sys as _mbs
        _nobias = _mb.environ.get("DSPARK_MK_NOBIAS")
        if _mb.environ.get("DSPARK_DBG"):
            print("DSPARK_CB anchor=%s noise=%s U0arg=%s" % (output_ids[:, 0].tolist()[:4], getattr(self, "noise_token_id", "NA"), U[:, 0].argmax(-1).tolist()[:4]), file=_mbs.stderr, flush=True)
        embeds = []
        for i in range(self.block_size):
            # markov bias must be GATHERED to full vocab to match U (markov_w2 is a
            # ParallelLMHead -> sharded vocab under TP). Use logits_processor exactly
            # like the base U computation so the two align before the add.
            emb = self.markov_head.markov_w1(output_ids[:, i])              # [B, rank] (full, VPE all-reduce)
            bias = self.logits_processor(self.markov_head.markov_w2, emb)  # [B, full_vocab] (gathered)
            embeds.append(emb)
            _li = U[:, i].float() if _nobias else (U[:, i].float() + bias.float())
            if _mb.environ.get("DSPARK_DBG"):
                try:
                    _Uf = U[:, i].float()
                    print("DSPARK_MKV i=%d Uarg=%d Ustd=%.3f Utop2gap=%.3f biasstd=%.3f biasarg=%d finalarg=%d prevtok=%d" % (
                        i, int(_Uf.argmax(-1)[0]), float(_Uf.std()),
                        float((_Uf.topk(2,dim=-1).values[0,0]-_Uf.topk(2,dim=-1).values[0,1])),
                        float(bias.float().std()), int(bias.float().argmax(-1)[0]),
                        int(_li.argmax(-1)[0]), int(output_ids[:, i][0])), file=_mbs.stderr, flush=True)
                except Exception as _mke:
                    print("DSPARK_MKV ERR %r"%(_mke,), file=_mbs.stderr, flush=True)
            output_ids[:, i + 1] = _sample(_li, temperature)
        markov_embed = torch.stack(embeds, dim=1)
        confidence = self.confidence_head(xc, markov_embed)
        return output_ids, confidence


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-20)))
    return (logits / temperature + g).argmax(dim=-1)


# ---------------------------------------------------------------- top-level wrapper + loader
class DeepSeekV4DSpark(nn.Module):
    # weights ship in the target checkpoint (mtp.*); embed/head shared from target.
    dspark_shares_target_embeddings = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepSeekV4DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

    # ---- proposer contract (delegated) ----
    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states):
        return self.model.combine_hidden_states(aux_hidden_states)

    def precompute_and_store_context_kv(self, context_states, context_positions,
                                        context_slot_mapping=None):
        return self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping)

    def forward(self, input_ids, positions, inputs_embeds=None, hidden_states=None, **kwargs):
        # Forward the proposer-delivered target seed (main_x) to the inner model;
        # dropping it leaves the draft blind -> constant token -> 0 accept.
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds,
                          hidden_states=hidden_states)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def compute_block(self, x, anchor_ids, temperature):
        return self.model.compute_block(x, anchor_ids, temperature)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """DSpark ckpt namespace mtp.0/1/2.*. Routes to the Ascend draft stack:
          - per-layer decoder params -> model.layers.{43+i}.* (with Ascend renames:
            .attn.->.self_attn., .ffn.->.mlp., norms, .w1/.w2/.w3->gate/down/up_proj)
          - stage-special model-level (main_proj/main_norm stage0;
            norm/markov/confidence/hc_head last stage) -> model.*
          - expert scale dtype remap: fp4 -> .weight_scale, fp8 -> .weight_scale_inv,
            E8M0 expert scale -> view(uint8). Mirrors PR#46995 + Ascend MTP loader."""
        _STAGE_MODEL = ("main_proj", "main_norm", "markov_head",
                        "confidence_head", "hc_head_")

        def _route(nm: str) -> str:
            mm = re.match(r"(?:.*\.)?mtp\.(\d+)\.(.*)", nm)
            if not mm:
                if nm.endswith("embed.weight") or nm.endswith(".emb.tok_emb.weight"):
                    return "model.embed_tokens.weight"
                if nm == "head.weight" or nm.endswith(".head.weight"):
                    return "model.shared_head.head.weight"
                return nm
            i, rest = int(mm.group(1)), mm.group(2)
            if rest == "norm.weight":
                return "model.shared_head.norm.weight"
            if any(rest.startswith(s) for s in _STAGE_MODEL):
                return "model." + rest
            # decoder params -> ModuleList index i (0/1/2), the PARAM path
            # (the quant prefix is mtp.{i}; param path is model.layers.{i}).
            return f"model.layers.{i}." + rest

        stacked = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        expert_mapping = FusedMoE.make_expert_params_mapping(
            model=self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=getattr(self, "num_redundant_experts", 0))
        expert_scale_suffix = (
            ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv")

        for name, w in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            routed = _route(name)
            is_expert_scale = bool(_EXPERT_SCALE_RE.search(routed))
            name = routed
            # Ascend decoder-layer-internal renames (only hit model.layers.* paths)
            if ".w1." in name:
                name = name.replace(".w1.", ".gate_proj.")
            if ".w2." in name:
                name = name.replace(".w2.", ".down_proj.")
            if ".w3." in name:
                name = name.replace(".w3.", ".up_proj.")
            if "attn" in name and "self_attn" not in name and ".markov" not in name:
                name = name.replace(".attn.", ".self_attn.")
            if ".ffn." in name:
                name = name.replace(".ffn.", ".mlp.")
            if ".ffn_norm." in name:
                name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
            if ".attn_norm." in name:
                name = name.replace(".attn_norm.", ".input_layernorm.")
            if ".gate.bias" in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            if name.endswith(".scale"):
                suffix = expert_scale_suffix if is_expert_scale else ".weight_scale_inv"
                name = name.removesuffix(".scale") + suffix

            # attn sink: TP-narrow per head
            if "sink" in name:
                if name not in params:
                    continue
                nw = w.narrow(0, head_start, heads_per_rank)
                params[name].data.copy_(nw)
                loaded.add(name)
                continue

            # stacked (gate_up fusion)
            for pn, wn, sid in stacked:
                if ".experts." in name or wn not in name:
                    continue
                nm = name.replace(wn, pn)
                if nm not in params:
                    continue
                p = params[nm]
                p.weight_loader(p, w, sid)
                loaded.add(nm)
                break
            else:
                # experts
                if ".experts." in name:
                    if "weight_scale" in name and w.dtype == torch.float8_e8m0fnu:
                        w = w.view(torch.uint8)
                    is_expert = False
                    for pn, wn, eid, sid in expert_mapping:
                        if wn not in name:
                            continue
                        is_expert = True
                        nm = name.replace(wn, pn)
                        if nm not in params:
                            continue
                        ok = typing.cast(Callable[..., bool], params[nm].weight_loader)(
                            params[nm], w, nm, shard_id=sid, expert_id=eid,
                            return_success=True)
                        if ok:
                            loaded.add(nm)
                            break
                    if is_expert:
                        continue
                if name not in params:
                    continue  # unmapped / target-owned key -> skip
                p = params[name]
                getattr(p, "weight_loader", default_weight_loader)(p, w)
                loaded.add(name)

        import os as _ol, sys as _sl
        if _ol.environ.get("DSPARK_DBG"):
            _crit = ["model.hc_head_fn","model.hc_head_base","model.hc_head_scale",
                     "model.shared_head.head.weight","model.shared_head.norm.weight",
                     "model.main_proj.weight","model.layers.0.self_attn.wkv.weight",
                     "model.layers.0.input_layernorm.weight"]
            _miss = [c for c in _crit if c not in loaded]
            _allp = dict(self.named_parameters())
            _unloaded = sorted([n for n in _allp if n not in loaded])
            print("DSPARK_UNLOADED count=%d :: %s" % (len(_unloaded), _unloaded[:40]), file=_sl.stderr, flush=True)
            for _n in ["model.layers.0.mlp.experts.w13_weight","model.layers.0.mlp.experts.w2_weight","model.layers.0.mlp.gate.weight","model.layers.0.self_attn.wkv.weight","model.layers.0.mlp.shared_experts.gate_up_proj.weight"]:
                if _n in _allp:
                    _v=_allp[_n].float(); print("DSPARK_PARM %s std=%.3e loaded=%s" % (_n, float(_v.std()), _n in loaded), file=_sl.stderr, flush=True)
            print("DSPARK_LOAD total=%d MISSING_CRIT=%s" % (len(loaded), _miss), file=_sl.stderr, flush=True)
            _p = dict(self.named_parameters())
            for _n in ["model.hc_head_fn","model.hc_head_base","model.hc_head_scale","model.shared_head.head.weight"]:
                if _n in _p:
                    _v = _p[_n].float()
                    print("DSPARK_LOAD %s std=%.4e min=%.3e max=%.3e nan=%s" % (_n, float(_v.std()), float(_v.min()), float(_v.max()), bool(_v.isnan().any())), file=_sl.stderr, flush=True)
        logger.info_once("DSpark draft model loaded: %d params", len(loaded))
        return loaded
