# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpine parallel-draft model for Ascend NPU (vLLM V1 path).

DSpine (tencent/Qwen3-8B-DSpine-b16) extends the DFlash block-parallel draft
with per-layer "code bridge" writes (Adjacent Causal Injection) and a last-write
refinement pass. The block's first row is the committed anchor token; rows
1..block_size-1 are speculative. After every decoder layer that is marked active
in ``dflash_config.selected_prefix_feature_layers``, the bridge injects a
predecessor-conditioned message into the residual stream and fuses the next
layer's input RMSNorm (or the final norm for the last layer).

Token selection after the single parallel draft forward is a serial last-write
refinement: the final bridge write is recomputed for every (sender, candidate)
pair over the top-K candidates of each proposal row, and the K-way table is
walked greedily from the anchor. This file contains the model side plus the
pure-torch refinement helpers used by ``AscendDSpineProposer``.

References:
  - sglang_dspine reference implementation (tencent/dspine, rev 07f326a).
  - Upstream DFlash: vllm.model_executor.models.qwen3_dflash.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

BRIDGE_PREFIX = "selected_prefix_bridge."
VERSION_TENSORS = ("draft_block_causal_version", "last_write_refine_version")


def _dflash_config(config) -> dict:
    cfg = getattr(config, "dflash_config", None)
    return cfg if isinstance(cfg, dict) else {}


def resolve_dspine_head_topk(config, num_layers: int) -> int:
    """Resolve the refinement width K (``head_topk``) from the draft config.

    ``code_bridge_second_pass`` (mode ``serial_table``) and
    ``dflash_config.selected_prefix_last_write_refine`` (kind ``exact_sender``)
    are the two sources; both must agree when present.
    """
    dflash_cfg = _dflash_config(config)
    found = {}
    second = getattr(config, "code_bridge_second_pass", None)
    if second is not None:
        second = dict(second)
        unknown = set(second) - {"mode", "first_exact", "head_topk"}
        if unknown:
            raise ValueError(f"Unknown code_bridge_second_pass keys: {sorted(unknown)}")
        if second.get("mode") != "serial_table":
            raise ValueError(
                "DSpine serves the serial last-write refinement only; "
                f"code_bridge_second_pass.mode={second.get('mode')!r}"
            )
        if int(second.get("first_exact", -1)) != num_layers - 1:
            raise ValueError(
                "serial_table redoes the LAST write only: "
                f"code_bridge_second_pass.first_exact must be {num_layers - 1}, "
                f"got {second.get('first_exact')!r}"
            )
        found["code_bridge_second_pass.head_topk"] = int(second["head_topk"])
    refine = dflash_cfg.get("selected_prefix_last_write_refine")
    if refine is not None:
        if refine.get("kind") != "exact_sender":
            raise ValueError(
                "DSpine expects selected_prefix_last_write_refine.kind == "
                f"'exact_sender', got {refine.get('kind')!r}"
            )
        found["dflash_config.selected_prefix_last_write_refine.head_topk"] = int(refine["head_topk"])
    if not found:
        raise ValueError(
            "DSpine checkpoint config names no refinement width: add "
            "code_bridge_second_pass {mode: serial_table, first_exact, head_topk} "
            "or dflash_config.selected_prefix_last_write_refine.head_topk"
        )
    values = set(found.values())
    if len(values) != 1:
        raise ValueError(f"DSpine refinement widths disagree: {found}")
    k = values.pop()
    if k & (k - 1) or not 2 <= k <= 64:
        raise ValueError(f"DSpine head_topk must be a power of two in [2, 64], got {k}")
    return k


class DSpineCodeBridge(nn.Module):
    """Per-layer adjacent-causal code write with fused next RMSNorm.

    Weights live under ``selected_prefix_bridge.*`` in the checkpoint:
    ``codebook`` [V, C], ``read_weight`` [C, H], ``early_read_weights``
    [L-1, C, H], ``feature_write_weights`` [L, H, R], ``gate`` / ``prev`` /
    ``predecessor_gate`` / ``out`` linears and two int64 version scalars.

    ``forward`` returns ``(next_norm(out), out)`` where ``out`` is the updated
    residual stream, mirroring the SGLang eager implementation bit-for-bit
    (same rounding points: fp32 math with bf16 stores).
    """

    def __init__(self, config) -> None:
        super().__init__()
        cfg = _dflash_config(config)
        if cfg.get("selected_prefix_terminal", True):
            raise ValueError("DSpine supports terminal-disabled code bridges only")
        if cfg.get("selected_prefix_feature_write_mode") != "independent":
            raise ValueError("DSpine requires independent learned writes")
        if cfg.get("selected_prefix_feature_writes") is not True:
            raise ValueError("DSpine requires selected_prefix_feature_writes = true")
        if not cfg.get("draft_block_causal"):
            raise ValueError("DSpine requires causal draft attention (draft_block_causal)")
        bridge = cfg.get("selected_prefix_code_bridge")
        if not bridge:
            raise ValueError("DSpine draft config lacks dflash_config.selected_prefix_code_bridge")
        self.hidden_size = int(config.hidden_size)
        self.code_size = int(bridge["code_size"])
        self.rank = int(bridge["rank"])
        self.eps = float(bridge["eps"])
        self.num_layers = int(config.num_hidden_layers)
        self.block_size = int(config.block_size)
        self.active_layers = frozenset(int(i) - 1 for i in cfg["selected_prefix_feature_layers"])
        if not self.active_layers <= set(range(self.num_layers)):
            raise ValueError(f"selected_prefix_feature_layers out of range: {sorted(self.active_layers)}")
        if self.num_layers - 1 not in self.active_layers:
            raise ValueError("DSpine refines the last layer's write, so the last layer must be active")
        self.prev = nn.Linear(self.code_size, self.rank, bias=False)
        self.gate = nn.Linear(self.code_size, self.rank, bias=True)
        self.predecessor_gate = nn.Linear(self.code_size, self.rank, bias=False)
        self.out = nn.Linear(self.rank, self.code_size, bias=False)
        self._code_scale = math.sqrt(self.code_size)
        self._rms_floor = torch.finfo(torch.float32).tiny
        self.register_buffer(
            "codebook",
            torch.empty(config.vocab_size, self.code_size, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "read_weight",
            torch.empty(self.code_size, self.hidden_size, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "early_read_weights",
            torch.empty(
                self.num_layers - 1,
                self.code_size,
                self.hidden_size,
                dtype=torch.bfloat16,
            ),
        )
        self.feature_write_weights = nn.Parameter(
            torch.empty(self.num_layers, self.hidden_size, self.rank, dtype=torch.bfloat16),
            requires_grad=False,
        )
        for name in ("independent_write_version", "terminal_disabled_version"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))
        self.register_buffer("packed_weight", None, persistent=False)
        self._loaded: set[str] = set()
        self._ready = False

    def load_weight(self, name: str, tensor: torch.Tensor) -> None:
        destinations = dict(self.named_parameters()) | dict(self.named_buffers())
        if name not in destinations or name == "packed_weight":
            raise ValueError(f"Unexpected DSpine code-bridge weight: {name}")
        dest = destinations[name]
        if tuple(dest.shape) != tuple(tensor.shape):
            raise ValueError(
                f"DSpine code-bridge shape mismatch: {name}: " f"{tuple(tensor.shape)} vs {tuple(dest.shape)}"
            )
        dest.data.copy_(tensor)
        self._loaded.add(name)

    def begin_load(self) -> None:
        self._loaded.clear()
        self._ready = False

    def prepare(self) -> None:
        expected = (set(dict(self.named_parameters())) | set(dict(self.named_buffers()))) - {"packed_weight"}
        missing = expected - self._loaded
        if missing:
            raise ValueError(f"Missing DSpine code-bridge tensors: {sorted(missing)}")
        if self.out.weight.count_nonzero().item():
            raise ValueError("Terminal-disabled checkpoint contains a nonzero terminal projection")
        for name in ("independent_write_version", "terminal_disabled_version"):
            if getattr(self, name).item() != 1:
                raise ValueError(f"Unsupported DSpine {name}")
        self.packed_weight = torch.cat(
            (self.gate.weight, self.predecessor_gate.weight, self.prev.weight), dim=0
        ).contiguous()
        self._ready = True

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        return self.read_weight if layer == self.num_layers - 1 else self.early_read_weights[layer]

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        input_ids: torch.Tensor,
        layer: int,
        next_norm: RMSNorm,
        stash: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One write after ``layer``.

        Returns ``(next_norm(out), out)``. When ``stash`` is a tuple of three
        preallocated buffers (raw [>=m, H] bf16, rms [>=m] fp32, packed
        [>=m, 3R] bf16) the pre-write state is copied into their first m rows
        for the refinement pass.
        """
        if not self._ready:
            raise RuntimeError("DSpine code-bridge weights were not prepared")
        t, r = self.block_size, self.rank
        m = hidden.shape[0]
        raw = hidden if residual is None else hidden + residual
        f = raw.float()
        rms = f.square().mean(-1).clamp_min(self._rms_floor).sqrt()
        normalized = (f / (rms[:, None] + self.eps)).to(raw.dtype)
        projected = F.linear(normalized, self.layer_read_weight(layer)).float()
        codes = self._code_scale * projected / (projected.norm(dim=-1, keepdim=True) + self.eps)
        anchors = input_ids.view(-1, t)[:, 0].long()
        anchor_codes = self._code_scale * F.embedding(anchors, self.codebook).float()
        codes = torch.cat((anchor_codes[:, None, :], codes.view(-1, t, self.code_size)[:, 1:]), dim=1).view(
            -1, self.code_size
        )
        packed = F.linear(codes.to(raw.dtype), self.packed_weight)
        if stash is not None:
            stash_raw, stash_rms, stash_packed = stash
            stash_raw[:m].copy_(raw)
            stash_rms[:m].copy_(rms)
            stash_packed[:m].copy_(packed)
        q, s, v = packed.float().view(-1, t, 3 * r).split(r, dim=-1)
        gate = torch.sigmoid(q[:, 1:] + s[:, :-1] + self.gate.bias.float())
        message = torch.cat((torch.zeros_like(q[:, :1]), gate * v[:, :-1]), dim=1).to(raw.dtype)
        delta = F.linear(message.view(-1, r), self.feature_write_weights[layer]).float()
        # Anchor rows already carry a zero message row, so no extra mask is needed.
        out = (f + rms[:, None] * delta).to(raw.dtype)
        x = out.float()
        y = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + next_norm.variance_epsilon)).to(raw.dtype)
        normed = (y.float() * next_norm.weight.float()).to(raw.dtype)
        return normed, out


class DSpineQwen3DecoderLayer(DFlashQwen3DecoderLayer):
    """DFlash layer whose input RMSNorm may already have been applied by a code write."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        input_normed: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_normed:
            if residual is None:
                raise RuntimeError("A pre-normed DSpine layer input needs its residual")
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DSpineQwen3Model(DFlashQwen3Model):
    """DFlash draft model with the DSpine code bridge.

    ``forward`` runs the DFlash layer stack and, after every active layer, a
    bridge write whose output feeds the next layer pre-normed. The last
    bridge's pre-write state (raw / rms / packed) is copied into the
    ``last_write_stash`` buffers for the refinement pass.

    Like ``DFlash2Qwen3Model`` this subclasses the decorated
    ``DFlashQwen3Model`` without re-applying ``@support_torch_compile``: the
    inherited ``__call__`` wrapper compiles this class's ``forward`` (which
    keeps the same parameter names as the parent signature).
    """

    decoder_layer_cls = DSpineQwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        config = vllm_config.speculative_config.draft_model_config.hf_config
        # Normalize to the upstream causality knob before the decoder layers
        # are built: DSpine checkpoints express block causality as
        # ``draft_block_causal``, while ``_dflash_layer_causal`` reads
        # ``dflash_config.causal`` (used by get_draft_attn_causal and friends).
        dflash_cfg = _dflash_config(config)
        if dflash_cfg.get("draft_block_causal") and "causal" not in dflash_cfg:
            dflash_cfg["causal"] = True
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )

        self.selected_prefix_bridge = DSpineCodeBridge(self.config)
        if self.selected_prefix_bridge.block_size != int(self.config.block_size):
            raise ValueError("DSpine bridge and draft block sizes differ")
        self.head_topk = resolve_dspine_head_topk(self.config, len(self.layers))

        # Precomputed per-layer wiring for forward: keeps the traced graph free
        # of len()/set-membership/ModuleList-indexing so dynamo can inline it.
        num_layers = len(self.layers)
        self._num_layers = num_layers
        self._post_write_norms: list[RMSNorm | None] = [
            (
                (self.layers[index + 1].input_layernorm if index < num_layers - 1 else self.norm)
                if index in self.selected_prefix_bridge.active_layers
                else None
            )
            for index in range(num_layers)
        ]

        max_rows = vllm_config.scheduler_config.max_num_seqs * self.selected_prefix_bridge.block_size
        h, r = self.config.hidden_size, self.selected_prefix_bridge.rank
        self.register_buffer(
            "stash_raw",
            torch.zeros(max_rows, h, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "stash_rms",
            torch.ones(max_rows, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "stash_packed",
            torch.zeros(max_rows, 3 * r, dtype=torch.bfloat16),
            persistent=False,
        )
        self._write_rows_stash = (
            self.stash_raw,
            self.stash_rms,
            self.stash_packed,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        bridge = self.selected_prefix_bridge
        hidden_states = input_embeds
        residual: torch.Tensor | None = None
        input_normed = False
        for index, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                input_normed=input_normed,
            )
            input_normed = False
            next_norm = self._post_write_norms[index]
            if next_norm is not None:
                hidden_states, residual = bridge(
                    hidden_states,
                    residual,
                    input_ids,
                    index,
                    next_norm,
                    stash=(self._write_rows_stash if index == self._num_layers - 1 else None),
                )
                input_normed = True
        if not input_normed:
            raise RuntimeError("DSpine's last layer must end with a (norm-fused) code write")
        return hidden_states


class DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DSpine draft head: DFlash backbone plus code bridge and serial tables."""

    model_cls = DSpineQwen3Model
    # DSpine checkpoints ship no lm_head / embed_tokens; both are shared from
    # the target model, like DFlash2.
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._dspine_tg: torch.Tensor | None = None
        self._dspine_tsv: torch.Tensor | None = None

    def build_dspine_serial_tables(self) -> None:
        """Build the vocabulary-wide greedy refinement tables (once, after lm_head share).

        TG [V, R] = bf16((LmHead * norm_w) @ W_last) and
        TSV [V, 2R] = bf16([G_p; A] @ (sqrt(C) * codebook)); both bf16 like the
        reference (fp32 accumulation happens at table time).
        """
        bridge = self.model.selected_prefix_bridge
        if not bridge._ready:
            raise RuntimeError("DSpine serial tables built before weights were prepared")
        lm_head_w = self.lm_head.weight
        if lm_head_w is None or lm_head_w.dtype != torch.bfloat16 or lm_head_w.dim() != 2 or lm_head_w.stride(1) != 1:
            raise ValueError("DSpine needs an unquantized row-contiguous BF16 lm_head")
        if lm_head_w.shape[0] != bridge.codebook.shape[0]:
            raise ValueError(f"DSpine: lm_head vocab {lm_head_w.shape[0]} != codebook " f"{bridge.codebook.shape[0]}")
        last = len(self.model.layers) - 1
        self._dspine_tg, self._dspine_tsv = build_serial_tables(
            lm_head_w.detach(),
            self.model.norm.weight.detach(),
            bridge.feature_write_weights[last].detach(),
            bridge.packed_weight[bridge.rank :].detach(),
            bridge.codebook,
        )
        logger.info(
            "DSpine serial tables built: tg %s tsv %s (head_topk=%d)",
            tuple(self._dspine_tg.shape),
            tuple(self._dspine_tsv.shape),
            self.model.head_topk,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load DFlash parameters plus ``selected_prefix_bridge.*`` tensors."""
        bridge = self.model.selected_prefix_bridge
        bridge.begin_load()

        def dflash_weights():
            for name, tensor in weights:
                short = name.removeprefix("model.")
                if short.startswith(BRIDGE_PREFIX):
                    bridge.load_weight(short[len(BRIDGE_PREFIX) :], tensor)
                    continue
                if short in VERSION_TENSORS:
                    if int(tensor.item()) != 1:
                        raise ValueError(f"Unsupported DSpine checkpoint version: {short}")
                    continue
                yield name, tensor

        super().load_weights(dflash_weights())
        bridge.prepare()
        logger.info(
            "Loaded DSpine draft: layers=%d active_layers=%s code_size=%d rank=%d head_topk=%d",
            len(self.model.layers),
            sorted(i + 1 for i in self.model.selected_prefix_bridge.active_layers),
            bridge.code_size,
            bridge.rank,
            self.model.head_topk,
        )


def build_serial_tables(
    lm_head_w: torch.Tensor,
    norm_w: torch.Tensor,
    write_w: torch.Tensor,
    pv_w: torch.Tensor,
    codebook: torch.Tensor,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(TG [V, R] bf16, TSV [V, 2R] bf16).

    lm_head_w [V, H], norm_w [H], write_w [H, R] (last layer's write weight),
    pv_w [2R, C] = [G_p; A], codebook [V, C].
    """
    v, h = lm_head_w.shape
    r = write_w.shape[1]
    c = codebook.shape[1]
    if (
        codebook.shape[0] != v
        or tuple(write_w.shape) != (h, r)
        or tuple(pv_w.shape) != (2 * r, c)
        or norm_w.shape != (h,)
    ):
        raise ValueError("build_serial_tables: shape contract violated")
    tg = torch.empty((v, r), device=lm_head_w.device, dtype=torch.bfloat16)
    tsv = torch.empty((v, 2 * r), device=lm_head_w.device, dtype=torch.bfloat16)
    nwf, wb, pvb, scale = (
        norm_w.float(),
        write_w.to(torch.bfloat16),
        pv_w.to(torch.bfloat16),
        float(c) ** 0.5,
    )
    for s in range(0, v, chunk):
        e = min(s + chunk, v)
        tg[s:e] = torch.matmul((lm_head_w[s:e].float() * nwf).to(torch.bfloat16), wb)
        tsv[s:e] = F.linear((scale * codebook[s:e].float()).to(torch.bfloat16), pvb)
    return tg, tsv


def dspine_serial_table(
    raw: torch.Tensor,
    norm_w: torch.Tensor,
    cand: torch.Tensor,
    input_ids: torch.Tensor,
    lm_head_w: torch.Tensor,
    tg: torch.Tensor,
    tsv: torch.Tensor,
    packed: torch.Tensor,
    rms: torch.Tensor,
    gate_b: torch.Tensor,
    block_size: int,
    row_chunk: int = 512,
) -> torch.Tensor:
    """Greedy refinement table -> idx int64 [n, K].

    ``idx[n, j]`` is the best candidate lane of proposal row n when the previous
    row chose lane j. Row n (= b*(T-1)+r) corresponds to block row r+1; the
    sender of the first proposal row is the anchor token. ``base`` scores
    candidate tokens against the pre-write state; the msg/g term adds the
    exact last-write delta. The final RMSNorm denominator is a positive scalar
    and is dropped (argmax-invariant).
    """
    t = int(block_size)
    m, h = raw.shape
    bs = m // t
    n, k = cand.shape
    r = tg.shape[1]
    raw_rows = raw.view(bs, t, h)[:, 1:, :].reshape(n, h)
    rms_rows = rms.view(bs, t)[:, 1:].reshape(n).float()
    q = packed.view(bs, t, -1)[:, 1:, :r].reshape(n, r)
    anchors = input_ids.view(bs, t)[:, 0].long()
    sid = torch.cat(
        (anchors.view(bs, 1, 1).expand(bs, 1, k), cand.view(bs, t - 1, k)[:, :-1, :]),
        dim=1,
    ).reshape(n, k)
    tsv_rows = tsv[sid.reshape(-1)].view(n, k, 2 * r).float()
    g = tg[cand.reshape(-1)].view(n, k, r).float()
    msg = torch.sigmoid(q.float()[:, None, :] + tsv_rows[..., :r] + gate_b.float()) * tsv_rows[..., r:]
    u = raw_rows.float() * norm_w.float()
    idx = torch.empty((n, k), dtype=torch.int64, device=raw.device)
    for s0 in range(0, n, row_chunk):
        e0 = min(n, s0 + row_chunk)
        lm_c = lm_head_w[cand[s0:e0].reshape(-1)].view(e0 - s0, k, h).float()
        base = torch.einsum("nh,nch->nc", u[s0:e0], lm_c)
        s = torch.einsum("njr,ncr->njc", msg[s0:e0], g[s0:e0])
        scores = base[:, None, :] + rms_rows[s0:e0, None, None] * s
        idx[s0:e0] = scores.argmax(dim=-1)
    return idx


def dspine_serial_chain(
    idx: torch.Tensor,
    cand: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Walk every request's greedy table. Returns int64 [bs*(T-1)]."""
    t = int(block_size)
    n, k = cand.shape
    bs = n // (t - 1)
    idx3 = idx.view(bs, t - 1, k)
    cand3 = cand.view(bs, t - 1, k)
    brow = torch.arange(bs, device=cand.device)
    j = idx3[:, 0, 0]
    out = torch.empty((bs, t - 1), dtype=torch.int64, device=cand.device)
    for row in range(t - 1):
        out[:, row] = cand3[brow, row, j]
        if row + 1 < t - 1:
            j = idx3[brow, row + 1, j]
    return out.view(-1)
