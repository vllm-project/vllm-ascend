# DSpark model classes for msModelSlim DeepSeek-V4 quant adapter (path A).
# Ported from DeepSeek-V4-Flash-DSpark inference/model.py (dspark-ref/inference_model.py
# L750-936), adapted to msModelSlim's msmodelslim/model/deepseek_v4/model.py base:
#   - n_layers -> num_hidden_layers
#   - reuse msModelSlim Block/Attention/RMSNorm/Linear/MoE (import from .model)
#   - heads kept simple (markov/confidence are EXCLUDED from quant -> bf16; only need
#     to exist as modules so the scaffold loads + the calib forward runs)
# Goal: provide the modules + forward msModelSlim's calib loop drives for the DSpark
# draft (3 mtp stages, main_proj over target layers [40,41,42], semi-AR head).
import torch
import torch.nn.functional as F
from torch import nn

from .model import Attention, Block, RMSNorm, apply_rotary_emb, sparse_attn
# NOTE: msModelSlim runs the model in bf16 for calibration; activation quant is done
# by the framework hooks, NOT in the forward -> do NOT call act_quant here (it is
# commented out in model.py). No self.scale_fmt/scale_dtype on Attention either.


def get_dspark_topk_idxs(window_size, bsz, block_size, start_pos):
    assert start_pos > 0
    matrix = torch.cat([torch.arange(min(window_size, start_pos + 1)),
                        window_size + torch.arange(block_size)])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(Attention):
    """MLA where CONTEXT KV comes from main_x (the projected target-layer hidden);
    the block's own KV/Q come from x. compress_ratio==0 (pure sliding window).
    Ref inference_model.py L750-792."""

    def forward(self, x, start_pos, main_x):
        # CALIBRATION forward (bf16, no act_quant). Goal: route BOTH main_x and the
        # block x through wkv (context + block KV), x through wq_a/wq_b, output through
        # wo_a/wo_b, so all linears see representative activations. Mirrors msModelSlim
        # Attention.forward style (simplified, always-prefill cache write).
        bsz, blk, _ = x.size()
        win = self.window_size
        rd = self.rope_head_dim
        ctx = main_x.size(1)

        # context KV from main_x -> window cache
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], self.freqs_cis[start_pos:start_pos + ctx])
        self.kv_cache[:bsz, :min(win, ctx)] = main_kv[:, -win:]

        # block q/kv from x
        fc = self.freqs_cis[start_pos + ctx:start_pos + ctx + blk]
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], fc)
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], fc)

        kv = torch.cat([self.kv_cache[:bsz], kv], dim=1)
        topk_idxs = get_dspark_topk_idxs(win, bsz, blk, max(start_pos, 1))
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], fc, True)

        o = o.view(bsz, blk, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


class DSparkMarkovHead(nn.Module):
    """Eq.5 Markov head. EXCLUDED from quant (bf16). markov_w1=embedding, markov_w2=head."""

    def __init__(self, vocab_size, dim, markov_rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def forward(self, token_ids):
        embed = self.markov_w1(token_ids)
        return self.markov_w2(embed), embed


class DSparkConfidenceHead(nn.Module):
    """Eq.7 confidence head (raw logit, fp32). EXCLUDED from quant (bf16)."""

    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False)

    def forward(self, hidden, markov_embed):
        h = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(h.float()).squeeze(-1)


class DSparkBlock(Block):
    """DSpark draft stage (mtp.* namespace). Extends msModelSlim Block but swaps in
    DSparkAttention and adds stage-special modules: main_proj/main_norm on stage 0,
    norm/markov_head/confidence_head/hc_head_* on the last stage.
    Ref inference_model.py DSparkBlock L818-874."""

    def __init__(self, layer_id, args):
        super().__init__(layer_id, args)
        self.attn = DSparkAttention(layer_id, args)      # swap MLA -> DSpark MLA
        self.dim = args.dim
        stage_id = layer_id - args.num_hidden_layers
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        n_target = len(args.dspark_target_layer_ids)
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        if stage_id == 0:
            self.main_proj = nn.Linear(args.dim * n_target, args.dim, bias=False)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dim, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)
            origin = torch.get_default_dtype(); torch.set_default_dtype(torch.float32)
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
            torch.set_default_dtype(origin)
        # shared from main model (set during scaffold wrap): embed, head
        object.__setattr__(self, "embed", None)
        object.__setattr__(self, "head", None)

    def forward(self, x, start_pos, input_ids, main_x):
        # During calib only the attn/ffn weights matter; reuse Block.forward path but
        # route main_x into DSparkAttention. Block.forward calls self.attn(x, start_pos)
        # -> we need (x, start_pos, main_x); override the attn call here.
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, main_x)              # DSparkAttention: KV from main_x
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x

    def forward_embed(self, main_hidden, input_ids):
        """stage-0: main_x = main_norm(main_proj(concat of target layers' mean-pooled
        hidden)); draft block = [anchor, noise...] embedded + hc-expanded. Ref L851-858."""
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)
        draft_ids[:, 0] = input_ids if input_ids.ndim == 1 else input_ids[:, -1]
        x = self.embed(draft_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x
