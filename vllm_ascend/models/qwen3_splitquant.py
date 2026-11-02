"""Qwen3 SplitQuant export with independent projection transforms (TP=PP=1).

Do not merge Q/K/V or gate/up: each exported projection owns its activation
clipping and rotation. All seven linears per decoder use the fused NPU op.
"""
from collections.abc import Iterable
import os

import torch
from torch import nn
import torch.nn.functional as F

from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import (Qwen3Attention, Qwen3DecoderLayer,
                                             Qwen3ForCausalLM)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory, make_layers, extract_layer_index
from vllm.model_executor.models.interfaces import SupportsPP, SupportsLoRA, MixtureOfExperts
from vllm.model_executor.models.utils import maybe_prefix


class SplitQuantAttention(Qwen3Attention):
    def __init__(self, *, config, cache_config, quant_config, prefix):
        # Reuse upstream attention/cache/rotary construction. Replace only the
        # projection modules; the temporary unquantized tensors are released.
        super().__init__(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            cache_config=cache_config, quant_config=None, prefix=prefix)
        del self.qkv_proj
        del self.o_proj
        from vllm.config import get_current_vllm_config
        context = get_current_vllm_config().compilation_config.static_forward_context
        context.pop(f"{prefix}.qkv_proj", None)
        context.pop(f"{prefix}.o_proj", None)
        for name, size in (("q_proj", self.q_size),
                           ("k_proj", self.kv_size),
                           ("v_proj", self.kv_size)):
            setattr(self, name, ColumnParallelLinear(
                config.hidden_size, size,
                bias=getattr(config, "attention_bias", False),
                quant_config=quant_config, prefix=f"{prefix}.{name}"))
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size,
            bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")

    def forward(self, positions, hidden_states):
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class SplitQuantMLP(nn.Module):
    def __init__(self, *, config, quant_config, prefix):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("SplitQuant Qwen3 requires silu")
        for name in ("gate_proj", "up_proj"):
            setattr(self, name, ColumnParallelLinear(
                config.hidden_size, config.intermediate_size, bias=False,
                quant_config=quant_config, prefix=f"{prefix}.{name}"))
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        output, _ = self.down_proj(self.act_fn(torch.cat((gate, up), dim=-1)))
        return output


class SplitQuantDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, cache_config=None, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = SplitQuantAttention(
            config=config, cache_config=cache_config, quant_config=quant_config,
            prefix=f"{prefix}.self_attn")
        self.mlp = SplitQuantMLP(config=config, quant_config=quant_config,
                                 prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class SplitQuantQwen3ForCausalLM(Qwen3ForCausalLM):
    packed_modules_mapping = {}

    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        if get_tensor_model_parallel_world_size() != 1 or not (
                get_pp_group().is_first_rank and get_pp_group().is_last_rank):
            raise NotImplementedError("SplitQuant Qwen3 currently requires TP=PP=1")
        self.config = vllm_config.model_config.hf_config
        self.lora_config = vllm_config.lora_config
        self.quant_config = vllm_config.quant_config
        if self.quant_config is None:
            raise ValueError("SplitQuant Qwen3 requires quantization='ascend'")
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"),
                                decoder_layer_type=SplitQuantDecoderLayer)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size,
            quant_config=self.quant_config, prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        if os.getenv("SPLITQUANT_DENSE") == "1":
            from vllm_ascend.quantization.splitquant_dense import install
            install(self)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        helper_markers = (".ln_trans.", ".o_trans.", ".vcache_trans.",
                          ".up_gate_trans.", ".down_trans.")
        aliases = ((".self_attn._parent_input_layernorm.weight", ".input_layernorm.weight"),
                   (".mlp._parent_post_attention_layernorm.weight", ".post_attention_layernorm.weight"))
        for source_name, value in weights:
            name = source_name
            for old, new in aliases:
                name = name.replace(old, new)
            if name not in params:
                if any(marker in source_name for marker in helper_markers):
                    continue
                raise KeyError(f"Unexpected SplitQuant checkpoint tensor: {source_name}")
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, value)
            loaded.add(name)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"SplitQuant checkpoint did not initialize: {sorted(missing)}")
        self._splitquant_loaded_parameters = len(loaded)
        return loaded


def _sq_fake_quant(x, clip_max, clip_min, group=128):
    shape = x.shape
    y = x.reshape(-1, shape[-1] // group, group)
    hi = torch.maximum(y.amax(-1, keepdim=True), torch.zeros_like(y[..., :1]))
    lo = torch.minimum(y.amin(-1, keepdim=True), torch.zeros_like(y[..., :1]))
    hi = hi * clip_max.to(x).reshape(1, -1, 1)
    lo = lo * clip_min.to(x).reshape(1, -1, 1)
    scale = torch.maximum(hi, lo.abs()) / 7
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (torch.clamp(torch.round(y / scale), -8, 7) * scale).reshape(shape)


class SplitQuantMoeExperts(nn.Module):
    """Reference-correct 3-D expert path used while the fused MoE kernel is absent."""
    def __init__(self, config, prefix):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden = config.hidden_size
        self.intermediate = config.moe_intermediate_size
        self.top_k = config.num_experts_per_tok
        self.group = 128
        self.act = nn.SiLU()
        # Parameters are intentionally plain tensors: the custom loader fills them
        # directly and Ascend's linear quantizer never attempts to fuse 3-D weights.
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2*self.intermediate, self.hidden, dtype=torch.int8), requires_grad=False)
        self.gate_up_proj_scale = nn.Parameter(torch.empty(self.num_experts, 2*self.intermediate, self.hidden//self.group), requires_grad=False)
        self.gate_up_proj_offset = nn.Parameter(torch.empty_like(self.gate_up_proj_scale), requires_grad=False)
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden, self.intermediate, dtype=torch.int8), requires_grad=False)
        self.down_proj_scale = nn.Parameter(torch.empty(self.num_experts, self.hidden, self.intermediate//self.group), requires_grad=False)
        self.down_proj_offset = nn.Parameter(torch.empty_like(self.down_proj_scale), requires_grad=False)
        for name, n in (("activation_clip_max", self.hidden//self.group), ("activation_clip_min", self.hidden//self.group),
                        ("hidden_activation_clip_max", self.intermediate//self.group), ("hidden_activation_clip_min", self.intermediate//self.group)):
            setattr(self, name, nn.Parameter(torch.empty(n), requires_grad=False))
        self.input_group_trans = nn.Parameter(torch.empty(self.hidden//self.group, self.group, self.group), requires_grad=False)
        self.down_group_trans = nn.Parameter(torch.empty(self.intermediate//self.group, self.group, self.group), requires_grad=False)
        self._catlass_weights_prepared = False

    @torch.no_grad()
    def prepare_catlass(self):
        """Compact expert W4 containers into the packed Catlass ABI in place.

        The checkpoint uses one signed byte per 4-bit code. Keeping a second
        packed copy for all 48 layers costs roughly another 12 GiB. After
        checkpoint loading the opt-in Catlass route owns the packed bytes, so
        replace the source Parameter storage and retain a non-persistent alias
        instead of duplicating every expert tensor.
        """
        if self._catlass_weights_prepared:
            return
        from vllm_ascend.quantization.w4a4_splitquant import load_splitquant_operator
        load_splitquant_operator()
        def pack(weight):
            codes = weight.to(torch.int16)
            if bool(((codes < -8) | (codes > 7)).any()):
                raise ValueError("expert weight contains values outside signed INT4")
            return ((codes[..., 0::2] & 15) |
                    ((codes[..., 1::2] & 15) << 4)).to(torch.uint8).contiguous()
        gate_up = pack(self.gate_up_proj)
        self.gate_up_proj.data = gate_up
        self.register_buffer("catlass_gate_up_packed", self.gate_up_proj.data,
                             persistent=False)
        # The dense host cache keys metadata by TensorImpl identity and retains
        # it. Recreating slices/casts on every call grows that cache without
        # bound. Keep exactly one FP32 metadata pair per expert for this model.
        self._catlass_gate_metadata = [
            (self.gate_up_proj_scale[e].float().contiguous(),
             self.gate_up_proj_offset[e].float().contiguous())
            for e in range(self.num_experts)]
        self.register_buffer("catlass_clip_max", self.activation_clip_max.float().contiguous(), persistent=False)
        self.register_buffer("catlass_clip_min", self.activation_clip_min.float().contiguous(), persistent=False)
        if os.getenv("SPLITQUANT_MOE") == "1":
            from vllm_ascend.quantization.splitquant_moe import prepare
            prepare(self)
        self._catlass_weights_prepared = True

    def _catlass_linear(self, x, expert, down=False):
        import torch_npu  # noqa: F401
        if not hasattr(self, "catlass_gate_up_packed"):
            self.prepare_catlass()
        if down:
            raise NotImplementedError("down requires quantization before its group transform")
        else:
            w = self.catlass_gate_up_packed[expert]
            if w.dim() == 4 and hasattr(self, "_prepared"):
                from vllm_ascend.quantization.splitquant_moe_native import unpack_up_nz
                w = unpack_up_nz(w)
            if w.dim() == 3 and hasattr(self, "_prepared"):
                w = w.contiguous().view(2*self.intermediate, self.hidden//2)
            ws, wo = self._catlass_gate_metadata[expert]
            cp, cn = self.catlass_clip_max, self.catlass_clip_min
        # The input is already in the reference path's required transform
        # state.  Passing no group transform avoids applying it twice.
        from vllm_ascend.quantization.w4a4_splitquant import splitquant_operator_name
        op_name = splitquant_operator_name()
        result = getattr(torch.ops.npu, op_name)(
            x.contiguous(), w.contiguous(), ws.float().contiguous(),
            wo.float().contiguous(), cp.float().contiguous(),
            cn.float().contiguous(), None, self.group).reshape(x.shape[0], -1)
        self.catlass_fused_operator_name = op_name
        self.catlass_fused_calls = getattr(self, "catlass_fused_calls", 0) + 1
        return result

    @staticmethod
    def _linear(x, w, scale, offset, group):
        if torch.count_nonzero(offset).item():
            raise ValueError("30B SplitQuant expert path requires zero weight offsets")
        import torch_npu
        return torch_npu.npu_weight_quant_batchmatmul(
            x=x.reshape(-1, x.shape[-1]),
            weight=w.transpose(-1, -2).contiguous(),
            antiquant_scale=scale.transpose(-1, -2).contiguous().to(x.dtype),
            antiquant_offset=None,
            antiquant_group_size=group,
        ).reshape(*x.shape[:-1], w.shape[-2])

    def forward(self, x, indices, weights):
        if os.getenv("SPLITQUANT_MOE") == "1":
            from vllm_ascend.quantization.splitquant_moe import forward
            return forward(self, x, indices, weights)
        return self._forward_reference(x, indices, weights)

    def _forward_reference(self, x, indices, weights):
        out = torch.zeros_like(x)
        use_catlass = self._catlass_weights_prepared
        for e in torch.unique(indices).tolist():
            tok, pos = torch.where(indices == e)
            xe = x[tok]
            if use_catlass:
                # Catlass performs the activation quantization inside the
                # dense ABI, so pass the transformed FP16/BF16 token directly.
                gu = self._catlass_linear(xe, e, down=False)
            else:
                xe = _sq_fake_quant(xe, self.activation_clip_max, self.activation_clip_min, self.group)
                gu = self._linear(xe, self.gate_up_proj[e], self.gate_up_proj_scale[e], self.gate_up_proj_offset[e], self.group)
            h = self.act(gu[:, :self.intermediate]) * gu[:, self.intermediate:]
            h = _sq_fake_quant(h, self.hidden_activation_clip_max, self.hidden_activation_clip_min, self.group)
            h = torch.einsum('mgk,gkl->mgl', h.reshape(-1, self.intermediate//self.group, self.group), self.down_group_trans.to(x)).reshape_as(h)
            # The 30B contract applies down_group_trans after hidden fake
            # quantization.  The current dense ABI quantizes its input again,
            # so keep this final projection on the exact reference call until
            # the prequantized activation kernel is available.
            y = self._linear(h, self.down_proj[e], self.down_proj_scale[e], self.down_proj_offset[e], self.group)
            out.index_add_(0, tok, y * weights[tok, pos, None].to(y))
        return out


class SplitQuantMoeBlock(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False, quant_config=None, prefix=f"{prefix}.gate")
        self.experts = SplitQuantMoeExperts(config, f"{prefix}.experts")
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
    def forward(self, x):
        orig = x.shape; x = x.reshape(-1, orig[-1])
        transform = self.experts.input_group_trans.to(x)
        x = torch.einsum('mgk,gkl->mgl', x.reshape(-1, x.shape[-1]//128, 128), transform).reshape_as(x)
        logits, _ = self.gate(x)
        # The fused router is the validated default for the SplitQuant MoE
        # path.  Keep an explicit ``=0`` escape hatch for diagnostics.
        router_enabled = (os.getenv("SPLITQUANT_DENSE") == "1"
                          and os.getenv("SPLITQUANT_MOE") == "1"
                          and os.getenv("SPLITQUANT_FUSED_ROUTER", "1") == "1")
        if router_enabled and x.dtype == torch.bfloat16 and logits.shape[1] == 128 and self.top_k == 8 and 1 <= x.shape[0] <= 32:
            from vllm_ascend.quantization.splitquant_router import route
            weights, indices = route(logits, self.top_k, self.norm_topk_prob)
            self.experts.router_calls = getattr(self.experts, "router_calls", 0) + 1
        else:
            if router_enabled:
                self.experts.router_fallback_calls = getattr(self.experts, "router_fallback_calls", 0) + 1
            probs = F.softmax(logits.float(), dim=-1)
            weights, indices = torch.topk(probs, self.top_k, dim=-1)
            if self.norm_topk_prob: weights = weights / weights.sum(-1, keepdim=True)
        return self.experts(x, indices, weights.to(x.dtype)).reshape(orig)


class SplitQuantMoeAttention(Qwen3MoeAttention):
    def __init__(self, *, config, cache_config, quant_config, prefix):
        super().__init__(hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
                         num_kv_heads=config.num_key_value_heads, rope_theta=config.rope_theta,
                         rope_scaling=config.rope_scaling, max_position_embeddings=config.max_position_embeddings,
                         rms_norm_eps=config.rms_norm_eps, qkv_bias=False, head_dim=config.head_dim,
                         cache_config=cache_config, quant_config=None, prefix=prefix)
        del self.qkv_proj; del self.o_proj
        from vllm.config import get_current_vllm_config
        context = get_current_vllm_config().compilation_config.static_forward_context
        context.pop(f"{prefix}.qkv_proj", None)
        context.pop(f"{prefix}.o_proj", None)
        for n,s in (("q_proj",self.q_size),("k_proj",self.kv_size),("v_proj",self.kv_size)):
            setattr(self,n,ColumnParallelLinear(config.hidden_size,s,bias=False,quant_config=quant_config,prefix=f"{prefix}.{n}"))
        self.o_proj = RowParallelLinear(self.total_num_heads*self.head_dim, config.hidden_size,bias=False,quant_config=quant_config,prefix=f"{prefix}.o_proj")
    def forward(self, positions, hidden_states):
        q,_=self.q_proj(hidden_states); k,_=self.k_proj(hidden_states); v,_=self.v_proj(hidden_states)
        q=self.q_norm(q.reshape(-1,self.num_heads,self.head_dim)).reshape(q.shape); k=self.k_norm(k.reshape(-1,self.num_kv_heads,self.head_dim)).reshape(k.shape)
        q,k=self.rotary_emb(positions,q,k); y=self.attn(q,k,v); y,_=self.o_proj(y); return y


class SplitQuantMoeLayer(nn.Module):
    def __init__(self, vllm_config, prefix):
        super().__init__(); c=vllm_config.model_config.hf_text_config
        self.self_attn=SplitQuantMoeAttention(config=c,cache_config=vllm_config.cache_config,quant_config=vllm_config.quant_config,prefix=f"{prefix}.self_attn")
        self.mlp=SplitQuantMoeBlock(c,f"{prefix}.mlp")
        self.input_layernorm=RMSNorm(c.hidden_size,eps=c.rms_norm_eps); self.post_attention_layernorm=RMSNorm(c.hidden_size,eps=c.rms_norm_eps)
    def forward(self, positions, hidden_states, residual):
        if residual is None: residual=hidden_states; hidden_states=self.input_layernorm(hidden_states)
        else: hidden_states,residual=self.input_layernorm(hidden_states,residual)
        hidden_states=self.self_attn(positions,hidden_states); hidden_states,residual=self.post_attention_layernorm(hidden_states,residual); return self.mlp(hidden_states), residual


class SplitQuantQwen3MoeModel(nn.Module):
    def __init__(self,vllm_config,prefix=""):
        super().__init__(); c=vllm_config.model_config.hf_text_config; self.config=c
        self.embed_tokens=VocabParallelEmbedding(c.vocab_size,c.hidden_size,quant_config=None,prefix=f"{prefix}.embed_tokens")
        self.layers=nn.ModuleList([SplitQuantMoeLayer(vllm_config,f"{prefix}.layers.{i}") for i in range(c.num_hidden_layers)])
        self.norm=RMSNorm(c.hidden_size,eps=c.rms_norm_eps); self.make_empty_intermediate_tensors=make_empty_intermediate_tensors_factory(["hidden_states","residual"],c.hidden_size)
    def get_input_embeddings(self, ids): return self.embed_tokens(ids)
    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        h=inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids); r=None
        for l in self.layers: h,r=l(positions,h,r)
        return self.norm(h,r)[0]


class SplitQuantQwen3MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA, MixtureOfExperts):
    packed_modules_mapping={}; fall_back_to_pt_during_load=False
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(); c=vllm_config.model_config.hf_text_config; self.config=c; self.quant_config=vllm_config.quant_config
        self.model=SplitQuantQwen3MoeModel(vllm_config,maybe_prefix(prefix,"model")); self.lm_head=ParallelLMHead(c.vocab_size,c.hidden_size,quant_config=None,prefix=maybe_prefix(prefix,"lm_head")); self.logits_processor=LogitsProcessor(c.vocab_size); self.make_empty_intermediate_tensors=self.model.make_empty_intermediate_tensors
        if os.getenv("SPLITQUANT_DENSE") == "1":
            import types
            from vllm_ascend.quantization.splitquant_dense import attention_forward
            for layer in self.model.layers:
                layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)

    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None): return self.model(input_ids,positions,intermediate_tensors,inputs_embeds)
    def compute_logits(self,hidden_states): return self.logits_processor(self.lm_head,hidden_states)
    def get_input_embeddings(self,input_ids): return self.model.get_input_embeddings(input_ids)
    def load_weights(self,weights):
        p=dict(self.named_parameters()); loaded=set()
        for n,v in weights:
            if n not in p:
                if any(x in n for x in ("_moe_in_trans","rotary_emb.inv_freq","mtp.")): continue
                raise KeyError(f"Unexpected SplitQuant 30B tensor: {n}")
            p[n].data.copy_(v.to(device=p[n].device,dtype=p[n].dtype)); loaded.add(n)
        miss=set(p)-loaded
        if miss: raise ValueError(f"SplitQuant 30B missing tensors: {sorted(miss)[:8]}")
        self._splitquant_loaded_parameters=len(loaded)
        if os.getenv("SPLITQUANT_MOE_CATLASS", "0") == "1":
            for layer_index, layer in enumerate(self.model.layers):
                layer.mlp.experts.prepare_catlass()
                if os.getenv("SPLITQUANT_MOE") == "1":
                    # Repacking retains a second gate container. Release retired
                    # load-time temporaries before AICPU needs its own HBM heap.
                    torch.npu.synchronize()
                    torch.npu.empty_cache()
                    if os.getenv("SPLITQUANT_MOE_PREPARE_TRACE") == "1":
                        print("MOE_PREPARED", layer_index,
                              "allocated", torch.npu.memory_allocated(),
                              "reserved", torch.npu.memory_reserved(), flush=True)
        return loaded

