import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.llama_eagle3 import (
    Eagle3LlamaForCausalLM,
)
from vllm.model_executor.models.llama_eagle3 import (
    LlamaDecoderLayer as Eagle3LlamaDecoderLayer,
)
from vllm.model_executor.models.llama_eagle3 import (
    LlamaModel as Eagle3LlamaModel,
)
from vllm.model_executor.models.utils import get_draft_quant_config, maybe_prefix


def _linear(inp, out, vc, qc, pfx):
    return ReplicatedLinear(
        input_size=inp,
        output_size=out,
        bias=False,
        params_dtype=vc.model_config.dtype,
        quant_config=qc,
        prefix=pfx,
        return_bias=False,
    )


class PreVwnLayerV1(nn.Module):
    def __init__(self, vllm_config, prefix="", config=None, quant_config=None):
        super().__init__()
        cfg = config or vllm_config.model_config.hf_config
        hs, m, r = cfg.hidden_size, getattr(cfg, "vwn_m", 1), getattr(cfg, "vwn_r", 1)
        wd = int(hs * r)
        self.m, self.hidden_size, self.wider_dim = m, hs, wd
        self.input_layernorm = RMSNorm(hs, eps=cfg.rms_norm_eps)
        self.hidden_norm = RMSNorm(hs, eps=cfg.rms_norm_eps)
        self.fc = _linear(2 * hs, hs, vllm_config, quant_config, maybe_prefix(prefix, "fc"))
        self.upward = _linear(hs // m, wd // m, vllm_config, quant_config, maybe_prefix(prefix, "upward"))

    def forward(self, embeds, hidden_states):
        x = self.fc(torch.cat([self.input_layernorm(embeds), self.hidden_norm(hidden_states)], dim=-1))
        return self.upward(x.view(-1, self.hidden_size // self.m)).view(-1, self.wider_dim)


class VwnLlamaDecoderLayer(Eagle3LlamaDecoderLayer):
    def __init__(self, vllm_config, prefix="", config=None, layer_idx=0):
        super().__init__(vllm_config, prefix=prefix, config=config, layer_idx=layer_idx)
        cfg = config or vllm_config.model_config.hf_config
        qc = self.get_quant_config(vllm_config)
        m, r = getattr(cfg, "vwn_m", 1), getattr(cfg, "vwn_r", 1)
        hs, wd = self.hidden_size, int(self.hidden_size * r)
        self.m, self.wider_dim, self.layer_idx = m, wd, layer_idx

        if layer_idx == 0:
            self.self_attn.qkv_proj = QKVParallelLinear(
                hs,
                self.self_attn.head_dim,
                self.self_attn.total_num_heads,
                self.self_attn.total_num_kv_heads,
                bias=getattr(cfg, "attention_bias", False),
                quant_config=qc,
                prefix=maybe_prefix(prefix, "self_attn.qkv_proj"),
            )

        mp = maybe_prefix
        self.pre_vwn_layer = PreVwnLayerV1(vllm_config, mp(prefix, "layers.pre_vwn_layer"), cfg, qc)
        self.downward_and_forgot = _linear(wd // m, (hs + wd) // m, vllm_config, qc, mp(prefix, "downward_and_forgot"))
        self.pre_attention_layernorm = RMSNorm(hs, eps=cfg.rms_norm_eps)
        self.upward_after_attn = _linear(hs // m, wd // m, vllm_config, qc, mp(prefix, "upward_after_attn"))
        self.downward_and_forgot_after_attn = _linear(
            wd // m, (hs + wd) // m, vllm_config, qc, mp(prefix, "downward_and_forgot_after_attn")
        )
        self.post_attention_layernorm = RMSNorm(hs, eps=cfg.rms_norm_eps)
        self.upward_after_mlp = _linear(hs // m, wd // m, vllm_config, qc, mp(prefix, "upward_after_mlp"))
        self.downward = _linear(wd // m, hs // m, vllm_config, qc, mp(prefix, "downward"))

    def forward(self, positions, embeds, hidden_states, residual):
        if self.layer_idx == 0:
            hs, wd, m = self.hidden_size, self.wider_dim, self.m
            wider = self.pre_vwn_layer(embeds, hidden_states)
            # Attention
            out = self.downward_and_forgot(wider.view(-1, wd // m)).view(-1, hs + wd)
            hidden, res = out.split([hs, wd], dim=-1)
            hidden = self.self_attn(positions=positions, hidden_states=self.pre_attention_layernorm(hidden))
            wider = self.upward_after_attn(hidden.view(-1, hs // m)).view(-1, wd) + res
            # MLP
            out = self.downward_and_forgot_after_attn(wider.view(-1, wd // m)).view(-1, hs + wd)
            hidden, res = out.split([hs, wd], dim=-1)
            wider = (
                self.upward_after_mlp(self.mlp(self.post_attention_layernorm(hidden)).view(-1, hs // m)).view(-1, wd)
                + res
            )
            # Downward
            hidden_states = self.downward(wider.view(-1, wd // m)).view(-1, hs)
        return hidden_states, residual


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": -1, "hidden_states": 0, "input_embeds": 0})
class VwnLlamaModel(Eagle3LlamaModel):
    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        eagle_config = getattr(self.config, "eagle_config", None)
        if eagle_config is not None and "use_aux_hidden_state" in eagle_config:
            self.use_aux_hidden_state = eagle_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True
        self.norm_before_fc = getattr(self.config, "norm_before_fc", False)

        vc = get_current_vllm_config()
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                VwnLlamaDecoderLayer(vc, maybe_prefix(prefix, f"layers.{i + start_layer_id}"), self.config, layer_idx=i)
                for i in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * 3
            else:
                fc_input_size = self.config.hidden_size * 3
            if self.norm_before_fc:
                self.input_norm = RMSNorm(
                    fc_input_size,
                    eps=self.config.rms_norm_eps,
                )
            else:
                self.input_norm = None

            self.fc_norm = None
            self.num_aux_hidden_states = 3
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def forward(self, input_ids, positions, hidden_states, input_embeds=None):
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions, embeds=input_embeds, hidden_states=hidden_states, residual=residual
            )
        return self.norm(hidden_states, residual), hidden_states


class Eagle3VwnLlamaForCausalLM(Eagle3LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size

        n = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.config.target_layer_count = n

        self.model = VwnLlamaModel(vllm_config=vllm_config, prefix="model", start_layer_id=n)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
        )
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

        self.use_parallel_drafting = vllm_config.speculative_config.parallel_drafting

        if self.use_parallel_drafting:
            self.register_buffer(
                "mask_hidden",
                torch.zeros(
                    1,
                    (3 if self.model.use_aux_hidden_state else 1) * self.config.hidden_size,
                ),
                persistent=False,
            )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                f"Expected logits to have shape (*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        # combine multiple auxiliary hidden states returned by eagle3

        if self.model.norm_before_fc:
            hidden_states = self.model.input_norm(hidden_states)

        # `norm_before_fc` adds a single RMSNorm before the FC layer, whereas `fc_norm`
        # applies separate RMSNorms to each chunk of the hidden states.
        if self.model.fc_norm is not None:
            chunks = hidden_states.chunk(self.model.num_aux_hidden_states, dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.model.fc_norm, chunks)],
                dim=-1,
            )

        return self.model.fc(hidden_states)
