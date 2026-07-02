# SPDX-License-Identifier: Apache-2.0
"""DSpark draft model for DeepSeek-V4 on Ascend NPU.

Mirrors the upstream vllm PR #46995 reference (vllm/models/deepseek_v4/nvidia/dspark.py).
The DSpark draft is **NOT** the serial MTP path even though its weights live
under the target checkpoint's ``mtp.*`` namespace — it's a parallel block
drafter with a Markov head and a confidence head bolted on. We therefore
declare it as a separate model class instead of reusing
``vllm_ascend/models/deepseek_v4_mtp.py``.

Key design decisions (all from upstream PR #46995):

* ``embed_tokens`` and ``lm_head`` are aliased from the target model at load
  time (see ``vllm_ascend/spec_decode/dspark/load.py``).
* The draft body is N (default 3) ``DeepseekV2DecoderLayer`` blocks reusing
  the existing DSv4 decoder layer implementation. The final stage carries the
  Markov + Confidence heads.
* ``mtp.{i}.*`` checkpoint keys are split: head-stack params (norm / hc_head /
  markov / confidence / main_proj / main_norm) land at ``model.*``; everything
  else routes to ``model.layers.{i}.*``.
* Non-causal sliding-window attention is achieved at the speculator side via
  a non-causal sparse attention extension (or dense fallback). This model
  class itself only owns the forward; the attention backend is selected by
  the speculator when it builds attn metadata.
"""

from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.models.deepseek_v4 import DeepseekV2DecoderLayer, DeepseekV2MixtureOfExperts, DeepseekV4MoE

# Default DSpark hyperparameters (override via inference/config.json).
# Match deepseek-ai/DeepSeek-V4-Flash-DSpark inference/config.json.
DSPARK_DEFAULT_N_MTP_LAYERS = 3
DSPARK_DEFAULT_BLOCK_SIZE = 5
DSPARK_DEFAULT_MARKOV_RANK = 256

# MoE expert scale suffix differs by expert dtype:
#   fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DSparkMarkovHead(nn.Module):
    """Low-rank Markov head (paper §3.1, equation 5).

    Logit bias for position k given the previously sampled token:
        B(x_{k-1}, ·) = W₁[x_{k-1}] · W₂
    where W₁ ∈ R^{V×r} (embedding) and W₂ ∈ R^{r×V} (head). r = 256 default.

    The bias is returned separately from the embedding so the speculator can
    feed the embedding into the confidence head once the block is complete.
    """

    def __init__(
        self,
        vocab_size: int,
        markov_rank: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(
            vocab_size,
            markov_rank,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "markov_w2"),
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """One Markov W₁ lookup; returns the low-rank embedding."""
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor, logits_processor: LogitsProcessor) -> torch.Tensor:
        """Project ``markov_embed`` through W₂ to produce the per-vocab bias."""
        return logits_processor(self.markov_w2, markov_embed)


class DSparkConfidenceHead(nn.Module):
    """Confidence head (paper §3.2.1, equation 7).

    c_k = σ(W_proj · [h_k ; W₁[x_{k-1}]])

    Kept in fp32 because the checkpoint stores ``proj`` in bf16 and the
    downstream prefix scheduler (M2-follow-up DSD) wants fp32 cumulative
    products for survival probability arithmetic.
    """

    def __init__(self, input_dim: int, prefix: str = "") -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed], dim=-1).float()
        return self.proj(x).squeeze(-1)


class DSparkDeepseekV4Model(nn.Module):
    """Draft body: N (=n_mtp_layers, default 3) decoder layers + heads.

    Layer indices are offset by ``num_hidden_layers`` so the layer prefixes are
    ``model.layers.{num_hidden_layers}..``, ``{+1}``, ``{+2}`` — the same naming
    convention upstream uses (vllm PR #46995 dspark.py L94).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config: PretrainedConfig = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(getattr(config, "dspark_target_layer_ids", ()))
        assert self.target_layer_ids, (
            "DSpark config missing `dspark_target_layer_ids` (the target layers whose "
            "hidden states get fused into the draft context)."
        )
        self.num_dspark_layers = int(getattr(config, "n_mtp_layers", None) or DSPARK_DEFAULT_N_MTP_LAYERS)
        self.dspark_block_size = int(getattr(config, "dspark_block_size", DSPARK_DEFAULT_BLOCK_SIZE))
        self.dspark_markov_rank = int(getattr(config, "dspark_markov_rank", DSPARK_DEFAULT_MARKOV_RANK))

        # Shared with the target (will be aliased by load_dspark_model).
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Main projection of target aux hidden states ([T, H * len(target_layer_ids)] -> [T, H]).
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    config=config,
                    is_draft_layer=True,
                )
                for i in range(self.num_dspark_layers)
            ]
        )

        # Head stack — only on the LAST DSpark layer in the reference impl,
        # but in the model class we keep them at model-level (matches the
        # checkpoint key layout where ``mtp.<last>.markov_head.*`` is stored
        # but functionally these are shared by the model output).
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32), requires_grad=False)
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False)
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            self.dspark_markov_rank,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + self.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "confidence_head"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states))."""
        return self.main_norm(self.main_proj(aux_hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # The DFlash base proposer builds model_kwargs generically for non-dflash
        # methods (llm_base_proposer.py L985): it always passes input_ids /
        # positions / inputs_embeds and, when pass_hidden_states_to_model is set
        # (True for the DFlash family we inherit), also `hidden_states` — the
        # target aux hidden states. We accept **kwargs so any additional
        # DFlash/eagle plumbing keys don't raise TypeError.
        #
        # TODO(M-follow-up): wire the actual draft-block forward. For now we
        # fold the target aux hidden (via combine_hidden_states) into the input
        # embedding and run the draft layers — a placeholder that exercises the
        # pipeline end-to-end and returns pre-norm head hidden for compute_logits.
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        if hidden_states is not None:
            aux = hidden_states
            # If the base passed the raw per-layer aux stack ([T, H*len(ids)]),
            # combine it down to [T, H]; if it is already [T, H], use as-is.
            expected_aux = self.hidden_size * len(self.target_layer_ids)
            if aux.dim() == 2 and aux.shape[-1] == expected_aux:
                aux = self.combine_hidden_states(aux)
            if aux.shape == inputs_embeds.shape:
                inputs_embeds = inputs_embeds + aux
        block = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = None
        for layer in self.layers:
            block, residual = layer(positions=positions, hidden_states=block, residual=residual)
        # NPU-side mhc_post + hc_head_fused will be wired in the next sprint.
        # For now, naively reduce the hc copies by summing — placeholder so the
        # rest of the pipeline can be exercised end-to-end.
        return block.sum(dim=-2) if block.dim() == 3 else block


class DSparkDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):
    """Top-level DSpark causal LM. Mirrors PR #46995 dspark.py L268+."""

    # Tells load_dspark_model whether to alias target embed_tokens + lm_head.
    dspark_shares_target_embeddings = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config: PretrainedConfig = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DSparkDeepseekV4Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

        # MoE hook-up (fork's DeepseekV2MixtureOfExperts mixin needs the layer list).
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        if example_moe is not None:
            self.extract_moe_parameters(example_moe)

    # --- Hooks for AscendDsparkSpeculator (mirror PR #46995 L288+) -------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # The base proposer calls self.model(**model_kwargs); forward the DFlash
        # plumbing (hidden_states + any extra keys) down to the body.
        return self.model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.confidence_head(head_hidden, markov_embed)

    # --- Weight loading (port of PR #46995 dspark.py L337-484) -----------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``mtp.{0..n_mtp_layers-1}.*`` draft weights from the target ckpt.

        Non-mtp weights belong to the target model and are skipped.
        ``embed_tokens`` / ``lm_head`` are aliased post-load by
        ``vllm_ascend.spec_decode.dspark.load.load_dspark_model``.
        """
        expert_mapping = FusedMoE.make_expert_params_mapping(
            model=self.model,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=getattr(self, "num_redundant_experts", 0),
        )
        expert_scale_suffix = (
            ".weight_scale" if getattr(self.config, "expert_dtype", "fp4") == "fp4" else ".weight_scale_inv"
        )
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped

            if name.endswith(".scale"):
                suffix = expert_scale_suffix if _EXPERT_SCALE_RE.search(name) else ".weight_scale_inv"
                name = name.removesuffix(".scale") + suffix

            # MoE expert weights — go through the expert mapping.
            if ".experts." in name:
                if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
                    loaded_weight = loaded_weight.view(torch.uint8)
                for param_name, weight_name, expert_id, shard_id in expert_mapping:
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded.add(name_mapped)
                        break
                continue

            # Stacked-shard rule applies only to decoder-layer weights — the
            # head stack (markov_w1/w2, hc_head_*) must skip otherwise
            # ``markov_w1`` would collide with the ``w1`` shard id.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded.add(name)
                break
            else:
                # attention sinks (sharded by head).
                if "attn_sink" in name and name in params_dict:
                    narrow = loaded_weight[head_start:head_end]
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded.add(name)
                    continue
                if ".shared_experts.w2" in name:
                    name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        logger.info_once("DSpark draft model loaded: %d params", len(loaded))
        return loaded

    @staticmethod
    def _remap_dspark_name(name: str) -> str | None:
        """``mtp.<i>.<rest>`` → this model's parameter path.

        Returns None for non-mtp weights (owned by the target model). Head
        stack params live at ``model.<rest>``; everything else is per-stage
        at ``model.layers.<i>.<rest>``.
        """
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        head_prefixes = (
            "norm.",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "markov_head.",
            "confidence_head.",
        )
        if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(head_prefixes):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"
