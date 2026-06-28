# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import enable_dsa_cp

from .deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV2MixtureOfExperts,
    DeepseekV4MoE,
    get_spec_layer_idx_from_weight_name,
)

# DSpark (paper arxiv:2606.19348). Defaults match deepseek-ai/DeepSeek-V4-Flash-DSpark
# `inference/config.json`. Only used when `envs.VLLM_ASCEND_ENABLE_DSPARK` is on.
DSPARK_DEFAULT_BLOCK_SIZE = 5
DSPARK_DEFAULT_MARKOV_RANK = 256
DSPARK_DEFAULT_N_MTP_LAYERS = 3


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class DSparkMarkovHead(nn.Module):
    """DSpark Markov head (paper arxiv:2606.19348, §3.1).

    Low-rank token-dependent bias: ``markov_w2(markov_w1(prev_token))`` injected
    into the parallel backbone logits to add intra-block dependency without
    paying the latency of a full autoregressive head.

    Weight names in the open-source ``deepseek-ai/DeepSeek-V4-Flash-DSpark``
    checkpoint: ``markov_head.markov_w1.weight`` (``[vocab, rank]``) and
    ``markov_head.markov_w2.weight`` (``[vocab, rank]``).
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

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one Markov step.

        Returns ``(logits_bias, markov_embed)``:
          * ``logits_bias`` (``[*, vocab]``): the per-token bias to be added onto
            the parallel backbone logits at the SAME draft position.
          * ``markov_embed`` (``[*, markov_rank]``): the unprojected low-rank
            embedding consumed by ``DSparkConfidenceHead`` after the full draft
            block is sampled.

        Per ``inference/model.py``:
            embed = markov_w1(token_ids)
            logits = markov_w2(embed)         # full_logits=True
            return logits, embed
        """
        markov_embed = self.markov_w1(token_ids)
        # ``markov_w2`` is a ``ParallelLMHead``: it owns the projection weight
        # ``[vocab, markov_rank]`` but does NOT execute the matmul on its own
        # (vLLM routes the head through a ``LogitsProcessor``). Use the
        # underlying ``weight`` directly so this returns a plain ``[*, vocab]``
        # bias tensor that the caller can ``add_()`` onto sampler logits.
        logits_bias = torch.nn.functional.linear(markov_embed, self.markov_w2.weight)
        return logits_bias, markov_embed


class DSparkConfidenceHead(nn.Module):
    """DSpark confidence head (paper arxiv:2606.19348, §3.2).

    Predicts per-position acceptance probability from ``concat(hidden, markov_embed)``.
    Kept in fp32 because the checkpoint stores ``proj`` in fp32 (per
    ``inference/model.py`` comment) and the confidence score needs full precision
    for the downstream prefix scheduler in M2.

    Weight name in checkpoint: ``confidence_head.proj.weight`` (``[1, hidden+rank]``).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(x.float()).squeeze(-1)


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str, is_last_mtp_layer: bool = False) -> None:
        super().__init__()

        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.e_proj = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=False, quant_config=quant_config, return_bias=False
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size, config.hidden_size, bias=False, quant_config=quant_config, return_bias=False
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.device = current_platform.device_type

        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        self.shared_head = SharedHead(config=config, prefix=prefix, quant_config=quant_config)
        self.mtp_block = DeepseekV2DecoderLayer(
            vllm_config,
            prefix,
            config=self.config,
            topk_indices_buffer=topk_indices_buffer,
            is_draft_layer=True,
        )
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.norm_eps = config.rms_norm_eps

        # DSpark (paper arxiv:2606.19348): attach Markov head + Confidence head
        # only to the LAST MTP stage. They are silent placeholders unless the
        # DSpark proposer runs the markov sample loop in forward (M1.5).
        # The checkpoint stores these weights at `mtp.<last_stage>.markov_head.*`
        # and `mtp.<last_stage>.confidence_head.proj.weight`.
        self.is_dspark_last_layer = bool(envs.VLLM_ASCEND_ENABLE_DSPARK) and is_last_mtp_layer
        if self.is_dspark_last_layer:
            self.dspark_block_size = int(getattr(config, "dspark_block_size", DSPARK_DEFAULT_BLOCK_SIZE))
            self.dspark_markov_rank = int(getattr(config, "dspark_markov_rank", DSPARK_DEFAULT_MARKOV_RANK))
            self.markov_head = DSparkMarkovHead(
                vocab_size=config.vocab_size,
                markov_rank=self.dspark_markov_rank,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "markov_head"),
            )
            self.confidence_head = DSparkConfidenceHead(
                input_dim=config.hidden_size + self.dspark_markov_rank,
            )
        else:
            self.markov_head = None
            self.confidence_head = None

    def apply_dspark_markov_bias(
        self,
        logits: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply one DSpark Markov step on top of backbone-produced logits.

        Used by ``DSparkProposer`` (M1.6) inside its serial sample loop. The
        proposer calls this once per draft position with the token sampled at
        the previous position, and the returned ``markov_embed`` is collected
        across the draft block to feed the confidence head once the block is
        complete.

        Returns ``(biased_logits, markov_embed)``. When this layer is not the
        DSpark last-stage layer (i.e. ``markov_head is None``), returns the
        ``logits`` untouched and ``markov_embed=None``.

        Note: this is a pure helper. The forward path (``self.forward``) is
        unchanged so non-DSpark workloads keep byte-for-byte parity.
        """
        if self.markov_head is None:
            return logits, None
        logits_bias, markov_embed = self.markov_head(prev_token_ids)
        return logits + logits_bias, markov_embed

    def compute_dspark_confidence(
        self,
        hidden_states: torch.Tensor,
        markov_embeds: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute per-position acceptance scores for a completed draft block.

        ``hidden_states`` is the per-position backbone hidden  ``[bsz, block_size, dim]``;
        ``markov_embeds`` is the stacked Markov embeddings  ``[bsz, block_size, rank]``
        accumulated across the serial sample loop. Returns ``None`` when this
        layer is not the DSpark last-stage layer.
        """
        if self.confidence_head is None:
            return None
        return self.confidence_head(hidden_states, markov_embeds)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        # masking inputs at position 0, as not needed by MTP
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = previous_hidden_states.view(-1, self.hc_mult, self.config.hidden_size)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.e_proj(inputs_embeds).unsqueeze(-2) + self.h_proj(previous_hidden_states)

        hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)

        # hidden_states = self.hc_head(hidden_states, self.hc_head_fn,
        #                              self.hc_head_scale, self.hc_head_base)

        return hidden_states

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)


class DeepSeekMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        # DSpark checkpoints ship with 3 MTP layers but some HF root config.json
        # files still report num_nextn_predict_layers=1 for legacy transformers
        # compatibility. When DSpark is enabled, honor `dspark_n_mtp_layers`
        # (config field or fallback to 3) so the layer ModuleDict matches the
        # checkpoint's `mtp.[0,1,2].*` weight layout.
        if envs.VLLM_ASCEND_ENABLE_DSPARK:
            dspark_n = int(getattr(config, "dspark_n_mtp_layers", DSPARK_DEFAULT_N_MTP_LAYERS))
            if dspark_n > self.num_mtp_layers:
                self.num_mtp_layers = dspark_n
        # to map the exact layer index from weights

        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.{idx}",
                    is_last_mtp_layer=(idx == self.num_mtp_layers - 1),
                )
                for idx in range(
                    0,
                    self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(current_step_idx)]
        hidden_states = hidden_states.view(-1, mtp_layer.hc_mult, mtp_layer.config.hidden_size)
        hidden_states = mtp_layer.hc_head(
            hidden_states, mtp_layer.hc_head_fn, mtp_layer.hc_head_scale, mtp_layer.hc_head_base
        )
        logits = self.logits_processor(mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states))
        return logits


@support_torch_compile
class DeepSeekV4MTP(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepSeekMultiTokenPredictor(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp"))
        # Set MoE hyperparameters
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, DeepSeekMultiTokenPredictorLayer)
            layer = layer.mtp_block
            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV4MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, hidden_states, inputs_embeds, spec_step_idx)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        rocm_aiter_moe_shared_expert_enabled = getattr(get_ascend_config(), "mix_placement", False)
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            model=self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if rocm_aiter_moe_shared_expert_enabled else 0),
            num_redundant_experts=self.num_redundant_experts,
        )

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.quant_config is not None and self.quant_config.get_name() == "fp8":
                if name == "embed.weight":
                    name = "mtp.0.emb.tok_emb.weight"

                if name == "head.weight":
                    name = "mtp.0.head.weight"

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue

            # Multi-MTP-stage checkpoints (DSpark: mtp.0/mtp.1/mtp.2) route to
            # the corresponding `model.layers.<spec_layer>.*` slot. The shared
            # embedding lives only at mtp.0 in legacy ckpts; DSpark keeps the
            # same convention.
            mtp_prefix = f"mtp.{spec_layer}."
            assert mtp_prefix in name, f"unexpected MTP weight name without {mtp_prefix} prefix: {name}"
            target_layer_prefix = f"model.layers.{spec_layer}."
            if ".emb.tok_emb." in name:
                name = name.replace(mtp_prefix, "model.")
            elif ".markov_head." in name or ".confidence_head." in name:
                # DSpark heads live on the last MTP stage; map to the layer-level
                # module (no `.mtp_block.` indirection).
                name = name.replace(mtp_prefix, target_layer_prefix)
            elif self.no_mtp_block_in_name(name):
                name = name.replace(mtp_prefix, target_layer_prefix)
            else:
                name = name.replace(mtp_prefix, f"{target_layer_prefix}mtp_block.")

            if ".w1." in name:
                name = name.replace(".w1.", ".gate_proj.")
            if ".w2." in name:
                name = name.replace(".w2.", ".down_proj.")
            if ".w3." in name:
                name = name.replace(".w3.", ".up_proj.")

            if name.endswith(".scale"):
                name = name.replace(".scale", ".weight_scale")

            if ".head." in name:
                name = name.replace(".head.", ".shared_head.head.")

            if ".norm." in name:
                name = name.replace(".norm.", ".shared_head.norm.")

            if ".emb.tok_emb." in name:
                name = name.replace(".emb.tok_emb.", ".embed_tokens.")

            if "attn" in name and "self_attn" not in name:
                name = name.replace(".attn.", ".self_attn.")
            if ".ffn." in name:
                name = name.replace(".ffn.", ".mlp.")
            if ".ffn_norm." in name:
                name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
            if ".attn_norm." in name:
                name = name.replace(".attn_norm.", ".input_layernorm.")

            if ".gate.bias" in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            if "sink" in name:
                param = params_dict[name]
                if enable_dsa_cp():
                    param.data.copy_(loaded_weight)
                else:
                    # Handle attention sinks (distributed across ranks)
                    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                    param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            is_fusion_moe_shared_experts_layer = rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        if split_dim == 0:
                            weight_to_load = loaded_weight[j * chunk_size : (j + 1) * chunk_size, :]
                        else:
                            weight_to_load = loaded_weight[:, j * chunk_size : (j + 1) * chunk_size]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    is_expert_weight = False
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        # # According to DeepSeek-V3 Technical Report, MTP modules
                        # # shares embedding layer. We only load the first weights.
                        # if (
                        #     spec_layer != self.model.mtp_start_layer_idx
                        #     and ".layers" not in name
                        # ):
                        #     continue

                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)
        return loaded_params

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        """
        Rewrite the weight name to match the format of the original model.
        Add .mtp_block for modules in transformer layer block for spec layer
        and rename shared layer weights to be top level.
        """
        spec_layer_weight_names = [
            "embed_tokens",
            "enorm",
            "hnorm",
            "eh_proj",
            "shared_head",
        ]
        shared_weight_names = ["embed_tokens"]
        spec_layer_weight = False
        shared_weight = False
        for weight_name in spec_layer_weight_names:
            if weight_name in name:
                spec_layer_weight = True
                if weight_name in shared_weight_names:
                    shared_weight = True
                break
        if not spec_layer_weight:
            # treat rest weights as weights for transformer layer block
            name = name.replace(f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block.")
        elif shared_weight:
            # treat shared weights as top level weights
            name = name.replace(f"model.layers.{spec_layer}.", "model.")
        return name

    def no_mtp_block_in_name(self, layer_name: str) -> bool:
        names = [
            ".hc_head_fn",
            ".hc_head_base",
            ".hc_head_scale",
            ".e_proj.",
            ".h_proj.",
            ".enorm.",
            ".hnorm.",
            ".norm.",
            ".head.",
            ".emb.tok_emb.",
        ]
        return any(name in layer_name for name in names)
