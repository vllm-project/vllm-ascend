# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 DSpark draft model for Ascend.

DSpark weights are stored under the target checkpoint's ``mtp.*`` namespace,
but the draft path is a block drafter rather than the ordinary serial MTP
module. The target model provides selected layer hidden states; this model
projects them into the draft attention context and emits a full draft block.
"""

import torch
import torch.nn as nn
import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead, DSparkMarkovHead
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.models.common.deepseek_dspark import (
    DeepseekDSparkForCausalLMMixin,
    DeepseekDSparkModelMixin,
    _apply_dsv4_rope,
    _get_dspark_num_mtp_layers,
)
from vllm_ascend.models.common.ops.sequence_parallel import sp_padding_mask, sp_shard
from vllm_ascend.models.deepseek_v4.model import (
    DeepseekV2DecoderLayer as DeepseekV4DecoderLayer,
)


class DeepseekV4DSparkModel(nn.Module, DeepseekDSparkModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.vllm_config = vllm_config
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.num_dspark_layers = _get_dspark_num_mtp_layers(config)
        self.mtp_start_layer_idx = config.num_hidden_layers

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + idx): DeepseekV4DecoderLayer(
                    vllm_config,
                    prefix=f"mtp.{idx}",
                    is_draft_layer=True,
                )
                for idx in range(self.num_dspark_layers)
            }
        )

        self.needs_moe_input_ids = any(
            layer.mlp.gate.tid2eid is not None or layer.mlp.gate.bias_vl is not None for layer in self.layers.values()
        )
        first_layer = self.layers[str(self.mtp_start_layer_idx)]
        self.use_sequence_parallel_moe = first_layer.use_sequence_parallel_moe

        _model_quant_cfg = getattr(config, "quantization_config", None)
        _main_proj_qconfig = (
            vllm_config.quant_config
            if _model_quant_cfg is not None and _model_quant_cfg.get("quant_method") == "fp8"
            else None
        )
        self.main_proj = ColumnParallelLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=_main_proj_qconfig,
            prefix=maybe_prefix(prefix, f"layers.{self.mtp_start_layer_idx}.main_proj"),
            gather_output=True,
        )
        self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        first_layer.main_proj = self.main_proj
        first_layer.main_norm = self.main_norm

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        last_layer_idx = self.mtp_start_layer_idx + self.num_dspark_layers - 1
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(
                prefix,
                f"layers.{last_layer_idx}.markov_head",
            ),
        )

        self.confidence_head = DSparkConfidenceHead(
            input_dim=config.hidden_size + config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "confidence_head"),
            bias=False,
            with_markov=True,
        )
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        last_layer = self.layers[str(last_layer_idx)]
        last_layer.norm = self.norm
        last_layer.markov_head = self.markov_head
        last_layer.hc_head_fn = self.hc_head_fn
        last_layer.hc_head_base = self.hc_head_base
        last_layer.hc_head_scale = self.hc_head_scale

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids).unsqueeze(-2).repeat(1, self.hc_mult, 1)
        full_num_tokens = positions.shape[0]
        use_sp = self.use_sequence_parallel_moe
        orig_is_padding = None
        forward_context = None
        if use_sp:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                orig_is_padding = forward_context.is_padding
                forward_context.is_padding = sp_padding_mask(orig_is_padding, hidden_states)
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        residual = None
        moe_input_ids = input_ids
        if self.needs_moe_input_ids:
            moe_input_ids = torch.where(input_ids == -1, 0, input_ids)
        for layer in self.layers.values():
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling=None,
                input_ids=moe_input_ids,
            )
        if use_sp:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if forward_context is not None:
            forward_context.is_padding = orig_is_padding
        head_hidden = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        return head_hidden


@support_torch_compile
class DSparkDeepseekV4ForCausalLM(DeepseekDSparkForCausalLMMixin, nn.Module, SupportsEagle3):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config

        # check if quant config exist
        from vllm_ascend.utils import get_rotation_path

        self.rotation_path = get_rotation_path(vllm_config) if vllm_config.quant_config is not None else None

        self.model = DeepseekV4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.set_moe_parameters()


__all__ = ["DSparkConfidenceHead", "DSparkMarkovHead", "_get_dspark_num_mtp_layers", "_apply_dsv4_rope"]
