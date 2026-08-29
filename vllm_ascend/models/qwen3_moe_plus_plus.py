# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Inference-only Qwen3 MoE++ (ZEDA Dynamic MoE) model for vLLM-Ascend.

The ZEDA model extends Qwen3-30B-A3B with additional "zero experts" (ZCE)
that participate in top-k routing but produce zero output, creating a
"dynamic MoE activation" effect by diluting real expert weights.

Architecture differences from stock Qwen3MoE:
- Gate outputs logits for total_num_experts (128 real + 64 zero = 192)
- Zero experts are handled by AscendRoutedExperts.forward_impl()
  via zero_expert_num / zero_expert_type attributes set on the
  AscendRoutedExperts instance
- No ZeroExpertRouter is created (avoids e_score_correction_bias assertion)
"""

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    maybe_prefix,
)


class Qwen3MoePlusPlusDecoderLayer(Qwen3MoeDecoderLayer):
    """Decoder layer that upgrades the MoE block to support zero experts."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        if isinstance(self.mlp, Qwen3MoeSparseMoeBlock):
            config = vllm_config.model_config.hf_text_config
            zce_nums = getattr(config, "zce_nums", []) or []
            zce_types = getattr(config, "zce_types", []) or []
            total_num_experts = config.num_experts + sum(zce_nums)

            # Recreate gate to output logits for all experts (real + zero).
            # The old 128-dim gate is discarded; the AscendMoERunner's
            # internal _gate reference becomes stale but is harmless because
            # is_internal_router returns False (no weight_fp32 on
            # ReplicatedLinear) and there are no shared experts.
            self.mlp.gate = ReplicatedLinear(
                config.hidden_size,
                total_num_experts,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp.gate",
            )

            # Set zero expert attributes on AscendRoutedExperts so that
            # AscendRoutedExperts.forward_impl() reads them via getattr()
            # and calls zero_experts_compute() after select_experts().
            routed_experts = self.mlp.experts.routed_experts
            routed_experts.zero_expert_num = sum(zce_nums)
            routed_experts.zero_expert_type = zce_types[0] if zce_types else "zero"


class Qwen3MoePlusPlusModel(Qwen3MoeModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=Qwen3MoePlusPlusDecoderLayer,
        )


class Qwen3MoePlusPlusForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Skip Qwen3MoeForCausalLM.__init__ to use our PlusPlus model class.
        # nn.Module.__init__ is sufficient; the mixin classes have no __init__.
        super(Qwen3MoeForCausalLM, self).__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping: dict[str, list[str]] = dict(self.packed_modules_mapping)
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = Qwen3MoePlusPlusModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self.expert_weights: list = []
        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts
