# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mistral Large 3 text-model adapter for Ascend.

Mistral Large 3 uses the DeepSeek-V3-style MLA and granular MoE blocks. The
upstream implementation provides the model topology and forward path, while
this adapter translates Mistral's native checkpoint names to vLLM names.
Ascend's FusedMoE runner is selected before this module is imported.
"""

from collections.abc import Iterable

import regex as re
import torch
from vllm.model_executor.models.deepseek_v2 import DeepseekV3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class MistralLarge3ForCausalLM(DeepseekV3ForCausalLM):
    """Ascend entry point for the Mistral Large 3 MoE architecture.

    The inherited forward path contains MLA attention, dense warm-up layers,
    routed experts, and the shared expert. Keeping that path intact preserves
    expert parallelism and the Ascend FusedMoE runner integration.
    """

    # All regular expressions are anchored so translated keys cannot match a
    # later rule when WeightsMapper applies the rules sequentially.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={
            re.compile(r"\Alayers\.(\d+)\.attention_norm\.weight\Z"): (r"model.layers.\1.input_layernorm.weight"),
            re.compile(r"\Alayers\.(\d+)\.attention\.wq_a\.(\w+)\Z"): (r"model.layers.\1.self_attn.q_a_proj.\2"),
            re.compile(r"\Alayers\.(\d+)\.attention\.q_a_norm\.weight\Z"): (
                r"model.layers.\1.self_attn.q_a_layernorm.weight"
            ),
            re.compile(r"\Alayers\.(\d+)\.attention\.wq_b\.(\w+)\Z"): (r"model.layers.\1.self_attn.q_b_proj.\2"),
            re.compile(r"\Alayers\.(\d+)\.attention\.wkv_a_with_mqa\.(\w+)\Z"): (
                r"model.layers.\1.self_attn.kv_a_proj_with_mqa.\2"
            ),
            re.compile(r"\Alayers\.(\d+)\.attention\.kv_a_norm\.weight\Z"): (
                r"model.layers.\1.self_attn.kv_a_layernorm.weight"
            ),
            re.compile(r"\Alayers\.(\d+)\.attention\.wkv_b\.(\w+)\Z"): (r"model.layers.\1.self_attn.kv_b_proj.\2"),
            re.compile(r"\Alayers\.(\d+)\.attention\.wo\.(\w+)\Z"): (r"model.layers.\1.self_attn.o_proj.\2"),
            re.compile(r"\Alayers\.(\d+)\.ffn_norm\.weight\Z"): (r"model.layers.\1.post_attention_layernorm.weight"),
            re.compile(r"\Alayers\.(\d+)\.feed_forward\.w1\.(\w+)\Z"): (r"model.layers.\1.mlp.gate_proj.\2"),
            re.compile(r"\Alayers\.(\d+)\.feed_forward\.w2\.(\w+)\Z"): (r"model.layers.\1.mlp.down_proj.\2"),
            re.compile(r"\Alayers\.(\d+)\.feed_forward\.w3\.(\w+)\Z"): (r"model.layers.\1.mlp.up_proj.\2"),
            re.compile(r"\Alayers\.(\d+)\.gate\.weight\Z"): r"model.layers.\1.mlp.gate.weight",
            re.compile(r"\Alayers\.(\d+)\.shared_experts\.w1\.(\w+)\Z"): (
                r"model.layers.\1.mlp.shared_experts.gate_proj.\2"
            ),
            re.compile(r"\Alayers\.(\d+)\.shared_experts\.w2\.(\w+)\Z"): (
                r"model.layers.\1.mlp.shared_experts.down_proj.\2"
            ),
            re.compile(r"\Alayers\.(\d+)\.shared_experts\.w3\.(\w+)\Z"): (
                r"model.layers.\1.mlp.shared_experts.up_proj.\2"
            ),
            re.compile(r"\Alayers\.(\d+)\.experts\.(\d+)\.w1\.(\w+)\Z"): (
                r"model.layers.\1.mlp.experts.\2.gate_proj.\3"
            ),
            re.compile(r"\Alayers\.(\d+)\.experts\.(\d+)\.w2\.(\w+)\Z"): (
                r"model.layers.\1.mlp.experts.\2.down_proj.\3"
            ),
            re.compile(r"\Alayers\.(\d+)\.experts\.(\d+)\.w3\.(\w+)\Z"): (r"model.layers.\1.mlp.experts.\2.up_proj.\3"),
            re.compile(r"\Anorm\.weight\Z"): "model.norm.weight",
            re.compile(r"\Atok_embeddings\.weight\Z"): "model.embed_tokens.weight",
            re.compile(r"\Aoutput\.weight\Z"): "lm_head.weight",
        },
        # qscale_weight_2 is the reserved NVFP4 secondary-scale interface.
        # The quantization backend remains responsible for allocating it.
        orig_to_new_suffix={
            ".qscale_act": ".input_scale",
            ".qscale_weight": ".weight_scale",
            ".qscale_weight_2": ".weight_scale_2",
        },
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load native Mistral, FP8, or NVFP4-named tensors."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
