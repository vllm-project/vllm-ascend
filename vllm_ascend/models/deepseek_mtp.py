from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_ascend.utils import is_rot_weight_used, vllm_version_is


@support_torch_compile
class AscendDeepSeekMTP(DeepSeekMTP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.is_rot_weight_used = is_rot_weight_used(vllm_config)
        if self.is_rot_weight_used:
            self.rot = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if self.is_rot_weight_used:
            hidden_states = self.rot(hidden_states)
        return super().forward(input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds, spec_step_idx)

    def _maybe_set_own_lm_head(self, loaded_weights: set[str]) -> None:
        """Expose a checkpoint-provided MTP head to the runner.

        DeepSeekMTP always constructs ``shared_head``, so module existence does
        not prove head ownership. Treat it as independent only when its weight
        was loaded from the checkpoint.
        """
        mtp_layer_idx = self.model.mtp_start_layer_idx
        own_head_weight = f"model.layers.{mtp_layer_idx}.shared_head.head.weight"
        if own_head_weight not in loaded_weights:
            # GLM-5.3 and friends ship no MTP head, leaving shared_head.head at
            # its allocation-time contents. Record that so the proposer shares
            # the target head instead of inspecting those values.
            self.has_own_lm_head = False
            return

        self.has_own_lm_head = True
        mtp_layer = self.model.layers[str(mtp_layer_idx)]
        self.lm_head = mtp_layer.shared_head.head

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.quant_config is not None and (cache_scale_mapper := self.quant_config.get_cache_scale_mapper()):
            weights = cache_scale_mapper.apply(weights)

        weights_mapper = WeightsMapper(
            orig_to_new_prefix={"rot.": f"model.layers.{self.config.num_hidden_layers}.rot."},
        )
        loaded_weights = super().load_weights(weights_mapper.apply(weights))
        self._maybe_set_own_lm_head(loaded_weights)
        return loaded_weights

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if "rot" in name:
            name = name.replace(f"model.layers.{spec_layer}.rot.", "rot.")
            return name
        return super()._rewrite_spec_layer_name(spec_layer, name)


class AscendGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if vllm_version_is("0.27.1"):
            loader = AutoWeightsLoader(self, skip_prefixes=["rot."])
            return loader.load_weights(weights)
        mapper = WeightsMapper(orig_to_new_prefix={"rot.": None})
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)
