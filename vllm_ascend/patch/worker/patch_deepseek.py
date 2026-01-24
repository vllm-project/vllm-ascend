from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.model_executor.models.deepseek_v2 import (DeepseekV2Model,
                                                    _get_llama_4_scaling)
from vllm.sequence import IntermediateTensors


def forward(
    self,
    input_ids,
    positions,
    intermediate_tensors,
    inputs_embeds,
):
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    # Compute llama 4 scaling once per forward pass if enabled
    # Note(wxy): This is a hack fix to avoid graph mode error for torch 2.8
    # We'll find a better way to remove this patch.
    try:
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling")
    except AttributeError:
        llama_4_scaling_config = None
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    for layer in islice(self.layers, self.start_layer, self.end_layer):
        hidden_states, residual = layer(positions, hidden_states, residual,
                                        llama_4_scaling)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual
        })

    hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states


DeepseekV2Model.forward = forward
