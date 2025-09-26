# SPDX-License-Identifier: Apache-2.0
from typing import Optional, List
import torch

from vllm.attention import  AttentionMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.gemma3_mm import Gemma3ForConditionalGeneration

class Gemma3ForConditionalGeneration(Gemma3ForConditionalGeneration):


    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor] = None,
                attn_metadata: AttentionMetadata = None,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs: object) -> IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches=kv_caches,
                                                  attn_metadata=attn_metadata,
                                                  intermediate_tensors=intermediate_tensors,
                                                  inputs_embeds=inputs_embeds,
                                                  **kwargs)

        return hidden_states
