#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/v1/spec_decode/eagle.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
from typing import Optional, List

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.eagle import EagleProposer

from omni.adaptors.vllm.forward_context import set_forward_context
from omni.models.common.layers.attention.backend.attention import AscendAttentionState

logger = init_logger(__name__)

class PostDrafter(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)
        self.drafter_list = []
        self.method = self.vllm_config.speculative_config.method
        self.mark_static = False
        self.rejection_sampler = runner.rejection_sampler

        # eagle proposer set dtype as int32, while we need int64
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=device)
        self.positions = None
        self.hidden_states = None
        self.arange = None
        if self.method not in ('deepseek_mtp', 'pangu_ultra_moe_mtp'):
            raise ValueError(f"Speculative method should be one of ('deepseek_mtp','pangu_ultra_moe_mtp'), while get {self.method}.")
    
    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        self.model = get_model(vllm_config=self.vllm_config, model_config=draft_model_config)
        self.model.set_share_weight(target_model)

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)

        self.attn_layer_names = list(draft_attn_layer_names)

    def verify_and_prepare_inputs(self,
                                  input_ids,
                                  logits,
                                  logits_indices,
                                  sampling_metadata,
                                  num_decodes,
                                  num_prefills,
                                  chunk_next_tokens: Optional[torch.Tensor] = None,
                                  chunk_next_indices: Optional[torch.Tensor] = None,
                                  ):
        sampler_output, forward_tokens, last_accepted_index, accepted_num = self.rejection_sampler(
            input_ids=input_ids,
            logits=logits,
            logits_indices=logits_indices,
            sampling_metadata=sampling_metadata,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
        )
        self.input_ids[:input_ids.numel() - 1] = input_ids[1:]
        if num_decodes > 0:
            self.input_ids[last_accepted_index] = forward_tokens.view(-1)[last_accepted_index]
        elif num_prefills> 0:
            self.input_ids[logits_indices] = forward_tokens.view(-1)[last_accepted_index]
            if chunk_next_indices is not None:
                self.input_ids[chunk_next_indices] = chunk_next_tokens
        
        return sampler_output, last_accepted_index, accepted_num

    def prepare_dummy_input(self, input_ids):
        self.input_ids[:input_ids.numel() - 1] = input_ids[1:]

    @torch.inference_mode()
    def propose(self,
                num_tokens,
                positions,
                kv_caches,
                attn_metadata,
                previous_hidden_states,
                last_accepted_index,
                sample_indices,
                **kwargs,
    ):
        input_ids = self.input_ids[:num_tokens]
        if kv_caches is None:
            with set_forward_context(None, self.vllm_config):
                for layer_idx in range(self.speculative_config.num_speculative_tokens):
                    self.model(
                        input_ids=input_ids,
                        positions=positions,
                        kv_caches=None,
                        attn_metadata=None,
                        previous_hidden_states=previous_hidden_states,
                        mtp_layer_idx=layer_idx,
                    )
                return None
        else:
            first_attn_metadate = attn_metadata
            if isinstance(attn_metadata, dict):
                 first_attn_metadate = attn_metadata[self.attn_layer_names[0]]
            attn_state = first_attn_metadate.attn_state
            draft_forward_tokens_list = []
            
            if self.runner.enable_torchair_graph_mode and attn_state == AscendAttentionState.DecodeOnly \
                and (not self.mark_static):
                    torch._dynamo.mark_static(input_ids)
                    torch._dynamo.mark_static(previous_hidden_states)
                    self.mark_static = True
            with set_forward_context(attn_metadata, self.vllm_config):
                is_dummy = (last_accepted_index is None) or (sample_indices is None)
                for layer_idx in range(self.speculative_config.num_speculative_tokens):
                    drafter_logits, previous_hidden_states = self.model(
                        input_ids=input_ids,
                        positions=positions,
                        kv_caches=kv_caches,
                        attn_metadata=attn_metadata,
                        previous_hidden_states=previous_hidden_states,
                        selected_indices=None if attn_state == AscendAttentionState.DecodeOnly else sample_indices,
                        mtp_layer_idx=layer_idx,
                    )
                    if not is_dummy:
                        draft_forward_tokens = drafter_logits[last_accepted_index].argmax(dim=-1)
                        draft_forward_tokens_list.append(draft_forward_tokens)
                    if layer_idx == self.speculative_config.num_speculative_tokens - 1:
                        break
                    if not is_dummy:
                        input_ids = torch.roll(input_ids, -1, -1)
                        if attn_state == AscendAttentionState.DecodeOnly:
                            input_ids[last_accepted_index] = draft_forward_tokens
                        else: # prefill
                            input_ids[sample_indices] = draft_forward_tokens
            if last_accepted_index is None:
                return None
            else:
                return torch.stack(draft_forward_tokens_list, dim=1)
