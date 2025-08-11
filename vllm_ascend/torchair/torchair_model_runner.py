#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

from typing import Optional

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context

from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ,
                               maybe_converting_weight_acl_format)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class NPUTorchairModelRunner(NPUModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

    def _get_forward_metadata_across_dp_and_pad(
            self, num_tokens: int, with_prefill: bool, enable_dbo: bool
    ) -> tuple[int, Optional[torch.Tensor], bool, bool]:
        if self.dp_size == 1:
            if not with_prefill:
                maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                    num_tokens)
                return maybe_padded_num_tokens, None, with_prefill, enable_dbo
            return num_tokens, None, with_prefill, enable_dbo

        num_tokens_across_dp, with_prefill, enable_dbo = self._get_forward_metadata_across_dp(
            num_tokens, with_prefill, enable_dbo)

        if not with_prefill:
            max_num_token = num_tokens_across_dp.max().item()
            maybe_padded_num_tokens = self.select_torchair_padded_batch_size(
                max_num_token)
            num_tokens_across_dp = torch.full((self.dp_size, ),
                                              maybe_padded_num_tokens,
                                              dtype=torch.int32,
                                              device="cpu")
        else:
            maybe_padded_num_tokens = num_tokens

        return maybe_padded_num_tokens, num_tokens_across_dp, with_prefill, enable_dbo

    def _build_attention_metadata(self, with_prefill, num_reqs, skip_attn):
        # NOTE: If torchair graph mode and not with_prefill,
        # we can't skip_attn, it will cause graph recompile.
        if not with_prefill:
            attn_metadata = self.attn_metadata_builder.build_torchair_graph_dummy(
                num_reqs=num_reqs, num_actual_tokens=1)
        else:
            attn_metadata = super()._build_attention_metadata(
                with_prefill, num_reqs, skip_attn)
        return attn_metadata

    def _generate_dummy_run_hidden_states(self, with_prefill,
                                          is_torchair_compile, input_ids,
                                          positions, attn_metadata, num_tokens,
                                          intermediate_tensors, inputs_embeds):

        if not with_prefill:
            # Only mark static while compiling
            if is_torchair_compile:
                torch._dynamo.mark_static(input_ids)
                torch._dynamo.mark_static(positions)
                torch._dynamo.mark_static(attn_metadata.decode.block_table)
                torch._dynamo.mark_static(attn_metadata.decode.input_positions)
                torch._dynamo.mark_static(get_forward_context().mc2_mask)
                if hasattr(attn_metadata.decode, "sin"):
                    torch._dynamo.mark_static(attn_metadata.decode.sin)
                    torch._dynamo.mark_static(attn_metadata.decode.cos)
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
                if self.speculative_config:
                    torch._dynamo.mark_static(attn_metadata.decode.attn_mask)
                for kv in self.kv_caches:
                    assert isinstance(kv, tuple), "kv_cache must be a tuple"
                    torch._dynamo.mark_static(kv[0])
                    torch._dynamo.mark_static(kv[1])

            maybe_converting_weight_acl_format(self.model,
                                               ACL_FORMAT_FRACTAL_NZ)

            compiled_model = self._get_torchair_lazy_compiled_model(num_tokens)
            model_kwargs = {}
            model_kwargs["kv_caches"] = self.kv_caches
            model_kwargs["attn_metadata"] = attn_metadata
            hidden_states = compiled_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
                **model_kwargs,
            )
        else:
            hidden_states = super()._generate_dummy_run_hidden_states(
                with_prefill, is_torchair_compile, input_ids, positions,
                attn_metadata, num_tokens, intermediate_tensors, inputs_embeds)
        return hidden_states
