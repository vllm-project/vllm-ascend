#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch_npu
import vllm.envs as vllm_envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

from vllm_ascend.distributed.kv_transfer.simple_buffer import SimpleBuffer
from vllm_ascend.distributed.kv_transfer.simple_pipe import SimplePipe

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


class SimpleConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.config = config
        self.model_config = config.model_config.hf_config
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.rank = rank
        self.local_rank = local_rank
        self.is_deepseek_mla = config.model_config.is_deepseek_mla
        self.use_mla_opt = not vllm_envs.VLLM_MLA_DISABLE
        self.n_layer = self.config.model_config.get_num_layers(
            self.config.parallel_config)

        self.producer_data_pipe: Optional[SimplePipe]
        self.consumer_data_pipe: Optional[SimplePipe]

        self.producer_buffer: Optional[SimpleBuffer]
        self.consumer_buffer: Optional[SimpleBuffer]

        if self.config.kv_transfer_config.is_kv_producer:
            self.producer_data_pipe = SimplePipe(
                rank=rank,
                local_rank=local_rank,
                kv_transfer_config=config.kv_transfer_config,
                hostname="",
                port_offset=rank,
            )
            self.producer_buffer = SimpleBuffer(self.producer_data_pipe)
        else:
            self.consumer_data_pipe = SimplePipe(
                rank=rank,
                local_rank=local_rank,
                kv_transfer_config=config.kv_transfer_config,
                hostname="",
                port_offset=rank,
            )
            self.consumer_buffer = SimpleBuffer(self.consumer_data_pipe)

    def select(
        self,
        input_tokens: Optional[torch.Tensor],
        roi: Optional[torch.Tensor],
        req_id: str,
    ) -> List[Optional[torch.Tensor]]:

        assert self.consumer_buffer is not None, (
            "Please initialize the "
            "consumer buffer before calling select.")
        return self.consumer_buffer.drop_select(input_tokens, roi, req_id)

    def insert(
        self,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        hidden: torch.Tensor,
        req_id: str,
    ) -> None:

        assert self.producer_buffer is not None, (
            "Please initialize the "
            "producer buffer before calling insert.")
        self.producer_buffer.insert(input_tokens, roi, keys, values, hidden,
                                    req_id)

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = self.model_config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads

        # Deepseek's MLA (Multi-head Latent Attention) uses two different
        # kv_cache shapes based on whether VLLM_MLA_DISABLE is set to 0.
        # When VLLM_MLA_DISABLE=0 (default), forward absorb is applied,
        # resulting in a kv_cache shape of [num_blks, blk_size, 1,
        # kv_lora_rank + qk_rope_head_dim].
        # When VLLM_MLA_DISABLE=1, standard FA is used instead, leading
        # to a kv_cache shape of [2, num_blks, blk_size,
        # num_key_value_heads / tp, qk_nope_head_dim + qk_rope_head_dim].
        # For more details, see vllm/attention/backends/mla/common.py.
        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = (model_config.kv_lora_rank +
                         model_config.qk_rope_head_dim)
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = (model_config.qk_nope_head_dim +
                         model_config.qk_rope_head_dim)
        else:
            head_size = getattr(
                model_config,
                "head_dim",
                int(hidden_size // num_attention_heads),
            )
        # Enumerate over all requests and insert them one by one.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You have some decode requests while using "
                               "SimpleConnector. Their KVCache won't be sent.")
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                if self.is_deepseek_mla and self.use_mla_opt:
                    key_cache = kv_cache.reshape(-1, num_heads, head_size)
                    value_cache = kv_cache.reshape(-1, num_heads, head_size)
                else:
                    key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                    value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            # shape: [num_layers, num_tokens, num_heads, head_size]
            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)
            cur_req_id = list(model_input.request_ids_to_seq_ids.keys())[idx]
            # Currently we haven't considered situation of roi, pass None here.
            self.insert(
                current_tokens,
                None,
                keys,
                values,
                hidden_or_intermediate_states[start_pos:end_pos],
                cur_req_id,
            )

        logger.info("[rank%d][P]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata", ]:
        bypass_model_exec = True

        model_config = self.model_config

        # get model config
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_heads, head_dim = kv_caches[0].shape[-2:]
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        num_layers = end_layer - start_layer
        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = (model_config.kv_lora_rank +
                         model_config.qk_rope_head_dim)
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = (model_config.qk_nope_head_dim +
                         model_config.qk_rope_head_dim)
        else:
            head_size = getattr(
                model_config,
                "head_dim",
                int(hidden_size // num_attention_heads),
            )
        self.consumer_buffer.num_heads = num_heads  # type: ignore
        self.consumer_buffer.num_layers = num_layers  # type: ignore
        self.consumer_buffer.head_size = head_size  # type: ignore
        self.consumer_buffer.dtype = kv_caches[0].dtype  # type: ignore
        self.consumer_buffer.hidden_size = hidden_size  # type: ignore

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        total_tokens = model_input.attn_metadata.num_prefill_tokens + model_input.attn_metadata.num_decode_tokens
        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        # enumerate different requests
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                logger.warning("You should set --enable_chunked_prefill=False "
                               "and --max_num_batched_tokens "
                               "should be equal to --max_seq_len_to_capture")
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            cur_req_id = list(model_input.request_ids_to_seq_ids.keys())[idx]

            ret = self.select(
                current_tokens,
                torch.ones_like(current_tokens, dtype=bool),
                cur_req_id,
            )
            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue

            keys: torch.Tensor = ret[0]
            values: torch.Tensor = ret[1]
            hidden: torch.Tensor = ret[2]

            num_computed_tokens = keys.shape[1]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory
            for i in range(
                    model_executable.model.start_layer,
                    model_executable.model.end_layer,
            ):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                if self.is_deepseek_mla and self.use_mla_opt:
                    layer.self_attn.attn = layer.self_attn.mla_attn
                    key_cache = kv_cache
                    slots = slot_mapping[start_pos:end_pos]
                    sliced_key = keys[i - model_executable.model.start_layer]
                    torch_npu._npu_reshape_and_cache_siso(key=sliced_key,
                                                          key_cache=key_cache,
                                                          slot_indices=slots)
                else:
                    key_cache, value_cache = kv_cache[0], kv_cache[1]
                    sliced_key = keys[i - model_executable.model.start_layer]
                    sliced_value = values[i -
                                          model_executable.model.start_layer]
                    torch_npu._npu_reshape_and_cache(
                        key=sliced_key,
                        value=sliced_value,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        slot_indices=slot_mapping[start_pos:end_pos],
                    )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            if get_dp_group().world_size > 1:
                bypass_model_exec = True
                hidden_or_intermediate_states = torch.empty(
                    [total_tokens, hidden_size],
                    dtype=kv_caches[0].dtype,
                    device=kv_caches[0].device)
                logger.warning(
                    "[Detect there is more one DP rank in this decode node, in this scenario, no recompute is expected when kv cache dose not received.]"
                )
            else:
                logger.warning(
                    "[rank%d]: Failed to receive all KVs and hidden "
                    "states, redo model forwarding.",
                    torch.distributed.get_rank())
                hidden_or_intermediate_states = None
        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.",
                torch.distributed.get_rank(),
            )
            # Can't directly concat here which might cause error when bs = 1.
            # hidden_or_intermediate_states = torch.empty(total_num_tokens, hidden_size, dtype=kv_caches[0].dtype, device=kv_caches[0].device)
            if len(hidden_or_intermediate_states_for_one_req) == 1:
                hidden = hidden_or_intermediate_states_for_one_req[0]
                tmp_indice = torch.tensor([0] * hidden.shape[0],
                                          dtype=torch.int64).npu()
                hidden_or_intermediate_states = torch.empty_like(hidden)
                torch_npu.scatter_update_(
                    hidden_or_intermediate_states,
                    tmp_indice,
                    hidden,
                    axis=-1,
                )
            else:
                hidden_or_intermediate_states = torch.cat(
                    hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        self.producer_data_pipe.close()  # type: ignore
        self.consumer_data_pipe.close()  # type: ignore
        self.producer_buffer.close()  # type: ignore
        self.consumer_buffer.close()  # type: ignore
