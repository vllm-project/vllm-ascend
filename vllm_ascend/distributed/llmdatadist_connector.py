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
import os
import re
import subprocess
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import torch_npu
import torchair  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

import llm_datadist  # type: ignore

TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32
}

# Get all device ips using hccn_tool
HCCN_TOOL_PATH = envs.HCCN_PATH


def get_device_ips():
    world_size = 8
    npu_info = subprocess.run(['npu-smi', 'info', '-m'],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)
    if npu_info.returncode != 0 or not os.path.exists(HCCN_TOOL_PATH):
        raise RuntimeError("No npu-smi/hccn_tool tools provided for NPU.")
    npu_start_idx = int(
        re.match(r'.*\n\t([0-9]+).*', npu_info.stdout).group(1))
    device_ip_list = []
    for ip_offset in range(world_size):
        cmd = [
            HCCN_TOOL_PATH, '-i', f'{npu_start_idx + ip_offset}', '-ip', '-g'
        ]
        device_ip_info = subprocess.run(cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
        device_ip = re.match(r'ipaddr:(.*)\n', device_ip_info.stdout).group(1)
        device_ip_list.append(device_ip)
    return device_ip_list


class KVTransferEngine:

    def __init__(self, world_size, n_layer, role, local_rank):
        self.world_size = world_size
        self.n_layer = n_layer
        self.role = role
        self.device_ip_list = get_device_ips()
        self.local_rank = local_rank
        self.cluster_id = local_rank
        self.data_dist = llm_datadist.LLMDataDist(self.role, self.cluster_id)

        prompt_device_ids = envs.PROMPT_DEVICE_ID
        decode_device_ids = envs.DECODE_DEVICE_ID
        if prompt_device_ids is None or decode_device_ids is None:
            raise ValueError(
                "Please specify env PROMPT_DEVICE_ID or DECODE_DEVICE_ID")

        prompt_ids = [
            int(x.strip()) for x in prompt_device_ids.split(",") if x.strip()
        ]
        decode_ids = [
            int(x.strip()) for x in decode_device_ids.split(",") if x.strip()
        ]

        self.prompt_ip_list = [self.device_ip_list[i] for i in prompt_ids]
        self.decode_ip_list = [self.device_ip_list[i] for i in decode_ids]

    def prepare_data_dist(self):
        options = {
            "llm.SyncKvCacheWaitTime": envs.LLMDATADIST_SYNC_CACHE_WAIT_TIME,
        }
        if self.role == llm_datadist.LLMRole.PROMPT:
            options["ge.exec.deviceId"] = str(self.local_rank)
            options[
                "llm.listenIpInfo"] = f"{self.prompt_ip_list[self.local_rank]}:{envs.LLMDATADIST_COMM_PORT}"
        else:
            options["ge.exec.deviceId"] = str(self.local_rank)
        self.data_dist.init(options)
        self.kv_transfer = self.data_dist.kv_cache_manager
        logger.info(
            f"{self.local_rank}/{self.world_size} rank data dist is ready")

    def make_cluster(self, prefill_ip, cluster_id=-1):
        cluster = llm_datadist.LLMClusterInfo()
        cluster.remote_cluster_id = cluster_id
        local_ip = self.decode_ip_list[self.local_rank]
        remote_ip = prefill_ip
        cluster.append_local_ip_info(local_ip, 0)
        cluster.append_remote_ip_info(remote_ip, 26000)
        return cluster


class LLMDataDistConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.config = config
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.rank = rank
        self.local_rank = local_rank

        if self.config.kv_transfer_config.kv_role == "kv_producer":
            self.role = llm_datadist.LLMRole.PROMPT
        elif self.config.kv_transfer_config.kv_role == "kv_consumer":
            self.role = llm_datadist.LLMRole.DECODER
        else:
            raise NotImplementedError(
                "kv_role should be inside [kv_producer, kv_consumer]")

        self.world_size = self.config.parallel_config.world_size
        self.n_layer = self.config.model_config.get_num_layers(
            self.config.parallel_config)

        self.llm_datadist_engine = KVTransferEngine(self.world_size,
                                                    self.n_layer, self.role,
                                                    self.local_rank)
        if self.role == llm_datadist.LLMRole.PROMPT:
            self.llm_datadist_engine.prepare_data_dist()
        else:
            self.llm_datadist_engine.prepare_data_dist()
            self.cluster = self.llm_datadist_engine.make_cluster(
                self.llm_datadist_engine.prompt_ip_list[self.local_rank],
                self.llm_datadist_engine.cluster_id)
            _, ret = self.llm_datadist_engine.data_dist.link_clusters(
                [self.cluster], 20000)
            logger.info(f"local_rank {self.local_rank} link, ret={ret}")

    def send_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors]
    ) -> None:
        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)

        num_layer = end_layer - start_layer

        # Get shape of input_tokens_tensor and kv_cache
        input_shape = (1, input_tokens_tensor.shape[0], 1, 1)
        hidden_shape = (1, input_tokens_tensor.shape[0], 1, hidden_size)
        kv_shape = (1, input_tokens_tensor.shape[0], num_heads, head_size)

        assert kv_caches[0].dtype == hidden_or_intermediate_states.dtype
        kv_hidden_dtype = kv_caches[0].dtype
        input_dtype = torch.int32

        # initialize LLMDatadist data structure
        key_desc = llm_datadist.CacheDesc(
            num_layer,
            kv_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[kv_hidden_dtype],
            seq_len_dim_index=1)
        value_desc = llm_datadist.CacheDesc(
            num_layer,
            kv_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[kv_hidden_dtype],
            seq_len_dim_index=1)
        input_desc = llm_datadist.CacheDesc(
            1,
            input_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[input_dtype],
            seq_len_dim_index=-1)
        hidden_desc = llm_datadist.CacheDesc(
            1,
            hidden_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[kv_hidden_dtype],
            seq_len_dim_index=-1)

        key_cache_keys = [
            llm_datadist.CacheKey(self.llm_datadist_engine.cluster_id, 0, 1)
        ]
        value_cache_keys = [
            llm_datadist.CacheKey(self.llm_datadist_engine.cluster_id, 0, 2)
        ]
        input_cache_keys = [
            llm_datadist.CacheKey(self.llm_datadist_engine.cluster_id, 0, 3)
        ]
        hidden_cache_keys = [
            llm_datadist.CacheKey(self.llm_datadist_engine.cluster_id, 0, 4)
        ]

        self.key_buffer = self.llm_datadist_engine.kv_transfer.allocate_cache(
            key_desc, key_cache_keys)
        self.value_buffer = self.llm_datadist_engine.kv_transfer.allocate_cache(
            value_desc, value_cache_keys)
        self.input_buffer = self.llm_datadist_engine.kv_transfer.allocate_cache(
            input_desc, input_cache_keys)
        self.hidden_buffer = self.llm_datadist_engine.kv_transfer.allocate_cache(
            hidden_desc, hidden_cache_keys)

        key_buffer_addr = self.key_buffer.per_device_tensor_addrs[0]
        value_buffer_addr = self.value_buffer.per_device_tensor_addrs[0]
        input_buffer_addr = self.input_buffer.per_device_tensor_addrs[0]
        hidden_buffer_addr = self.hidden_buffer.per_device_tensor_addrs[0]

        self.key_cache = torchair.llm_datadist.create_npu_tensors(
            key_desc.shape, kv_hidden_dtype, key_buffer_addr)
        self.value_cache = torchair.llm_datadist.create_npu_tensors(
            value_desc.shape, kv_hidden_dtype, value_buffer_addr)
        self.input_cache = torchair.llm_datadist.create_npu_tensors(
            input_desc.shape, input_dtype, input_buffer_addr)
        self.hidden_cache = torchair.llm_datadist.create_npu_tensors(
            hidden_desc.shape, kv_hidden_dtype, hidden_buffer_addr)

        indices = torch.tensor([0], dtype=torch.int64).npu()

        # copy cache data into llm datadist cache using scatter update
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos].to(
                torch.int32)

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                key_cache = kv_cache[0].view(-1, num_heads, head_size)
                value_cache = kv_cache[1].view(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                # copy key into datadist
                k = self.key_cache[layer_id][:, start_pos:end_pos, :, :]
                new_k = key_cache[current_slot_mapping].unsqueeze(0)
                torch_npu.scatter_update_(k, indices, new_k, axis=-2)

                # copy value into datadist
                val = self.value_cache[layer_id][:, start_pos:end_pos, :, :]
                new_val = value_cache[current_slot_mapping].unsqueeze(0)
                torch_npu.scatter_update_(val, indices, new_val, axis=-2)

            # copy input into datadist
            inp = self.input_cache[0][:, start_pos:end_pos, :, :]
            new_inp = current_tokens.view(1, current_tokens.shape[0], 1, 1)
            torch_npu.scatter_update_(inp, indices, new_inp, axis=-2)

            # copy hidden into datadist
            hid = self.hidden_cache[0][:, start_pos:end_pos, :, :]
            hid_shape0, hid_shape1 = hidden_or_intermediate_states[
                start_pos:end_pos].shape
            new_hid = hidden_or_intermediate_states[start_pos:end_pos].view(
                1, hid_shape0, 1, hid_shape1)
            torch_npu.scatter_update_(hid, indices, new_hid, axis=-2)

        logger.info("[rank%d][P]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        # get model config
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = int(hidden_size / num_attention_heads)
        num_layer = end_layer - start_layer

        # get input_tensor_shape and hidden_shape
        input_shape = (1, input_tokens_tensor.shape[0], 1, 1)
        hidden_shape = (1, input_tokens_tensor.shape[0], 1, hidden_size)
        kv_shape = (1, input_tokens_tensor.shape[0], num_heads, head_size)

        kv_hidden_dtype = kv_caches[0].dtype
        input_dtype = torch.int32

        # Add LLM DataDist initialization
        key_desc = llm_datadist.CacheDesc(
            num_layer,
            kv_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[kv_hidden_dtype],
            seq_len_dim_index=-1)
        value_desc = llm_datadist.CacheDesc(
            num_layer,
            kv_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[kv_hidden_dtype],
            seq_len_dim_index=-1)
        input_desc = llm_datadist.CacheDesc(
            1,
            input_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[input_dtype],
            seq_len_dim_index=-1)
        hidden_desc = llm_datadist.CacheDesc(
            1,
            hidden_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[kv_hidden_dtype],
            seq_len_dim_index=-1)
        self.decode_key_buffer = self.llm_datadist_engine.kv_transfer.allocate_cache(
            key_desc)
        self.decode_value_buffer = self.llm_datadist_engine.kv_transfer.allocate_cache(
            value_desc)
        self.decode_input_buffer = self.llm_datadist_engine.kv_transfer.allocate_cache(
            input_desc)
        self.decode_hidden_buffer = self.llm_datadist_engine.kv_transfer.allocate_cache(
            hidden_desc)
        key_buffer_addrs = self.decode_key_buffer.per_device_tensor_addrs[0]
        value_buffer_addrs = self.decode_value_buffer.per_device_tensor_addrs[
            0]
        input_buffer_addrs = self.decode_input_buffer.per_device_tensor_addrs[
            0]
        hidden_buffer_addrs = self.decode_hidden_buffer.per_device_tensor_addrs[
            0]
        self.key_cache = torchair.llm_datadist.create_npu_tensors(
            key_desc.shape, kv_hidden_dtype, key_buffer_addrs)
        self.value_cache = torchair.llm_datadist.create_npu_tensors(
            value_desc.shape, kv_hidden_dtype, value_buffer_addrs)
        self.input_cache = torchair.llm_datadist.create_npu_tensors(
            input_desc.shape, input_dtype, input_buffer_addrs)
        self.hidden_cache = torchair.llm_datadist.create_npu_tensors(
            hidden_desc.shape, kv_hidden_dtype, hidden_buffer_addrs)

        key_cache_key = llm_datadist.CacheKeyByIdAndIndex(
            self.cluster.remote_cluster_id, 1, 0)
        value_cache_key = llm_datadist.CacheKeyByIdAndIndex(
            self.cluster.remote_cluster_id, 2, 0)
        input_cache_key = llm_datadist.CacheKeyByIdAndIndex(
            self.cluster.remote_cluster_id, 3, 0)
        hidden_cache_key = llm_datadist.CacheKeyByIdAndIndex(
            self.cluster.remote_cluster_id, 4, 0)

        self.llm_datadist_engine.kv_transfer.pull_cache(
            key_cache_key, self.decode_key_buffer, 0)
        self.llm_datadist_engine.kv_transfer.pull_cache(
            value_cache_key, self.decode_value_buffer, 0)
        self.llm_datadist_engine.kv_transfer.pull_cache(
            input_cache_key, self.decode_input_buffer, 0)
        self.llm_datadist_engine.kv_transfer.pull_cache(
            hidden_cache_key, self.decode_hidden_buffer, 0)

        keys = self.key_cache
        values = self.value_cache
        inputs = self.input_cache
        hidden = self.hidden_cache

        # enumerate different requests
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            num_computed_tokens = inputs[0][0, start_pos:end_pos, 0,
                                            0].shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):
                kv_cache = kv_caches[i - model_executable.model.start_layer]
                key_cache, value_cache = kv_cache[0], kv_cache[1]

                sliced_key = keys[i - model_executable.model.start_layer][
                    0, start_pos:end_pos, :, :]
                sliced_value = values[i - model_executable.model.start_layer][
                    0, start_pos:end_pos, :, :]

                torch_npu._npu_reshape_and_cache(
                    key=sliced_key,
                    value=sliced_value,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_indices=slot_mapping[start_pos:end_pos])

            hidden_or_intermediate_states_for_one_req.append(
                hidden[0][0, start_pos:end_pos, 0, :])

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.info(
                "[rank%d][D]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None
        else:
            logger.info(
                "[rank%d][D]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self, ):
        self.llm_datadist_engine.data_dist.unlink_clusters([self.cluster],
                                                           5000)