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
import hashlib
import os
import re
import struct
import subprocess
import time
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import torch_npu
import torchair  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

import llm_datadist  # type: ignore
from llm_datadist import LLMException, LLMStatusCode

import vllm_ascend.envs as envs

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
                "Please specify env PROMPT_DEVICE_ID and DECODE_DEVICE_ID")

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
        # Use the `kv_cache_layers` variable for the entire key-value cache, and
        # `kv_caches` for the current request's cache.
        kv_cache_layers = kv_caches
        kv_caches = []

        attn_metadata = model_input.attn_metadata
        query_start_loc = attn_metadata.query_start_loc.tolist()
        slot_mapping_flat = attn_metadata.slot_mapping.flatten()
        # Assuming that the order of request IDs in `request_ids_to_seq_ids`
        # matches the order in the batch. If this assumption is incorrect, a
        # more reliable method to determine the order of request IDs is
        # required.
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        # Get model config
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        num_layer = end_layer - start_layer
        # If use MLA, the kv cache shape is [num_blocks, block_size, num_heads,
        # head_dim], otherwise it is [2, num_blocks, block_size, num_heads,
        # head_dim].
        is_mla = len(kv_cache_layers[0].shape) == 4

        kv_hidden_dtype = kv_cache_layers[0].dtype
        assert kv_hidden_dtype == hidden_or_intermediate_states.dtype, \
            "KV cache and hidden states should have the same dtype"
        indices = torch.tensor([0], dtype=torch.int64).npu()
        for idx in range(len(query_start_loc) - 1):
            request_id = request_ids[idx]
            start_pos = query_start_loc[idx]
            end_pos = query_start_loc[idx + 1]
            req_slot_mapping = slot_mapping_flat[start_pos:end_pos]

            # Each request uses the same llm_datadist request_id, which needs to
            # be converted to an integer value.
            datadist_request_id = string_to_int64_hash(request_id)

            # Extract the kv caches of the current request from the vllm kv
            # caches.
            kv_caches = []
            for layer_id in range(start_layer, end_layer):
                kv_cache_layer = kv_cache_layers[layer_id - start_layer]
                kv_cache = self._extract_kv_from_layer(kv_cache_layer,
                                                       req_slot_mapping,
                                                       is_mla)
                kv_cache = kv_cache.unsqueeze(0)
                kv_caches.append(kv_cache)

            # Initialize the datadist kv cache buffer, and copy the generated kv
            # caches to the datadist kv cache buffer.
            kv_shape = kv_caches[0].shape
            kv_cache_keys = [
                llm_datadist.CacheKey(self.llm_datadist_engine.cluster_id,
                                      datadist_request_id, 1)
            ]
            kv_buffer, pushed_kv_caches = create_cache_tensors(
                self.llm_datadist_engine.kv_transfer, num_layer, kv_shape,
                kv_hidden_dtype, kv_cache_keys)
            for kv_cache in kv_caches:

                datadist_kv_cache = pushed_kv_caches[layer_id]
                torch_npu.scatter_update_(datadist_kv_cache,
                                          indices,
                                          kv_cache,
                                          axis=-2)

            # Get the current request's hidden state from the current batch, and
            # copy it to the datadist buffer.
            hid_shape0, hid_shape1 = hidden_or_intermediate_states[
                start_pos:end_pos].shape
            req_hidden_states = hidden_or_intermediate_states[
                start_pos:end_pos].view(1, hid_shape0, 1, hid_shape1)
            hidden_state_shape = tuple(req_hidden_states.shape)
            hidden_cache_keys = [
                llm_datadist.CacheKey(self.llm_datadist_engine.cluster_id,
                                      datadist_request_id, 2)
            ]
            hidden_buffer, pushed_hidden_states = create_cache_tensors(
                self.llm_datadist_engine.kv_transfer, 1, hidden_state_shape,
                kv_hidden_dtype, hidden_cache_keys)
            torch_npu.scatter_update_(pushed_hidden_states[0],
                                      indices,
                                      req_hidden_states,
                                      axis=-2)

            # Release reference count
            self.llm_datadist_engine.kv_transfer.deallocate_cache(kv_buffer)
            self.llm_datadist_engine.kv_transfer.deallocate_cache(
                hidden_buffer)

        # put prefill info to coordinator
        logger.info("[rank%d][P]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        # Use the `kv_cache_layers` variable for the entire key-value cache, and
        # `kv_caches` for the current request's cache.
        kv_cache_layers = kv_caches
        kv_caches = []

        bypass_model_exec = True

        attn_metadata = model_input.attn_metadata
        query_start_loc = attn_metadata.query_start_loc.tolist()
        slot_mapping_flat = attn_metadata.slot_mapping.flatten()
        # Assuming that the order of request IDs in `request_ids_to_seq_ids`
        # matches the order in the batch. If this assumption is incorrect, a
        # more reliable method to determine the order of request IDs is
        # required.
        request_ids = list(model_input.request_ids_to_seq_ids.keys())

        # Get model config
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer
        hidden_size = int(model_executable.model.config.hidden_size)
        num_layer = end_layer - start_layer
        # If use MLA, the kv cache shape is [num_blocks, block_size, num_heads,
        # head_dim], otherwise it is [2, num_blocks, block_size, num_heads,
        # head_dim].
        is_mla = len(kv_cache_layers[0].shape) == 4

        kv_hidden_dtype = kv_cache_layers[0].dtype
        indices = torch.tensor([0],
                               dtype=torch.int64,
                               device=torch.npu.current_device())

        kv_cache_layer_shape = kv_cache_layers[0].shape
        num_heads = int(kv_cache_layer_shape[-2])
        head_size = int(kv_cache_layer_shape[-1])

        hidden_or_intermediate_states_for_one_req = []
        for idx in range(len(query_start_loc) - 1):
            request_id = request_ids[idx]
            start_pos = int(query_start_loc[idx])
            end_pos = int(query_start_loc[idx + 1])
            slen = end_pos - start_pos
            req_slot_mapping = slot_mapping_flat[start_pos:end_pos]

            # Each request uses the same llm_datadist request_id, which needs to
            # be converted into an integer value.
            llm_datadist_request_id = string_to_int64_hash(request_id)
            remote_cluster_id = self.cluster.remote_cluster_id

            # Pull kv cache from prefill node by request
            if is_mla:
                kv_shape: Tuple[int, ...] = (1, slen, num_heads, head_size)
            else:
                kv_shape = (1, 2, slen, num_heads, head_size)
            kv_buffer, pulled_kv_caches = create_cache_tensors(
                self.llm_datadist_engine.kv_transfer, num_layer, kv_shape,
                kv_hidden_dtype)
            key_cache_key = llm_datadist.CacheKey(remote_cluster_id,
                                                  llm_datadist_request_id, 1)
            self.llm_datadist_engine.kv_transfer.pull_cache(
                key_cache_key, kv_buffer, 0)

            # Pull hidden states from prefill node by request
            hidden_shape = (1, slen, 1, hidden_size)
            hidden_buffer, pulled_hidden_states = create_cache_tensors(
                self.llm_datadist_engine.kv_transfer, 1, hidden_shape,
                kv_hidden_dtype)
            hidden_cache_key = llm_datadist.CacheKey(remote_cluster_id,
                                                     llm_datadist_request_id,
                                                     2)
            self.llm_datadist_engine.kv_transfer.pull_cache(
                hidden_cache_key, hidden_buffer, 0)

            # Check for any transmission failures; we need to redo the
            # forwarding to compute the missing states.
            if pulled_kv_caches is None or pulled_hidden_states is None:
                bypass_model_exec = False
                logger.error(
                    "[rank%d][D]: Failed to receive all KVs and hidden "
                    "states, redo model forwarding.",
                    torch.distributed.get_rank(),
                )
                break

            # Put received KV caches into paged memory
            for i in range(start_layer, end_layer):
                kv_cache_layer = kv_cache_layers[i - start_layer]
                pulled_kv_cache = pulled_kv_caches[i - start_layer]
                self._inject_kv_into_layer(kv_cache_layer, pulled_kv_cache,
                                           req_slot_mapping, is_mla)

            hidden_states = torch.empty_like(pulled_hidden_states[0])
            torch_npu.scatter_update_(hidden_states,
                                      indices,
                                      pulled_hidden_states[0],
                                      axis=1)
            hidden_or_intermediate_states_for_one_req.append(
                hidden_states[0, :, 0, :])

            # Release the reference count
            self.llm_datadist_engine.kv_transfer.deallocate_cache(kv_buffer)
            self.llm_datadist_engine.kv_transfer.deallocate_cache(
                hidden_buffer)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.info(
                "[rank%d][D]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.",
                torch.distributed.get_rank(),
            )
            hidden_or_intermediate_states = None
        else:
            logger.info(
                "[rank%d][D]: Successfully received all KVs and hidden "
                "states, skip model forwarding.",
                torch.distributed.get_rank(),
            )
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        self.llm_datadist_engine.data_dist.unlink_clusters([self.cluster],
                                                           5000)

    def _inject_kv_into_layer(
        self,
        dst_kv_cache_layer: torch.Tensor,
        pulled_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        is_mla: bool,
    ) -> None:
        """Inject the KV cache into the layer.

        Args:
            dst_kv_cache_layer (torch.Tensor): the destination KV cache
                layer. In shape [2, num_blocks, block_size, num_heads, head_dim]
                if not using MLA, [num_blocks, block_size, num_heads, head_dim]
                otherwise.
            src_kv_cache (torch.Tensor): the source KV cache. In shape
                [1, 2, num_tokens, num_heads, head_dim] if not using MLA, [1,
                num_tokens, num_heads, head_dim] otherwise.
            slot_mapping (torch.Tensor): the slot mapping. In shape
                [num_tokens].
        """
        # The performance of this function is suboptimal. Using
        # `torch_npu._npu_reshape_and_cache` or
        # `torch_npu._npu_reshape_and_cache_siso` could improve performance
        # significantly. However, attempts to use these methods have failed, and
        # the root cause remains unclear. The only available information is an
        # error log from the ATB log file, which states:
        # "ReshapeAndCacheOperation_1 invalid param, setup check fail, error
        # code: 13."

        # The pulled KV cache resides in the mbuf memory space and cannot be
        # directly copied to the kv_cache_layer. Therefore, it must first be
        # copied to a standard torch tensor using `scatter_update_`.
        kv_cache = torch.empty_like(pulled_kv_cache)
        indices = torch.tensor([0], dtype=torch.int64, device="npu")
        torch_npu.scatter_update_(kv_cache, indices, pulled_kv_cache, axis=-2)
        kv_cache = kv_cache.squeeze(0)

        dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
        if is_mla:
            block_size = dst_kv_cache_layer_shape[1]
            num_heads = dst_kv_cache_layer_shape[2]
            head_dim = dst_kv_cache_layer_shape[3]
            idx_for_copy = slot_mapping // block_size * block_size + slot_mapping % block_size
            dst_kv_cache_layer = dst_kv_cache_layer.view(
                -1, num_heads, head_dim)
            dst_kv_cache_layer[idx_for_copy, ...] = kv_cache
        else:
            block_size = dst_kv_cache_layer_shape[2]
            num_heads = dst_kv_cache_layer_shape[3]
            head_dim = dst_kv_cache_layer_shape[4]
            idx_for_copy = slot_mapping // block_size * block_size + slot_mapping % block_size
            dst_kv_cache_layer = dst_kv_cache_layer.view(
                2, -1, num_heads, head_dim)
            dst_kv_cache_layer[:, idx_for_copy, ...] = kv_cache

    def _extract_kv_from_layer(
        self,
        kv_cache_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        is_mla: bool,
    ) -> torch.Tensor:
        """Extract the KV cache from the layer.

        Assume the shape of the layer is [2, num_blocks, block_size, num_heads,
        head_dim] if MLA is not used, and [num_blocks, block_size, num_heads,
        head_dim] otherwise.
        """
        if is_mla:
            num_heads, head_dim = kv_cache_layer.shape[
                2], kv_cache_layer.shape[3]
            return kv_cache_layer.view(-1, num_heads, head_dim)[slot_mapping,
                                                                ...]

        num_heads, head_dim = kv_cache_layer.shape[2], kv_cache_layer.shape[3]
        return kv_cache_layer.view(2, -1, num_heads, head_dim)[:, slot_mapping,
                                                               ...]


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


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def create_cache_tensors(kv_transfer,
                         num_layer: int,
                         shape: Tuple[int, ...],
                         dtype: torch.dtype,
                         cache_keys=[]):
    cache_desc = llm_datadist.CacheDesc(num_layer,
                                        shape,
                                        TORCH_DTYPE_TO_NPU_DTYPE[dtype],
                                        seq_len_dim_index=-1)

    # At present, there is no method to determine the available space in the
    # mbuf memory. Therefore, we can only attempt to handle allocation failures;
    # if the failure is due to insufficient space, we pause briefly before
    # retrying until the allocation succeeds.
    while True:
        try:
            cache_buf = kv_transfer.allocate_cache(cache_desc, cache_keys)
            break
        except LLMException as e:
            if e.status_code == LLMStatusCode.LLM_DEVICE_OUT_OF_MEMORY:
                logger.warning(
                    "allocate_cache failed due to insufficient space in the mbuf memory."
                )
                time.sleep(0.03)  # wait for cache buf to be ready
            else:
                raise e

    cache_buf_addrs = cache_buf.per_device_tensor_addrs[0]
    cache_tensors = torchair.llm_datadist.create_npu_tensors(
        cache_desc.shape, dtype, cache_buf_addrs)
    return cache_buf, cache_tensors
