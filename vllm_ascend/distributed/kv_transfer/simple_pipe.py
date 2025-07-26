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

import threading
import time
from typing import Optional

import llm_datadist  # type: ignore
import msgpack  # type: ignore
import torch
import torch_npu
import torchair  # type: ignore
import zmq  # type: ignore
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import logger
from vllm.utils import get_ip

import vllm_ascend.envs as envs
from vllm_ascend.distributed.kv_transfer.utils import NPU_DTYPE_TO_TORCH_DTYPE


class SimplePipe(KVPipeBase):

    def __init__(
            self,
            rank,
            local_rank,
            kv_transfer_config,
            hostname: str = "",
            port_offset: int = 0,  # NPU offset in current P/D instance.
    ):
        self.rank = rank
        self.local_rank = local_rank
        # Currently for 1P1D situation, we use cluster_id=0 for both Prefill and Decode
        # Will change here in the future to support xPyD.
        self.cluster_id = 0
        self.config = kv_transfer_config
        kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config
        kv_role = kv_transfer_config.kv_role
        if kv_role == "kv_producer":
            self.role = llm_datadist.LLMRole.PROMPT
        elif kv_role == "kv_consumer":
            self.role = llm_datadist.LLMRole.DECODER
        else:
            raise NotImplementedError(
                "kv_role should be inside [kv_producer, kv_consumer]")

        prefill_device_ips = kv_connector_extra_config.get(
            "prefill_device_ips", None)
        decode_device_ips = kv_connector_extra_config.get(
            "decode_device_ips", None)
        if prefill_device_ips is None or decode_device_ips is None:
            raise ValueError(
                "Please specify prefill_device_ips and decode_device_ips"
                "in kv_transfer_config.kv_connector_extra_config")
        p_device_num = len(prefill_device_ips)
        d_device_num = len(decode_device_ips)
        # When number of devices in P and D is not equal,
        # we assume that device in D can be mapped to any device in P.
        self.p_device_rank = self.rank % p_device_num
        self.d_device_rank = self.rank % d_device_num

        self.prompt_ip_list = prefill_device_ips
        self.decode_ip_list = decode_device_ips
        self.llmdatadist_comm_port = kv_connector_extra_config.get(
            "llmdatadist_comm_port", 26000)
        # LLMDataDist initializing.
        self.data_dist = llm_datadist.LLMDataDist(self.role, self.cluster_id)
        self._prepare_data_dist()
        # Decoder needs to initialize and link cluster
        if self.role == llm_datadist.LLMRole.DECODER:
            self.cluster = self._make_cluster()
            _, ret = self.data_dist.link_clusters([self.cluster], 20000)
            logger.info(
                f"rank {self.rank}, local_rank {self.local_rank} link, ret={ret}"
            )

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + str(proxy_port)

        self._register_thread = None
        if port_offset == 0 and self.proxy_address != "":
            # Initialize zmq socket and register to proxy.
            # Note that only NPU 0 of each P/D instance register to proxy.
            if not hostname:
                hostname = get_ip()  # Get ip of current host.
            port = int(kv_transfer_config.kv_port) + port_offset
            if port == 0:
                raise ValueError("Port cannot be 0")
            self._hostname = hostname
            self._port = port
            # Each card corresponds to a ZMQ address.
            self.zmq_address = f"{self._hostname}:{self._port}"

            self.context = zmq.Context()  # type: ignore
            self.router_socket = self.context.socket(
                zmq.ROUTER)  # type: ignore
            self.router_socket.bind(f"tcp://{self.zmq_address}")
            # The `http_port` must be consistent with the serving port of OpenAI.
            self.http_address = (
                f"{self._hostname}:"
                f"{self.config.kv_connector_extra_config['http_port']}")
            self._register_thread = threading.Thread(
                target=self._register_to_proxy, daemon=True)
            self._register_thread.start()

    def _prepare_data_dist(self):
        options = {
            "llm.SyncKvCacheWaitTime": envs.LLMDATADIST_SYNC_CACHE_WAIT_TIME,
        }
        if self.role == llm_datadist.LLMRole.PROMPT:
            options["ge.exec.deviceId"] = str(self.local_rank)
            options["llm.listenIpInfo"] = (
                f"{self.prompt_ip_list[self.p_device_rank]}:{self.llmdatadist_comm_port}"
            )
        else:
            options["ge.exec.deviceId"] = str(self.local_rank)
        print(f"prepare datadist, options: {options}")
        self.data_dist.init(options)
        self.kv_transfer = self.data_dist.kv_cache_manager
        print(f"{self.rank} rank data dist is ready")

    def _make_cluster(self):
        cluster = llm_datadist.LLMClusterInfo()
        cluster.remote_cluster_id = self.cluster_id
        local_ip = self.decode_ip_list[self.d_device_rank]
        remote_ip = self.prompt_ip_list[self.p_device_rank]
        cluster.append_local_ip_info(local_ip, 0)
        cluster.append_remote_ip_info(remote_ip, self.llmdatadist_comm_port)
        return cluster

    def _register_to_proxy(self):
        sock = self.context.socket(zmq.DEALER)  # type: ignore
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)  # type: ignore
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def send_tensor(
        self,
        tensor: Optional[torch.Tensor],
        tensor_desc: llm_datadist.CacheDesc,
        tensor_key: llm_datadist.CacheKey,
    ) -> llm_datadist.Cache:
        buffer = self.kv_transfer.allocate_cache(tensor_desc, [tensor_key])
        buffer_addr = buffer.per_device_tensor_addrs[0]
        data_tensor = torchair.llm_datadist.create_npu_tensors(
            tensor_desc.shape, tensor.dtype, buffer_addr)[0]  # type: ignore
        update_indices = torch.tensor(
            [0] * tensor.shape[0],  # type: ignore
            dtype=torch.int64).npu()
        torch_npu.scatter_update_(data_tensor, update_indices, tensor, axis=-1)
        # Free cache_id of buffer, actual deallocate will happen after consumer performing pull_cache.
        self.kv_transfer.deallocate_cache(buffer)
        return buffer

    def recv_tensor(
        self,
        tensor_desc: llm_datadist.CacheDesc,
        tensor_key: llm_datadist.CacheKey,
    ) -> llm_datadist.Cache:
        """Note that this function only creates empty tensor on buffer addr and returns it."""
        tmp_buffer = self.kv_transfer.allocate_cache(tensor_desc)
        buffer_addr = tmp_buffer.per_device_tensor_addrs[0]
        data_tensor = torchair.llm_datadist.create_npu_tensors(
            tensor_desc.shape,
            NPU_DTYPE_TO_TORCH_DTYPE[tensor_desc.data_type],
            buffer_addr,
        )[0]
        self.kv_transfer.pull_cache(tensor_key, tmp_buffer, 0)
        # tmp_buffer is allocated without key and will be deallocated here immediately.
        # Free buffer here will cause accuracy problem.
        # self.kv_transfer.deallocate_cache(tmp_buffer)
        return tmp_buffer, data_tensor

    def deallocate_buffer(self, buffer: llm_datadist.Cache):
        self.kv_transfer.deallocate_cache(buffer)

    def close(self):
        self.data_dist.unlink_clusters([self.cluster], 5000)
