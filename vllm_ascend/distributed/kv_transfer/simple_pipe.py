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
import json
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
from vllm_ascend.distributed.kv_transfer.utils import NPU_DTYPE_TO_TORCH_DTYPE, get_machine_type


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
        self.config = kv_transfer_config
        kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config
        kv_role = kv_transfer_config.kv_role
        if kv_role == "kv_producer":
            self.role = llm_datadist.LLMRole.PROMPT
        elif kv_role == "kv_consumer":
            self.role = llm_datadist.LLMRole.DECODER
        else:
            raise NotImplementedError(
                "kv_role should be inside [kv_producer, kv_consumer]"
            )
        self.machine_type = get_machine_type()

        self.llmdatadist_comm_port = kv_connector_extra_config.get(
            "llmdatadist_comm_port", 26000
        )
        global_rank_table = self._read_rank_table()
        p_device_num = len(global_rank_table["prefill_device_list"])
        d_device_num = len(global_rank_table["decode_device_list"])
        # When number of devices in P and D is not equal,
        # we assume that device in D can be mapped to any device in P.
        self.p_device_rank = self.rank % p_device_num
        self.d_device_rank = self.rank % d_device_num
        prefill_cluster_id = global_rank_table["prefill_device_list"][self.p_device_rank]["cluster_id"]
        decode_cluster_id = global_rank_table["decode_device_list"][self.d_device_rank]["cluster_id"]
        self.prefill_cluster_id = int(prefill_cluster_id)
        self.decode_cluster_id = int(decode_cluster_id)
        if self.role == llm_datadist.LLMRole.PROMPT:
            self.cluster_id = self.prefill_cluster_id
        else:
            self.cluster_id = self.decode_cluster_id
        # LLMDataDist initializing.
        self.data_dist = llm_datadist.LLMDataDist(self.role, self.cluster_id)
        self._prepare_data_dist()
        
        self.comm_id = self._make_cluster(global_rank_table)

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
        llm_config = llm_datadist.LLMConfig()
        llm_config.device_id = self.local_rank
        llm_config.enable_switch_role = True
        llm_config.enable_cache_manager = True
        llm_config.sync_kv_timeout = envs.LLMDATADIST_SYNC_CACHE_WAIT_TIME
        llm_config.enable_remote_cache_accessible = True
        llm_config.mem_pool_cfg = "{\"memory_size\": 18737418240, \"page_shift\": 16}"
        options = llm_config.generate_options()
        print(f"prepare datadist, options: {options}")
        self.data_dist.init(options)
        self.cache_manager = self.data_dist.cache_manager
        print(f"{self.rank} rank data dist is ready")

    def _read_rank_table(self):
        assert (
            envs.DISAGGREGATED_RPEFILL_RANK_TABLE_PATH
        ), "Please set path of rank_table to env variable DISAGGREGATED_RPEFILL_RANK_TABLE_PATH"
        rank_table_path = envs.DISAGGREGATED_RPEFILL_RANK_TABLE_PATH
        with open(rank_table_path, "r", encoding="utf-8") as f:
            global_rank_table = json.load(f)
        # global_rank_table = json.dumps(global_rank_table)
        return global_rank_table

    def _prepare_link_info(self, global_rank_table):
        # This function reads global rank table file and generates
        # local rank_table needed for link cluster
        decode_device_list = global_rank_table["decode_device_list"]
        prefill_device_list = global_rank_table["prefill_device_list"]
        decode_device_info = decode_device_list[self.d_device_rank]
        prefill_device_info = prefill_device_list[self.p_device_rank]
        decode_device_ip = decode_device_info["device_ip"]
        prefill_device_ip = prefill_device_info["device_ip"]
        # Communation range name.
        comm_name = f"pd_comm_{prefill_device_ip}_{decode_device_ip}"
        cluster_rank_info = {
            self.prefill_cluster_id: 0,
            self.decode_cluster_id: 1,
        }
        rank_table = {}
        version = "1.2" if self.machine_type == "A3" else "1.0"
        server_count = (
            1
            if prefill_device_info["server_id"]
            == decode_device_info["server_id"]
            else 2
        )
        rank_table["version"] = version
        rank_table["server_count"] = str(server_count)
        rank_table["status"] = "completed"
        prefill_server_device_info = {
            "device": [
                {
                    "device_id": prefill_device_info["device_id"],
                    "device_ip": prefill_device_info["device_ip"],
                    **({"super_device_id": prefill_device_info["super_device_id"]} if self.machine_type == "A3" else {}),
                    "rank_id": "0",
                }
            ],
            "server_id": prefill_device_info["server_id"],
        }
        rank_table["server_list"] = [prefill_server_device_info]
        if server_count == 2:
            decode_server_device_info = {
                "device": [
                    {
                        "device_id": decode_device_info["device_id"],
                        "device_ip": decode_device_info["device_ip"],
                        **({"super_device_id": decode_device_info["super_device_id"]} if self.machine_type == "A3" else {}),
                        "rank_id": "1",
                    }
                ],
                "server_id": decode_device_info["server_id"],
            }
            rank_table["server_list"].append(decode_server_device_info)
        else:
            decode_device_server_info = {
                "device_id": decode_device_info["device_id"],
                "device_ip": decode_device_info["device_ip"],
                **({"super_device_id": decode_device_info["super_device_id"]} if self.machine_type == "A3" else {}),
                "rank_id": "1",
            }
            rank_table["server_list"][0]["device"].append(
                decode_device_server_info
            )
        if version == "1.2":
            super_pod_list = []
            prefill_super_pod_info = {
                "super_pod_id": prefill_device_info["super_pod_id"],
                "server_list": [
                    {"server_id": prefill_device_info["server_id"]}
                ],
            }
            super_pod_list.append(prefill_super_pod_info)
            if (
                decode_device_info["super_pod_id"]
                == prefill_device_info["super_pod_id"]
            ):
                if server_count == 2:
                    super_pod_list[0]["server_list"].append(
                        {"server_id": decode_device_info["server_id"]}
                    )
            else:
                decode_super_pod_info = {
                    "super_pod_id": decode_device_info["super_pod_id"],
                    "server_list": [
                        {"server_id": decode_device_info["server_id"]}
                    ],
                }
                super_pod_list.append(decode_super_pod_info)
            rank_table["super_pod_list"] = super_pod_list
        return comm_name, cluster_rank_info, rank_table

    def _make_cluster(self, global_rank_table):
        comm_name, cluster_rank_info, rank_table = self._prepare_link_info(global_rank_table)
        comm_id = self.data_dist.link(comm_name, cluster_rank_info, json.dumps(rank_table))
        while True:
            ret = self.data_dist.query_register_mem_status(comm_id)
            if ret == llm_datadist.RegisterMemStatus.OK:
                logger.info(f"init link suc, comm_id: {comm_id}")
                break
            elif ret == llm_datadist.RegisterMemStatus.FAILED:
                logger.error(f"init link failed, comm_id: {comm_id}")
                raise RuntimeError("link failed")
            logger.info("Check query_register_mem_status again...")
            time.sleep(1)
        return comm_id

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
        tensor_key: llm_datadist.BlocksCacheKey,
    ) -> llm_datadist.Cache:
        buffer_of_blocks = self.cache_manager.allocate_blocks_cache(tensor_desc, tensor_key)
        buffer_addr = buffer_of_blocks.tensor_addrs
        data_tensor = torchair.llm_datadist.create_npu_tensors(
            tensor_desc.shape, tensor.dtype, buffer_addr
        )[
            0
        ]  # type: ignore
        update_indices = torch.tensor(
            [0] * tensor.shape[0], dtype=torch.int64  # type: ignore
        ).npu()
        torch_npu.scatter_update_(data_tensor, update_indices, tensor, axis=-1)
        return buffer_of_blocks

    def recv_tensor(
        self,
        tensor_desc: llm_datadist.CacheDesc,
        tensor_key: llm_datadist.BlocksCacheKey,
    ) -> llm_datadist.Cache:
        """Note that this function only creates empty tensor on buffer addr and returns it."""
        tmp_blocks_buffer = self.cache_manager.allocate_blocks_cache(tensor_desc)
        buffer_addr = tmp_blocks_buffer.tensor_addrs
        data_tensor = torchair.llm_datadist.create_npu_tensors(
            tensor_desc.shape,
            NPU_DTYPE_TO_TORCH_DTYPE[tensor_desc.data_type],
            buffer_addr,
        )[0]
        num_blocks = tensor_desc.shape[0]
        block_ids = list(range(num_blocks))
        self.cache_manager.pull_blocks(tensor_key, tmp_blocks_buffer, block_ids, block_ids)
        return tmp_blocks_buffer, data_tensor

    def deallocate_buffer(self, buffer: llm_datadist.Cache):
        self.cache_manager.deallocate_blocks_cache(buffer)

    def close(self):
        self.data_dist.unlink(self.comm_id)
