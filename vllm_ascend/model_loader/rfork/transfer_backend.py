#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import time
from typing import Any

import requests
import torch
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port


class RForkTransferBackend:
    def __init__(self):
        self.rfork_transfer_engine: Any | None = None
        self.rfork_transfer_engine_session_id = None
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        self._is_initialized = False
        self.init_transfer_engine()

    def init_transfer_engine(self):
        try:
            from yr.datasystem import TransferEngine  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("Please install @yuanrong-datasystem/transfer_engine first.") from e

        transfer_engine = TransferEngine()
        local_hostname = join_host_port(get_ip(), get_open_port())
        ret = transfer_engine.initialize(local_hostname, "ascend", f"npu:{torch.npu.current_device()}")
        if ret.is_error():
            raise RuntimeError(
                "TransferEngine initialization failed: "
                f"initialize({local_hostname}, ascend"
                f"npu:{int(torch.npu.current_device())}) -> {ret.to_string()}"
            )

        self.rfork_transfer_engine = transfer_engine
        self.rfork_transfer_engine_session_id = local_hostname
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def _get_transfer_engine(self) -> Any:
        if self.rfork_transfer_engine is None:
            raise RuntimeError("TransferEngine is not initialized.")
        return self.rfork_transfer_engine

    def register_memory_region(self, model):
        transfer_engine = self._get_transfer_engine()
        start_reg_mr_tic = time.time()

        weight_mr_dict = {}
        weight_addr_set = set()
        for name, weight in model.named_parameters():
            weight_mr_dict[name] = (
                weight.data_ptr(),
                weight.numel(),
                weight.element_size(),
            )
            weight_addr_set.add(weight.data_ptr())

        memory_snapshot = torch.npu.memory.memory_snapshot()
        weight_blocks_for_reg_mr = []
        for segment in memory_snapshot:
            current_weight_block = None
            for block in segment.get("blocks", []):
                address = block.get("address", -1)
                size = block.get("size", -1)
                state = block.get("state", "")
                if address < 0 or size < 0 or state == "":
                    continue
                if state == "active_allocated" and address in weight_addr_set:
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
            if current_weight_block is not None:
                weight_blocks_for_reg_mr.append(current_weight_block)

        addresses, sizes = zip(*weight_blocks_for_reg_mr) if weight_blocks_for_reg_mr else ((), ())
        ret = transfer_engine.batch_register_memory(addresses, sizes)
        if ret.is_error():
            logger.error(
                "batch_register_memory failed for %d blocks, ret: %s",
                len(weight_blocks_for_reg_mr),
                ret.to_string(),
            )
            return False

        self.rfork_transfer_engine_weights_info_dict = weight_mr_dict
        self.registered_weight_blocks = weight_blocks_for_reg_mr

        logger.info(
            "register_memory_region time: %.4fs",
            time.time() - start_reg_mr_tic,
        )
        return True

    def unregister_memory_region(self) -> bool:
        transfer_engine = self._get_transfer_engine()
        start_unreg_mr_tic = time.time()
        ret = transfer_engine.batch_unregister_memory([address for address, _ in self.registered_weight_blocks])
        if ret.is_error():
            logger.error(
                "batch_unregister_memory failed for %d blocks, ret: %s",
                len(self.registered_weight_blocks),
                ret.to_string(),
            )
            return False
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        logger.info(
            "unregister_memory_region time: %.4fs",
            time.time() - start_unreg_mr_tic,
        )
        return True

    def recv_from_source(
        self,
        model,
        seed_instance_ip,
        seed_instance_service_port,
        local_seed_key,
    ):
        transfer_engine = self._get_transfer_engine()
        seed_url = f"http://{seed_instance_ip}:{seed_instance_service_port}"
        seed_session_id, seed_weight_info = get_remote_instance_transfer_engine_info(seed_url, local_seed_key)
        if seed_session_id is None or seed_weight_info is None:
            logger.error("Cannot get transfer engine session or weight info.")
            return False

        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        for name, tensor in model.named_parameters():
            weight_info = seed_weight_info.get(name, None)
            if weight_info is None:
                logger.error("Cannot find weight info for %s.", name)
                return False

            seed_ptr, seed_len, seed_size = weight_info
            if seed_len != tensor.numel() or seed_size != tensor.element_size():
                logger.error(
                    "Weight info mismatch for %s, expected (%s, %s), got (%s, %s)",
                    name,
                    seed_len,
                    seed_size,
                    tensor.numel(),
                    tensor.element_size(),
                )
                return False

            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(tensor.data_ptr())
            client_len_list.append(tensor.numel() * tensor.element_size())

        start_transfer_tic = time.time()
        ret = transfer_engine.batch_transfer_sync_read(
            seed_session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list,
        )
        if ret.is_error():
            logger.error("Failed to transfer weights from remote instance, ret=%s", ret.to_string())
            return False

        logger.info("transfer weights time: %.4fs", time.time() - start_transfer_tic)
        return True


def get_remote_instance_transfer_engine_info(seed_url: str, local_seed_key: str):
    try:
        response = requests.get(
            f"{seed_url}/get_rfork_transfer_engine_info",
            params={"seed_key": local_seed_key},
        )
        if response.status_code != 200:
            logger.error("request.get failed: %s", response.status_code)
            return None, None

        data = response.json()
        info = data.get("rfork_transfer_engine_info", None)
        if info is not None and isinstance(info, list) and len(info) == 2:
            return info[0], info[1]

        logger.error("Failed to get `rfork_transfer_engine_info` in response.")
        return None, None
    except Exception as e:
        logger.error("Exception: %s", e)
        return None, None
