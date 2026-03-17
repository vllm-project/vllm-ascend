import time
from typing import Any

import requests
import torch
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

logger = init_logger(__name__)


class RForkTransferEngineBackendWorker:
    def __init__(self):
        self.rfork_transfer_engine = None
        self.rfork_transfer_engine_session_id = None
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        self._is_initialized = False
        self.init_transfer_engine()

    @staticmethod
    def _status_ok(ret: Any) -> bool:
        if ret is None:
            return True
        if isinstance(ret, bool):
            return ret
        if isinstance(ret, int):
            return ret == 0
        if hasattr(ret, "is_error"):
            try:
                return not bool(ret.is_error())
            except Exception:
                return False
        if hasattr(ret, "ok"):
            try:
                return bool(ret.ok())
            except Exception:
                return False
        if hasattr(ret, "code"):
            return getattr(ret, "code") == 0
        return True

    def init_transfer_engine(self):
        try:
            from yr.datasystem import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install @yuanrong-datasystem/transfer_engine first."
            ) from e

        self.rfork_transfer_engine = TransferEngine()
        local_hostname = join_host_port(get_ip(), get_open_port())
        ret = self.rfork_transfer_engine.initialize(
            local_hostname,
            int(torch.npu.current_device()),
            4,
        )
        if not self._status_ok(ret):
            raise RuntimeError(
                "TransferEngine initialization failed: "
                f"initialize({local_hostname}, "
                f"{int(torch.npu.current_device())}, 4) -> {ret}"
            )

        self.rfork_transfer_engine_session_id = local_hostname
        self._is_initialized = True

    def is_initialized(self) -> bool:
        return self._is_initialized

    def register_memory_region_v2(self, model):
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

        for address, size in weight_blocks_for_reg_mr:
            ret = self.rfork_transfer_engine.register_memory(address, size)
            if not self._status_ok(ret):
                logger.error(
                    "register_memory_region_v2 failed for address %s, size %s, ret: %s",
                    address,
                    size,
                    ret,
                )
                return False

        self.rfork_transfer_engine_weights_info_dict = weight_mr_dict
        self.registered_weight_blocks = weight_blocks_for_reg_mr

        logger.warning(
            "register_memory_region_v2 time: %.4fs",
            time.time() - start_reg_mr_tic,
        )
        return True

    def unregister_memory_region(self) -> bool:
        start_unreg_mr_tic = time.time()
        for address, _ in self.registered_weight_blocks:
            ret = self.rfork_transfer_engine.unregister_memory(address)
            if not self._status_ok(ret):
                logger.error("unregister memory failed for address %s, ret: %s", address, ret)
                return False
        self.rfork_transfer_engine_weights_info_dict = None
        self.registered_weight_blocks = []
        logger.warning(
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
        seed_url = f"http://{seed_instance_ip}:{seed_instance_service_port}"
        seed_session_id, seed_weight_info = get_remote_instance_transfer_engine_info(
            seed_url, local_seed_key
        )
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
        ret = self.rfork_transfer_engine.batch_transfer_sync_read(
            seed_session_id,
            client_ptr_list,
            seed_ptr_list,
            client_len_list,
        )
        if not self._status_ok(ret):
            logger.error("Failed to transfer weights from remote instance, ret=%s", ret)
            return False

        logger.warning("transfer weights time: %.4fs", time.time() - start_transfer_tic)
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
