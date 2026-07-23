import json
import os
from dataclasses import dataclass

import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import split_host_port

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend


@dataclass
class YuanrongConfig:
    worker_addr: str
    enable_remote_h2d: bool
    remote_h2d_transport_backend: str
    enable_fabric_mem: bool
    connect_timeout_ms: int = 9000
    request_timeout_ms: int = 0
    get_sub_timeout_ms: int = 0

    @staticmethod
    def from_file(file_path: str) -> "YuanrongConfig":
        with open(file_path) as f:
            config = json.load(f)
        return YuanrongConfig(
            worker_addr=config.get("worker_addr", ""),
            enable_remote_h2d=bool(config.get("enable_remote_h2d", False)),
            remote_h2d_transport_backend=config.get("remote_h2d_transport_backend", "HIXL"),
            enable_fabric_mem=bool(config.get("enable_fabric_mem", False)),
            connect_timeout_ms=config.get("connect_timeout_ms", 9000),
            request_timeout_ms=config.get("request_timeout_ms", 0),
            get_sub_timeout_ms=config.get("get_sub_timeout_ms", 0),
        )

    @staticmethod
    def load_from_env() -> "YuanrongConfig":
        config_path = os.getenv("YR_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'YR_CONFIG_PATH' is not set.")
        return YuanrongConfig.from_file(config_path)


class YuanrongBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig):
        try:
            from yr.datasystem.hetero_client import HeteroClient  # type: ignore[import-not-found]
            from yr.datasystem.kv_client import SetParam  # type: ignore[import-not-found]
            from yr.datasystem.object_client import WriteMode  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Please install openyuanrong-datasystem by following the instructions at "
                "https://atomgit.com/openeuler/yuanrong-datasystem/blob/master/docs/source_zh_cn/installation/installation_linux.md "  # noqa: E501
                "to run vLLM with AscendStoreConnector."
            ) from exc

        self._ds_set_param = SetParam()
        self._ds_set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

        self.config = YuanrongConfig.load_from_env()
        try:
            host, port = split_host_port(self.config.worker_addr)
        except Exception as exc:
            raise ValueError(
                f"Invalid worker_addr {self.config.worker_addr} in yuanrong config, expected '<host>:<port>'."
            ) from exc
        self.store = HeteroClient(
            host,
            int(port),
            connect_timeout_ms=self.config.connect_timeout_ms,
            req_timeout_ms=self.config.request_timeout_ms,
            enable_remote_h2d=self.config.enable_remote_h2d,
        )
        self.store.init()
        self._needs_dev_mem_pregister = (
            self.config.enable_remote_h2d
            and self.config.remote_h2d_transport_backend == "HIXL"
            and not self.config.enable_fabric_mem
        )
        self._registered_buffers: tuple[list[int], list[int]] | None = None
        self._buffers_registered = False

    def set_device(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        self._registered_buffers = (list(ptrs), list(lengths))
        self._register_buffers_if_needed()

    def _register_buffers_if_needed(self):
        if not self._needs_dev_mem_pregister:
            return
        if self._registered_buffers is None or self._buffers_registered:
            return
        ptrs, lengths = self._registered_buffers
        assert self.store is not None
        self.store.pre_register_device_memory(ptrs, lengths)
        self._buffers_registered = True

    def exists(self, keys: list[str]) -> list[int]:
        assert self.store is not None
        try:
            return self.store.batch_is_exist(keys)
        except Exception as exc:
            logger.error(
                "Failed to check keys. keys_count=%d, type=%s, error=%s. Check network and yuanrong service.",
                len(keys),
                type(exc).__name__,
                exc,
            )
            return [0] * len(keys)

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int] | None:
        assert self.store is not None
        failed_keys_for_log = keys
        try:
            failed_keys = self.store.mget_h2d_from_multi_buffers(keys, addrs, sizes, self.config.get_sub_timeout_ms)
            if failed_keys:
                logger.error(
                    "Failed to get %d keys out of %d. Check key existence and memory state.",
                    len(failed_keys),
                    len(keys),
                )
                logger.debug("Failed to get key details. failed_keys=%s", failed_keys)
            failed_set = set(failed_keys)
            return [1 if k in failed_set else 0 for k in keys]
        except Exception as exc:
            logger.error(
                "Failed to get %d keys out of %d. Check network and yuanrong service.",
                len(failed_keys_for_log),
                len(keys),
            )
            logger.debug(
                "Failed to get key details. keys=%s, type=%s, error=%s",
                failed_keys_for_log,
                type(exc).__name__,
                exc,
            )
            return None

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        assert self.store is not None
        failed_keys_for_log = keys
        try:
            self.store.mset_d2h_from_multi_buffers(keys, addrs, sizes, self._ds_set_param)
        except Exception as exc:
            logger.error(
                "Failed to put %d keys out of %d. Check network and yuanrong service.",
                len(failed_keys_for_log),
                len(keys),
            )
            logger.debug(
                "Failed to put key details. keys=%s, type=%s, error=%s",
                failed_keys_for_log,
                type(exc).__name__,
                exc,
            )
