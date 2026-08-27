import hashlib
import os
from dataclasses import dataclass
from typing import Any

import regex as re
import torch
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import split_host_port

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def _iter_slices(total: int, batch_size: int):
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield start, end


@dataclass
class YuanrongConfig:
    worker_addr: str
    enable_exclusive_connection: bool
    enable_remote_h2d: bool

    @staticmethod
    def load_from_env() -> "YuanrongConfig":
        worker_addr = os.getenv("DS_WORKER_ADDR")
        if not worker_addr:
            raise ValueError("Environment variable DS_WORKER_ADDR is required, expected format '<host>:<port>'.")

        return YuanrongConfig(
            worker_addr=worker_addr,
            enable_exclusive_connection=bool(int(os.getenv("DS_ENABLE_EXCLUSIVE_CONNECTION", "0"))),
            enable_remote_h2d=bool(int(os.getenv("DS_ENABLE_REMOTE_H2D", "0"))),
        )


class YuanrongHelper:
    _DS_KEY_MAX_LEN = 1024
    _DS_KEY_ALLOWED_PATTERN = re.compile(r"^[a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]+$")
    _DS_KEY_INVALID_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\-_!@#%\^\*\(\)\+\=\:;]")
    _DS_KEY_HASH_SUFFIX_LEN = 16

    def __init__(self, blob_cls, blob_list_cls):
        self._blob_cls = blob_cls
        self._blob_list_cls = blob_list_cls
        self._device_id: int | None = None

    def normalize_keys(self, keys: list[str]) -> list[str]:
        normalized: list[str] = []
        for key in keys:
            if len(key) <= self._DS_KEY_MAX_LEN and self._DS_KEY_ALLOWED_PATTERN.match(key):
                normalized.append(key)
                continue

            sanitized = self._DS_KEY_INVALID_CHAR_PATTERN.sub("_", key)
            hash_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            suffix = f"__{hash_digest[: self._DS_KEY_HASH_SUFFIX_LEN]}"
            max_prefix_len = self._DS_KEY_MAX_LEN - len(suffix)
            normalized.append(sanitized[:max_prefix_len] + suffix)
        return normalized

    def make_blob_lists(self, addrs_list: list[list[int]], sizes_list: list[list[int]]) -> list[Any]:
        total = len(addrs_list)
        if total != len(sizes_list):
            raise ValueError("Address list and size list length mismatch.")

        device_id = self._device_id
        if device_id is None:
            logger.error("Device id is not set. Check device initialization and configuration.")
            raise RuntimeError("Yuanrong backend device id is not initialized.")

        blob_lists: list[Any] = []
        for addrs, sizes in zip(addrs_list, sizes_list):
            if len(addrs) != len(sizes):
                raise ValueError("Address list and size list length mismatch.")
            blobs = [
                self._blob_cls(addr, size)  # type: ignore[misc]
                for addr, size in zip(addrs, sizes)
            ]
            blob_lists.append(
                self._blob_list_cls(device_id, blobs)  # type: ignore[misc]
            )
        return blob_lists


class YuanrongBackend(Backend):
    _DS_MAX_BATCH_KEYS = 10000

    def __init__(self, parallel_config: ParallelConfig):
        try:
            from yr.datasystem.hetero_client import Blob, DeviceBlobList, HeteroClient  # type: ignore[import-not-found]
            from yr.datasystem.kv_client import SetParam  # type: ignore[import-not-found]
            from yr.datasystem.object_client import WriteMode  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Please install openyuanrong-datasystem to use the yuanrong backend.") from exc

        self._helper = YuanrongHelper(Blob, DeviceBlobList)
        self._ds_set_param = SetParam()
        self._ds_set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT

        self.config = YuanrongConfig.load_from_env()
        try:
            host, port = split_host_port(self.config.worker_addr)
        except Exception as exc:
            raise ValueError(f"Invalid DS_WORKER_ADDR '{self.config.worker_addr}', expected '<host>:<port>'.") from exc
        self._hetero_client = HeteroClient(
            host,
            int(port),
            enable_exclusive_connection=self.config.enable_exclusive_connection,
            enable_remote_h2d=self.config.enable_remote_h2d,
        )
        self._hetero_client.init()
        self._is_a2 = get_ascend_device_type() in {AscendDeviceType.A2}
        self._registered_buffers: tuple[list[int], list[int]] | None = None
        self._buffers_registered = False

    def _ensure_device_ready(self):
        if self._helper._device_id is None:
            self.set_device()

    def set_device(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)
        self._helper._device_id = int(torch.npu.current_device())

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        self._registered_buffers = (list(ptrs), list(lengths))
        self._register_buffers_if_needed()

    def _register_buffers_if_needed(self):
        if self._is_a2:
            return
        if not self.config.enable_remote_h2d:
            return
        if self._registered_buffers is None or self._buffers_registered:
            return
        ptrs, lengths = self._registered_buffers
        self._hetero_client.pre_register_device_memory(ptrs, lengths)  # type: ignore[union-attr]
        self._buffers_registered = True

    def exists(self, keys: list[str]) -> list[int]:
        if len(keys) == 0:
            return []
        try:
            keys = self._helper.normalize_keys(keys)
            if len(keys) <= self._DS_MAX_BATCH_KEYS:
                exists = self._hetero_client.exist(keys)  # type: ignore[union-attr]
                return [1 if value else 0 for value in exists]
            results: list[int] = []
            for start, end in _iter_slices(len(keys), self._DS_MAX_BATCH_KEYS):
                exists = self._hetero_client.exist(keys[start:end])  # type: ignore[union-attr]
                results.extend(1 if value else 0 for value in exists)
            return results
        except Exception as exc:
            logger.error(
                "Failed to check keys. keys_count=%d, type=%s, error=%s. Check network and yuanrong service.",
                len(keys),
                type(exc).__name__,
                exc,
            )
            return [0] * len(keys)

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]) -> list[int] | None:
        if len(keys) == 0:
            return []
        failed_keys_for_log = keys
        try:
            self._ensure_device_ready()
            keys = self._helper.normalize_keys(keys)
            failed_keys_for_log = keys
            blob_lists = self._helper.make_blob_lists(addrs, sizes)
            failed_keys: list[str] = []
            if len(keys) <= self._DS_MAX_BATCH_KEYS:
                failed_keys = self._hetero_client.mget_h2d(  # type: ignore[union-attr]
                    keys, blob_lists, 0
                )
            else:
                for start, end in _iter_slices(len(keys), self._DS_MAX_BATCH_KEYS):
                    failed_keys_for_log = keys[start:end]
                    failed_keys.extend(
                        self._hetero_client.mget_h2d(  # type: ignore[union-attr]
                            keys[start:end], blob_lists[start:end], 0
                        )
                    )
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
        if len(keys) == 0:
            return
        failed_keys_for_log = keys
        try:
            self._ensure_device_ready()
            keys = self._helper.normalize_keys(keys)
            failed_keys_for_log = keys
            blob_lists = self._helper.make_blob_lists(addrs, sizes)
            if len(keys) <= self._DS_MAX_BATCH_KEYS:
                self._hetero_client.mset_d2h(  # type: ignore[union-attr]
                    keys, blob_lists, self._ds_set_param
                )
            else:
                for start, end in _iter_slices(len(keys), self._DS_MAX_BATCH_KEYS):
                    failed_keys_for_log = keys[start:end]
                    self._hetero_client.mset_d2h(  # type: ignore[union-attr]
                        keys[start:end], blob_lists[start:end], self._ds_set_param
                    )
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
