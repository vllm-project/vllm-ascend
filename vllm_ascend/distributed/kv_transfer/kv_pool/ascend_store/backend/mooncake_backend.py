# Standard
import functools
import json
import os
import threading
from dataclasses import dataclass
from typing import Any

import regex as re
import torch

# Third Party
from mooncake.store import ReplicateConfig  # type: ignore
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1.0 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB


@functools.lru_cache(maxsize=1)
def _mooncake_setup_supports_ssd_offload() -> bool:
    """True when installed Mooncake exposes SSD kwargs on setup() (v0.3.11+)."""
    from mooncake.store import MooncakeDistributedStore  # type: ignore

    setup = MooncakeDistributedStore.setup
    try:
        import inspect

        sig = inspect.signature(setup)
        return "enable_ssd_offload" in sig.parameters
    except (TypeError, ValueError):
        # pybind11 overloaded bindings often reject inspect.signature
        doc = setup.__doc__ or ""
        return "enable_ssd_offload" in doc


def _ssd_setup_kwargs(config: "MooncakeStoreConfig") -> dict[str, object]:
    """Keyword args for store.setup(); empty on old Mooncake or when SSD is off."""
    if not config.enable_ssd_offload:
        return {}
    if not _mooncake_setup_supports_ssd_offload():
        raise RuntimeError(
            "mooncake.json has enable_ssd_offload=true, but the installed "
            "Mooncake does not support enable_ssd_offload/ssd_offload_path in "
            "MooncakeDistributedStore.setup(). Upgrade Mooncake to v0.3.11 or "
            "later (see Mooncake ssd-offload.md Step 3A), or set "
            "enable_ssd_offload to false."
        )
    return {
        "enable_ssd_offload": config.enable_ssd_offload,
        "ssd_offload_path": config.ssd_offload_path,
    }


class MooncakeBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig, lazy_init: bool = False):
        self.config = MooncakeStoreConfig.load_from_env()
        if self.config.protocol != "ascend":
            raise NotImplementedError(f"MooncakeBackend does not support protocol {self.config.protocol!r}.")

        self.store: Any | None = None
        self.local_seg: str | None = None
        self._use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
        self._lazy_init = lazy_init and self._use_fabric_mem
        self._store_initialized = False
        self._store_init_lock = threading.Lock()

        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True

    def _ensure_initialized(self):
        if self._store_initialized:
            return

        with self._store_init_lock:
            if self._store_initialized:
                return

            logger.info("Initializing Mooncake store. metadata_server=%s", self.config.metadata_server)
            self.store = self._setup_store()
            self._store_initialized = True

    def _setup_store(self):
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        store = MooncakeDistributedStore()
        local_hostname = get_ip()
        ssd_kwargs = _ssd_setup_kwargs(self.config)
        # Each TP rank must use a separate SSD directory to avoid bucket file
        # collisions (independent BucketStorageBackend instances generate the
        # same bucket_id sequence and would overwrite each other's data).
        if ssd_kwargs and ssd_kwargs.get("ssd_offload_path"):
            local_rank = get_world_group().local_rank
            rank_path = os.path.join(str(ssd_kwargs["ssd_offload_path"]), f"rank_{local_rank}")
            try:
                os.makedirs(rank_path, exist_ok=True)
            except OSError as e:
                raise RuntimeError(f"Failed to create per-rank SSD offload directory: {rank_path!r} ({e})")
            ssd_kwargs["ssd_offload_path"] = rank_path
        # ASCEND_ENABLE_USE_FABRIC_MEM: Enable unified memory address direct transmission scheme
        # and only can be used for 800 I/T A3 series.
        # Required supporting hardware versions are as follows:
        if not self._use_fabric_mem:
            transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
            self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size,
                local_buffer_size=self.config.local_buffer_size,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
                engine=transfer_engine.get_engine(),
                **ssd_kwargs,
            )
        else:
            self.local_seg = local_hostname
            ret = store.setup(
                local_hostname=self.local_seg,
                metadata_server=self.config.metadata_server,
                global_segment_size=self.config.global_segment_size,
                local_buffer_size=0,
                protocol=self.config.protocol,
                rdma_devices=self.config.device_name,
                master_server_addr=self.config.master_server_address,
                **ssd_kwargs,
            )

        if ret != 0:
            msg = "Initialize mooncake failed."
            logger.error(
                "Initialize mooncake failed. ret=%d, metadata_server=%s. Check mooncake config and network.",
                ret,
                self.config.metadata_server,
            )
            raise RuntimeError(msg)
        if ssd_kwargs:
            logger.info(
                "Mooncake SSD offload enabled (Mode A): path=%s",
                self.config.ssd_offload_path,
            )
        return store

    def set_device(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        if not self._use_fabric_mem:
            local_hostname = get_ip()
            global_te.get_transfer_engine(local_hostname, device_name=None)
            global_te.register_buffer(ptrs, lengths)

    def exists(self, keys: list[str]) -> list[int]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "MooncakeBackend.exists called before store initialization; treating %d keys as missing.",
                len(keys),
            )
            return [0] * len(keys)
        assert self.store is not None
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        try:
            self._ensure_initialized()
            assert self.store is not None
            config = ReplicateConfig()
            if self.config.preferred_segment:
                config.preferred_segment = self.local_seg
            config.prefer_alloc_in_same_node = self.config.prefer_alloc_in_same_node
            res = self.store.batch_put_from_multi_buffers(keys, addrs, sizes, config)
            for value in res:
                if value < 0:
                    logger.error("Failed to put key. keys=%s, result=%s. Check memory and store capacity.", keys, res)
                    if self._lazy_init:
                        logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")
        except Exception as e:
            logger.error(
                "Failed to put key. keys=%s, type=%s, error=%s. Check store state and memory.",
                keys,
                type(e).__name__,
                e,
            )
            if self._lazy_init:
                logger.warning("First DSV4(compress) request failure is expected. This is normal behavior.")

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        if self._lazy_init and not self._store_initialized:
            logger.error("get() called before store init. keys=%s. Call put() first to trigger initialization.", keys)
            return
        assert self.store is not None
        logger.debug(
            "MooncakeBackend.get enter keys=%d sample_keys=%s",
            len(keys),
            keys[:3],
        )
        try:
            res = self.store.batch_get_into_multi_buffers(keys, addrs, sizes)
            res_list = list(res)
            logger.debug(
                "MooncakeBackend.get result keys=%d result_sample=%s negative_count=%d",
                len(keys),
                res_list[:12],
                sum(1 for value in res_list if value < 0),
            )
            for i, value in enumerate(res_list):
                if value < 0:
                    logger.error(
                        "Failed to get key. keys=%s, result=%s. Check key existence and memory state.", keys, res_list
                    )
                elif value > 0:
                    res_list[i] = 0
            return res_list
        except Exception as e:
            logger.error(
                "Failed to get key. keys=%s, type=%s, error=%s. Check store state and network.",
                keys,
                type(e).__name__,
                e,
            )
            return None


@dataclass
class MooncakeStoreConfig:
    metadata_server: str
    global_segment_size: int | str
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    preferred_segment: bool
    prefer_alloc_in_same_node: bool
    enable_ssd_offload: bool = False
    ssd_offload_path: str = ""

    def __post_init__(self) -> None:
        if not self.enable_ssd_offload:
            return
        if not self.ssd_offload_path:
            raise ValueError(
                "enable_ssd_offload is true but ssd_offload_path is empty. Set ssd_offload_path in mooncake.json."
            )
        if not os.path.isabs(self.ssd_offload_path):
            raise ValueError(f"ssd_offload_path must be an absolute path, got: {self.ssd_offload_path!r}")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        master_server_address = os.getenv("MOONCAKE_MASTER", None)
        global_segment_size_env = os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", None)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            global_segment_size=_parse_global_segment_size(
                global_segment_size_env
                if global_segment_size_env is not None
                else config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_global_segment_size(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            protocol=config.get("protocol", "ascend"),
            device_name=config.get("device_name", ""),
            master_server_address=master_server_address
            if master_server_address is not None
            else config.get("master_server_address"),
            preferred_segment=config.get("preferred_segment", False),
            prefer_alloc_in_same_node=config.get("prefer_alloc_in_same_node", True),
            enable_ssd_offload=bool(config.get("enable_ssd_offload", False)),
            ssd_offload_path=config.get("ssd_offload_path", ""),
        )

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError("The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeStoreConfig.from_file(config_path)


def _parse_global_segment_size(value) -> int:
    """
    Parse storage size strings with support for units: GB, MB, KB, B

    Args:
        value: Input value (int, str, or other convertible types)

    Returns:
        int: Size in bytes

    Raises:
        ValueError: For invalid format, missing number, or negative values
        TypeError: For unsupported input types
    """

    if isinstance(value, int):
        return value
    elif not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for global_segment_size: {type(value)}") from e

    cleaned_input = value.strip().lower()
    if not cleaned_input:
        raise ValueError("global segment size cannot be empty.")

    UNIT_MULTIPLIERS = {
        "gb": 1024**3,  # 1 GB = 1024^3 bytes
        "mb": 1024**2,  # 1 MB = 1024^2 bytes
        "kb": 1024,  # 1 KB = 1024 bytes
        "b": 1,  # 1 B = 1 byte
    }
    pattern = r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$"
    match = re.match(pattern, cleaned_input)

    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"

    multiplier = UNIT_MULTIPLIERS[unit]
    return _convert_to_bytes(number_str, multiplier, value)


def _convert_to_bytes(number_str: str, multiplier: int, original_input: str) -> int:
    """
    Convert numeric string to byte count

    Args:
        number_str: Numeric portion of input
        multiplier: Unit conversion factor
        original_input: Original input string (for error messages)

    Returns:
        int: Byte count

    Raises:
        ValueError: For invalid numbers or negative results
    """
    try:
        numeric_value = float(number_str)
    except ValueError:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{original_input}'")
    # Calculate byte count
    try:
        byte_count = int(numeric_value * multiplier)
    except OverflowError:
        raise ValueError(f"Storage size too large: '{original_input}'")
    return byte_count
