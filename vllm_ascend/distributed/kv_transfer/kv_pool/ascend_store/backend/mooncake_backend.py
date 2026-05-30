# Standard
import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import regex as re
import torch

# Third Party
from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

import vllm_ascend.envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend
from vllm_ascend.distributed.kv_transfer.utils import mooncake_rdma_utils
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824  # 1.0 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB

# Mirrors Mooncake's FileStorageConfig::local_buffer_size on the owner side.
# When ``enable_offload`` is on, the owner allocates this DirectIO staging
# buffer to bounce data between RDMA pool and SSD. The worker must size
# each GET sub-batch to fit within (roughly) this buffer.
DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES = 1280 * 1024 * 1024

# Mirrors DirectIO alignment in Mooncake's AllocateBatch.
_DIRECT_IO_ALIGNMENT = 4096
_DIRECT_IO_PADDING_BYTES = 2 * _DIRECT_IO_ALIGNMENT


# ``embedded``: each rank contributes ``global_segment_size`` in-process.
# ``standalone-store``: rank contributes 0; an external mooncake_client
# process owns the pool and the SSD tier (disk offload lives here).
MooncakeMode = Literal["embedded", "standalone-store"]


# ---------------------------------------------------------------------------
# Disk-staging budget helpers (adapted from upstream worker.py)
# ---------------------------------------------------------------------------
def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _estimate_disk_offload_staging_bytes(size_list: list[int]) -> int:
    """Rough upper bound on the owner-side DirectIO staging bytes a single
    key takes when its scatter-gather slices have the given sizes."""
    data_size = sum(size_list)
    return _align_up(data_size, _DIRECT_IO_ALIGNMENT) + _DIRECT_IO_PADDING_BYTES


def _get_usable_disk_offload_buffer_budget_bytes(raw_budget_bytes: int) -> int:
    ratio = ascend_envs.VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO
    return max(1, int(raw_budget_bytes * ratio))


def _split_disk_offload_load_batches(
    keys: list[str],
    addrs: list[list[int]],
    sizes: list[list[int]],
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]]]], str | None]:
    """Split a GET into sub-batches that fit the owner's staging buffer.

    ``addrs[i]`` / ``sizes[i]`` are scatter-gather lists (per-layer / K-V
    slices) for key ``i``. Returns ``(batches, oversize_key)``:

    * Each batch's total staging-bytes estimate is ``<= usable_budget_bytes``.
    * If any single key exceeds ``raw_budget_bytes`` it is **un-loadable**
      via the owner's staging buffer; the function returns ``([], key)`` so
      the caller can skip the request with a clear error.
    """
    batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
    batch_keys: list[str] = []
    batch_addrs: list[list[int]] = []
    batch_sizes: list[list[int]] = []
    batch_bytes = 0

    for key, addr, size in zip(keys, addrs, sizes, strict=True):
        key_bytes = _estimate_disk_offload_staging_bytes(size)
        if key_bytes > raw_budget_bytes:
            return [], key
        if key_bytes > usable_budget_bytes:
            if batch_keys:
                batches.append((batch_keys, batch_addrs, batch_sizes))
                batch_keys, batch_addrs, batch_sizes = [], [], []
                batch_bytes = 0
            # Single key that exceeds the soft cap but fits the hard cap
            # is sent on its own — still legal, just won't co-batch.
            batches.append(([key], [addr], [size]))
            continue
        if batch_keys and batch_bytes + key_bytes > usable_budget_bytes:
            batches.append((batch_keys, batch_addrs, batch_sizes))
            batch_keys, batch_addrs, batch_sizes = [], [], []
            batch_bytes = 0
        batch_keys.append(key)
        batch_addrs.append(addr)
        batch_sizes.append(size)
        batch_bytes += key_bytes

    if batch_keys:
        batches.append((batch_keys, batch_addrs, batch_sizes))
    return batches, None


# ---------------------------------------------------------------------------
# Replica-tier classification (observability for disk offload GETs)
# ---------------------------------------------------------------------------
def _call_replica_predicate(replica_desc: Any, method_name: str) -> bool:
    method = getattr(replica_desc, method_name, None)
    if method is None:
        return False
    try:
        return bool(method())
    except Exception:  # noqa: BLE001 — best-effort introspection
        return False


def _classify_replica_tier(replica_descs: Any) -> str:
    if not replica_descs:
        return "unknown"
    try:
        replica_desc = replica_descs[0]
    except (IndexError, KeyError, TypeError):
        return "unknown"
    if _call_replica_predicate(replica_desc, "is_memory_replica"):
        return "memory"
    if _call_replica_predicate(replica_desc, "is_disk_replica") or _call_replica_predicate(
        replica_desc, "is_local_disk_replica"
    ):
        return "disk"
    return "unknown"


def _get_replica_tiers_by_key(store: Any, keys: list[str]) -> dict[str, str]:
    """Return ``key -> tier`` for the given keys.

    Tolerates two return shapes from ``store.batch_get_replica_desc``:
    a mapping (preferred, dict-like) and a list parallel to ``keys``.
    Unknown shapes or per-key errors degrade to "unknown" without
    interrupting tier-log emission.
    """
    tiers_by_key = {key: "unknown" for key in keys}
    try:
        replica_descs_by_key = store.batch_get_replica_desc(keys)
    except Exception as e:  # noqa: BLE001 — older mooncake may lack the API
        logger.warning(
            "Failed to get Mooncake replica descriptors for tier logging "
            "(batch_keys=%d, error=%s); marking tiers unknown",
            len(keys),
            e,
        )
        return tiers_by_key

    # Prefer mapping access; fall back to positional indexing for list-like
    # returns so we don't silently mark every entry "unknown" when the
    # backend gives us a parallel list.
    is_mapping = hasattr(replica_descs_by_key, "get")
    for index, key in enumerate(keys):
        if is_mapping:
            replica_descs = replica_descs_by_key.get(key)
        else:
            try:
                replica_descs = replica_descs_by_key[index]
            except (IndexError, KeyError, TypeError):
                replica_descs = None
        tiers_by_key[key] = _classify_replica_tier(replica_descs)
    return tiers_by_key


def _log_mooncake_load_tier_summary(
    batch_keys: list[str],
    load_results: list[int],
    tiers_by_key: dict[str, str],
) -> None:
    tier_counts = {"memory": 0, "disk": 0, "unknown": 0}
    bytes_by_tier = {"memory": 0, "disk": 0, "unknown": 0}
    success_keys = 0
    failed_keys = 0
    for index, key in enumerate(batch_keys):
        tier = tiers_by_key.get(key, "unknown")
        if tier not in tier_counts:
            tier = "unknown"
        tier_counts[tier] += 1
        value = load_results[index] if index < len(load_results) else -1
        if value >= 0:
            success_keys += 1
            bytes_by_tier[tier] += int(value)
        else:
            failed_keys += 1
    logger.info(
        "Mooncake load tier summary: batch_keys=%d "
        "memory_keys=%d disk_keys=%d unknown_keys=%d "
        "success_keys=%d failed_keys=%d bytes_by_tier=%s",
        len(batch_keys),
        tier_counts["memory"],
        tier_counts["disk"],
        tier_counts["unknown"],
        success_keys,
        failed_keys,
        bytes_by_tier,
    )


class MooncakeBackend(Backend):
    def __init__(self, parallel_config: ParallelConfig, **kwargs: Any):
        """Mooncake KV-store backend with optional disk-offload support.

        ``kwargs`` may include ``kv_connector_extra_config`` (the
        ``kv_connector_extra_config`` dict from the engine's
        ``kv_transfer_config``); when supplied, it's consulted for the
        ``preferred_segment`` override that pins this rank's PUTs to a
        specific owner segment (required for SSD-tier replication).
        """
        try:
            from mooncake.store import MooncakeDistributedStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector."
            ) from e

        # ``ReplicateConfig`` is the newer (≥ mooncake disk-offload) API
        # used to pin PUTs to a specific owner segment. Older mooncake
        # versions don't expose it; treat as optional.
        try:
            from mooncake.store import ReplicateConfig  # type: ignore
        except ImportError:
            ReplicateConfig = None  # type: ignore[assignment,misc]

        extra_config: dict[str, Any] = kwargs.get("kv_connector_extra_config") or {}

        self.config = MooncakeStoreConfig.load_from_env()
        # Per-NPU RNIC selection when ``device_name`` is a CSV. Mooncake's
        # own auto-pick can converge multiple ranks onto the same NIC.
        self.config.device_name = mooncake_rdma_utils.get_configured_worker_rnic(
            protocol=self.config.protocol,
            configured_device=self.config.device_name,
        )

        self.store = MooncakeDistributedStore()
        if self.config.protocol == "ascend":
            base_hostname = get_ip()
            local_hostname = mooncake_rdma_utils.get_requester_local_hostname(base_hostname)
            # ASCEND_ENABLE_USE_FABRIC_MEM: Enable unified memory address direct transmission scheme
            # and only can be used for 800 I/T A3 series.
            # Required supporting hardware versions are as follows:
            if os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") != "1":
                transfer_engine = global_te.get_transfer_engine(local_hostname, device_name=None)
                self.local_seg = local_hostname + ":" + str(transfer_engine.get_rpc_port())
                ret = self.store.setup(
                    self.local_seg,
                    self.config.metadata_server,
                    self.config.global_segment_size,
                    self.config.local_buffer_size,
                    self.config.protocol,
                    self.config.device_name,
                    self.config.master_server_address,
                    transfer_engine.get_engine(),
                )
            else:
                self.local_seg = local_hostname
                ret = self.store.setup(
                    self.local_seg,
                    self.config.metadata_server,
                    self.config.global_segment_size,
                    0,
                    self.config.protocol,
                    self.config.device_name,
                    self.config.master_server_address,
                )

            if ret != 0:
                msg = "Initialize mooncake failed."
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            raise NotImplementedError(f"MooncakeBackend does not support protocol {self.config.protocol!r}.")

        # --- disk-offload wiring -------------------------------------------
        # Preferred-segment override controls which owner segment a PUT
        # lands on. In standalone-store + SSD mode the SSD tier lives on
        # one owner; without pinning, replicas may scatter and miss SSD.
        self.preferred_segment = mooncake_rdma_utils.get_configured_preferred_segment(extra_config)
        self.replicate_config: Any = None
        if ReplicateConfig is not None:
            try:
                self.replicate_config = ReplicateConfig()
                if self.preferred_segment is not None:
                    self.replicate_config.preferred_segment = self.preferred_segment
            except Exception as e:  # noqa: BLE001 — defensive against API drift
                logger.warning(
                    "Mooncake ReplicateConfig setup failed (%s); PUTs will use "
                    "the default replica policy and the SSD tier may be missed.",
                    e,
                )
                self.replicate_config = None
        elif self.preferred_segment is not None:
            logger.warning(
                "Mooncake preferred_segment=%s requested but the installed "
                "mooncake build does not expose ReplicateConfig; PUTs will "
                "use the default replica policy.",
                self.preferred_segment,
            )

        # When ``enable_offload`` is on, the owner allocates a fixed-size
        # DirectIO staging buffer; we must size each GET sub-batch to fit.
        self.disk_offload_buffer_budget_bytes: int | None = (
            DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES if self.config.enable_offload else None
        )
        self.usable_disk_offload_buffer_budget_bytes: int | None = (
            None
            if self.disk_offload_buffer_budget_bytes is None
            else _get_usable_disk_offload_buffer_budget_bytes(self.disk_offload_buffer_budget_bytes)
        )

        logger.info(
            "Mooncake mode=%s (global_segment_size=%d, local_buffer_size=%d, preferred_segment=%s, enable_offload=%s)",
            self.config.mode,
            self.config.global_segment_size,
            self.config.local_buffer_size,
            self.preferred_segment or "<none>",
            self.config.enable_offload,
        )
        if self.config.mode == "embedded":
            if self.config.enable_offload and self.preferred_segment is None:
                logger.warning(
                    "enable_offload is set in embedded mode without "
                    "preferred_segment; SSD tier will only see PUTs that "
                    "happen to land on the owner segment."
                )
            if self.preferred_segment is not None:
                logger.warning(
                    "preferred_segment=%s with mode=embedded: rank-contributed segments will be idle.",
                    self.preferred_segment,
                )
        elif self.config.mode == "standalone-store" and not self.config.enable_offload:
            logger.warning(
                "standalone-store mode without enable_offload: large prefills may exceed the owner DirectIO budget."
            )

    def set_device(self):
        local_rank = get_world_group().local_rank
        device = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(device)

    def register_buffer(self, ptrs: list[int], lengths: list[int]):
        if os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") != "1":
            global_te.register_buffer(ptrs, lengths)

    def exists(self, keys: list[str]) -> list[int]:
        return self.store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        try:
            res = self._batch_put(keys, addrs, sizes)
            for value in res:
                if value < 0:
                    logger.error("Failed to put key %s,res:%s", keys, res)
                    break
        except Exception as e:
            logger.error("Failed to put key %s,error:%s", keys, e)

    def _batch_put(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
    ) -> list[int]:
        """Wrap ``batch_put_from_multi_buffers`` so older mooncake builds
        (no ``replicate_config`` kwarg) still work."""
        if self.replicate_config is not None:
            try:
                return self.store.batch_put_from_multi_buffers(keys, addrs, sizes, self.replicate_config)
            except TypeError:
                # Old mooncake signature without ``replicate_config``.
                logger.debug(
                    "Mooncake batch_put_from_multi_buffers rejected "
                    "ReplicateConfig; falling back to the 3-arg signature."
                )
        return self.store.batch_put_from_multi_buffers(keys, addrs, sizes)

    def get(self, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
        """Batched ``batch_get_into_multi_buffers`` with disk-offload guards.

        When ``enable_offload`` is on, the owner allocates a fixed-size
        DirectIO staging buffer; sending one giant batch through it can
        overflow. We pre-split into sub-batches that fit the soft budget
        and skip any single key whose staging footprint exceeds the
        hard budget. ``VLLM_MOONCAKE_STORE_TIER_LOG`` emits a per-batch
        memory/disk-replica breakdown when set.
        """
        if not keys:
            return

        load_batches: list[tuple[list[str], list[list[int]], list[list[int]]]] = [
            (keys, addrs, sizes)
        ]
        if (
            self.usable_disk_offload_buffer_budget_bytes is not None
            and self.disk_offload_buffer_budget_bytes is not None
        ):
            total_staging_bytes = sum(
                _estimate_disk_offload_staging_bytes(size) for size in sizes
            )
            if total_staging_bytes > self.usable_disk_offload_buffer_budget_bytes:
                load_batches, oversized_key = _split_disk_offload_load_batches(
                    keys,
                    addrs,
                    sizes,
                    self.usable_disk_offload_buffer_budget_bytes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    oversized_key_index = keys.index(oversized_key)
                    oversized_key_bytes = _estimate_disk_offload_staging_bytes(
                        sizes[oversized_key_index]
                    )
                    logger.warning(
                        "Skipping Mooncake load for key %s because it requires "
                        "%d staging bytes, exceeding the owner DirectIO budget "
                        "%d (config: enable_offload=True).",
                        oversized_key,
                        oversized_key_bytes,
                        self.disk_offload_buffer_budget_bytes,
                    )
                    return

        tier_log_enabled = ascend_envs.VLLM_MOONCAKE_STORE_TIER_LOG
        current_batch_keys: list[str] = keys
        try:
            for batch_keys, batch_addrs, batch_sizes in load_batches:
                current_batch_keys = batch_keys
                tiers_by_key: dict[str, str] | None = None
                if tier_log_enabled:
                    tiers_by_key = _get_replica_tiers_by_key(self.store, batch_keys)
                res = self.store.batch_get_into_multi_buffers(
                    batch_keys, batch_addrs, batch_sizes
                )
                if tiers_by_key is not None:
                    _log_mooncake_load_tier_summary(batch_keys, res, tiers_by_key)
                failed = [
                    (key, value)
                    for key, value in zip(batch_keys, res, strict=True)
                    if value < 0
                ]
                if failed:
                    logger.error(
                        "Failed to get %d Mooncake keys from sub-batch "
                        "(batch_keys=%d, first_failures=%s)",
                        len(failed),
                        len(batch_keys),
                        failed[:3],
                    )
                    # Match upstream semantics: stop on first failed sub-batch
                    # so callers don't see partial fills mixed with errors.
                    break
        except Exception as e:  # noqa: BLE001 — backend errors must not crash transfer threads
            logger.error(
                "Failed to get Mooncake sub-batch %s, error: %s",
                current_batch_keys[:3],
                e,
            )


@dataclass
class MooncakeStoreConfig:
    """Configuration for ``MooncakeDistributedStore``.

    ``mode`` selects the topology:

    * ``embedded`` — each rank contributes ``global_segment_size`` in
      process. KV pool is the union of rank-contributed memory.
    * ``standalone-store`` — rank contributes 0; an external
      ``mooncake_client`` process owns the memory pool **and** the SSD
      tier. Required if ``enable_offload`` is on.

    ``enable_offload`` mirrors Mooncake's owner-side SSD tier. When set,
    the worker (this backend) caps GET sub-batches by the owner's
    DirectIO staging buffer (see ``DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES``)
    and prefers an owner segment for PUTs (via ``preferred_segment``).
    """

    metadata_server: str
    protocol: str
    device_name: str
    master_server_address: str
    mode: MooncakeMode = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    enable_offload: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("embedded", "standalone-store"):
            raise ValueError(f"unknown Mooncake mode: {self.mode!r}")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be > 0")
        if self.mode == "embedded" and self.global_segment_size == 0:
            raise ValueError("embedded mode requires global_segment_size > 0")
        if self.mode == "standalone-store" and self.global_segment_size != 0:
            raise ValueError("standalone-store mode requires global_segment_size == 0")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server"),
            master_server_address=config.get("master_server_address"),
            protocol=config.get("protocol", "ascend"),
            device_name=config.get("device_name", ""),
            mode=config.get("mode", "embedded"),
            global_segment_size=_parse_global_segment_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_global_segment_size(config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            enable_offload=bool(config.get("enable_offload", False)),
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
