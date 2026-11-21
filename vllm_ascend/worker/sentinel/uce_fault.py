#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""HBM UCE exceptions and address-to-KV-cache classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import logger

if TYPE_CHECKING:
    from vllm_ascend.worker.sentinel.kv_cache_region_registry import (
        KvCacheRegion,
        KvCacheRegionRegistry,
    )

_UCE_NO_ADDRESS = 0
_UCE_NON_PTA_MEMORY = 1


class HbmUceFault(RuntimeError):
    """Base HBM uncorrectable-error fault raised after classification."""

    def __init__(
        self,
        region: str,
        address: int | None,
        reason: str = "",
        hit: KvCacheRegion | None = None,
    ) -> None:
        self.region = region
        self.address = address
        self.reason = reason
        self.hit = hit
        addr_s = hex(address) if address is not None else "unknown"
        details = [f"HBM UCE region={region}", f"addr={addr_s}"]
        if reason:
            details.append(reason)
        if self.hit is not None:
            details.append(f"kv={self.hit.layer_name}/{self.hit.kind} rank={self.hit.rank} device={self.hit.device_id}")
        super().__init__(" ".join(details))


class HbmUceKvCacheFault(HbmUceFault):
    """UCE landed in this worker's KV-cache HBM range; retry is allowed."""

    def __init__(self, address: int | None, hit: KvCacheRegion, reason: str = "") -> None:
        super().__init__(region="kv_cache", address=address, reason=reason, hit=hit)


class HbmUceUnrecoverableFault(HbmUceFault):
    """UCE missed KV cache, or the address could not be classified."""


class HbmUceNonKvFault(HbmUceUnrecoverableFault):
    """Known UCE outside owned KV; repair the device but never resume inference."""


def classify_uce_after_stop(registry: KvCacheRegionRegistry | None, device_id: int | None = None) -> HbmUceFault | None:
    """Query UCE after device stop, mark fully KV-covered tensors safe, and return the fault classification."""
    check_result = _check_uce_in_memory(device_id)
    if check_result == _UCE_NO_ADDRESS:
        logger.debug("No torch_npu UCE address after stop_device; continue retry as a normal device reset")
        return None
    if check_result == _UCE_NON_PTA_MEMORY:
        return HbmUceUnrecoverableFault(
            region="unknown",
            address=None,
            reason="uce address belongs to non-PTA memory",
        )

    addr_entries = _get_uce_addr()
    address = _first_uce_address(addr_entries)
    if registry is None or not registry.regions:
        logger.error("HBM UCE detected but KV cache registry is empty; treating as unrecoverable")
        return HbmUceUnrecoverableFault(
            region="unknown",
            address=address,
            reason="kv cache registry is empty",
        )
    if not addr_entries:
        logger.error("HBM UCE detected but no device address could be parsed; treating as unrecoverable")
        return HbmUceUnrecoverableFault(
            region="unknown",
            address=None,
            reason="uce address missing from exception",
        )

    try:
        if any(int(entry["ptr"]) < 0 or int(entry["size"]) <= 0 for entry in addr_entries):
            raise ValueError("Invalid UCE range")
    except (KeyError, TypeError, ValueError, OverflowError):
        return HbmUceUnrecoverableFault(region="unknown", address=address, reason="invalid UCE address ranges")

    hit = registry.recover_uce_addresses(addr_entries)
    if hit is not None:
        logger.warning(
            "HBM UCE addr=%s hit KV cache layer=%s kind=%s rank=%s",
            hex(address) if address is not None else "unknown",
            hit.layer_name,
            hit.kind,
            hit.rank,
        )
        return HbmUceKvCacheFault(address=address, hit=hit, reason="address in kv cache region")

    logger.error(
        "HBM UCE addr=%s missed KV cache regions; treating as unrecoverable",
        hex(address) if address is not None else "unknown",
    )
    return HbmUceNonKvFault(
        region="non_kv_cache",
        address=address,
        reason="address outside kv cache regions",
    )


def _check_uce_in_memory(device_id: int | None) -> int:
    try:
        import torch

        check_result = int(torch.npu.check_uce_in_memory(device_id))
        logger.debug("HBM UCE memory check: device=%s code=%d", device_id, check_result)
        return check_result
    except Exception as exc:
        logger.warning("Failed to check torch_npu UCE memory status: %s", exc)
        raise HbmUceUnrecoverableFault(
            region="unknown", address=None, reason="unable to query UCE memory status"
        ) from exc


def _get_uce_addr() -> list[dict]:
    try:
        import torch_npu

        addr_entries = torch_npu.npu._get_uce_addr()
    except Exception as exc:
        logger.warning("Failed to get torch_npu UCE addresses: %s", exc)
        return []
    if not isinstance(addr_entries, list):
        logger.warning("Unexpected torch_npu UCE address payload: %s", addr_entries)
        return []
    return addr_entries


def _first_uce_address(addr_entries: list[dict]) -> int | None:
    for entry in addr_entries:
        try:
            return int(entry["ptr"])
        except (KeyError, TypeError, ValueError):
            continue
    return None
