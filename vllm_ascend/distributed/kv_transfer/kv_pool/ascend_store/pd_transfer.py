# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Immutable prompt snapshots handed from a pool producer to a decoder."""

import hashlib
import json
from copy import copy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import regex as re

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig
    from vllm.v1.core.sched.output import SchedulerOutput


def requires_layerwise_eager_load(scheduler_output: "SchedulerOutput", kv_config: "KVTransferConfig | None") -> bool:
    """A graph replay cannot execute Python per-layer KV transfer hooks.

    Keep the restore step eager; subsequent decode steps can replay the graph
    once all KV is resident in the decoder's cache.
    """
    return (
        kv_config is not None
        and kv_config.kv_connector_extra_config.get("use_layerwise") is True
        and getattr(scheduler_output, "has_sync_kv_loads", False) is True
    )


def prepare_pool_pd_forward(
    scheduler_output: "SchedulerOutput", kv_config: "KVTransferConfig | None"
) -> "SchedulerOutput":
    """Prepare layerwise PUTs before forward, including P misses and one-token D.

    Recent vLLM runners defer start_load_kv until after forward when there are
    no synchronous external hits. Pool PD uses that hook to prepare both PUT
    and GET tasks. A worker-local copy also resets tasks before decode steps.
    Older runners already start the connector before forward.
    """
    if (
        kv_config is not None
        and kv_config.kv_connector_extra_config.get("pool_pd") is True
        and hasattr(scheduler_output, "has_sync_kv_loads")
    ):
        scheduler_output = copy(scheduler_output)
        scheduler_output.has_sync_kv_loads = True
    return scheduler_output


def prompt_digest(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PoolPDTransfer:
    transfer_id: str
    num_tokens: int
    block_size: int
    layout_id: str
    prompt_digest: str
    block_hashes: tuple[str, ...]
    version: int = 1

    @property
    def tail_tokens(self) -> int:
        return self.num_tokens % self.block_size

    @property
    def tail_id(self) -> str | None:
        if not self.tail_tokens:
            return None
        return f"pd-tail-v1:{self.transfer_id}:{self.num_tokens}"

    @property
    def object_ids(self) -> tuple[str, ...]:
        return self.block_hashes + ((self.tail_id,) if self.tail_id else ())

    def to_wire(self) -> dict[str, Any]:
        result = asdict(self)
        result["block_hashes"] = list(self.block_hashes)
        result["tail_id"] = self.tail_id
        result["tail_tokens"] = self.tail_tokens
        return result

    @classmethod
    def create(cls, token_ids: list[int], block_size: int, layout_id: str, hashes: list[str]) -> "PoolPDTransfer":
        if not token_ids or block_size <= 0 or len(hashes) != len(token_ids) // block_size:
            raise ValueError("Pool PD requires a nonempty prompt and every complete prompt block hash")
        return cls(uuid4().hex, len(token_ids), block_size, layout_id, prompt_digest(token_ids), tuple(hashes))

    @classmethod
    def from_wire(cls, data: dict[str, Any], token_ids: list[int], block_size: int, layout_id: str) -> "PoolPDTransfer":
        if not isinstance(data, dict) or data.get("status") != "ready":
            raise ValueError("Pool PD transfer is not ready")
        if type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("Unsupported pool PD transfer version")
        if type(data.get("num_tokens")) is not int or data["num_tokens"] != len(token_ids) or not token_ids:
            raise ValueError("Pool PD prompt length mismatch")
        if type(data.get("block_size")) is not int or data["block_size"] != block_size:
            raise ValueError("Pool PD block size mismatch")
        if data.get("layout_id") != layout_id or data.get("prompt_digest") != prompt_digest(token_ids):
            raise ValueError("Pool PD prompt or KV layout mismatch")
        transfer_id = data.get("transfer_id")
        if not isinstance(transfer_id, str) or re.fullmatch(r"[0-9a-f]{32}", transfer_id) is None:
            raise ValueError("Invalid pool PD transfer id")
        hashes = data.get("block_hashes")
        if (
            not isinstance(hashes, list)
            or len(hashes) != len(token_ids) // block_size
            or any(not isinstance(h, str) or re.fullmatch(r"[0-9a-f]+", h) is None for h in hashes)
        ):
            raise ValueError("Invalid pool PD block hashes")
        return cls(transfer_id, len(token_ids), block_size, layout_id, data["prompt_digest"], tuple(hashes))
