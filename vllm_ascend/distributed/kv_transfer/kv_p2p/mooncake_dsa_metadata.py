# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for one-shot blockwise DSA Mooncake receives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

MAX_TCP_PORT = 65535


class DsaLocalResultKind(str, Enum):
    RECEIVE_COMPLETE = "receive_complete"
    TRANSFER_FAILED = "transfer_failed"


class DsaTransferPhase(str, Enum):
    INDEXER_D2D = "indexer_d2d"
    MAIN_D2RH = "main_d2rh"


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _normalize_block_ids(name: str, values: object) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    block_ids = tuple(values)
    for block_id in block_ids:
        _require_nonnegative_integer(name, block_id)
    if not block_ids:
        raise ValueError(f"{name} must not be empty")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError(f"{name} must not contain duplicates")
    return block_ids


@dataclass(frozen=True, slots=True)
class RemoteEndpoint:
    remote_host: str
    remote_port: int
    remote_engine_id: str

    def __post_init__(self) -> None:
        _require_nonempty_string("remote_host", self.remote_host)
        _require_nonnegative_integer("remote_port", self.remote_port)
        if self.remote_port == 0 or self.remote_port > MAX_TCP_PORT:
            raise ValueError("remote_port must be in range 1..65535")
        _require_nonempty_string("remote_engine_id", self.remote_engine_id)


@dataclass(frozen=True, slots=True)
class RemoteSource:
    remote_request_id: str
    endpoints_by_prefill_rank: tuple[RemoteEndpoint, ...]
    indexer_block_ids: tuple[int, ...]
    main_block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(
            "remote_request_id",
            self.remote_request_id,
        )
        endpoints = tuple(self.endpoints_by_prefill_rank)
        if not endpoints:
            raise ValueError("endpoints_by_prefill_rank must not be empty")
        if not all(
            isinstance(endpoint, RemoteEndpoint) for endpoint in endpoints
        ):
            raise TypeError(
                "endpoints_by_prefill_rank must contain RemoteEndpoint values"
            )
        object.__setattr__(self, "endpoints_by_prefill_rank", endpoints)
        object.__setattr__(
            self,
            "indexer_block_ids",
            _normalize_block_ids(
                "indexer_block_ids",
                self.indexer_block_ids,
            ),
        )
        object.__setattr__(
            self,
            "main_block_ids",
            _normalize_block_ids("main_block_ids", self.main_block_ids),
        )


@dataclass(frozen=True, slots=True)
class DsaStepRequest:
    """One PD receive into vLLM-owned Decode blocks."""

    request_id: str
    source: RemoteSource
    main_host_block_ids: tuple[int, ...]
    indexer_hbm_block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        if not isinstance(self.source, RemoteSource):
            raise TypeError("source must be RemoteSource")
        object.__setattr__(
            self,
            "main_host_block_ids",
            _normalize_block_ids(
                "main_host_block_ids",
                self.main_host_block_ids,
            ),
        )
        object.__setattr__(
            self,
            "indexer_hbm_block_ids",
            _normalize_block_ids(
                "indexer_hbm_block_ids",
                self.indexer_hbm_block_ids,
            ),
        )


@dataclass(frozen=True, slots=True)
class DsaConnectorMetadata(KVConnectorMetadata):
    """One-shot receive requests issued to Decode workers."""

    requests: tuple[DsaStepRequest, ...] = ()

    def __post_init__(self) -> None:
        requests_by_id: dict[str, DsaStepRequest] = {}
        for request in self.requests:
            if not isinstance(request, DsaStepRequest):
                raise TypeError(
                    "requests must contain DsaStepRequest values"
                )
            existing = requests_by_id.get(request.request_id)
            if existing is not None and existing != request:
                raise ValueError(
                    f"conflicting DSA requests for {request.request_id!r}"
                )
            requests_by_id[request.request_id] = request
        object.__setattr__(
            self,
            "requests",
            tuple(
                requests_by_id[key] for key in sorted(requests_by_id)
            ),
        )


@dataclass(frozen=True, slots=True)
class DsaLocalResult:
    request_id: str
    tp_rank: int
    kind: DsaLocalResultKind
    failure_phase: DsaTransferPhase | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        _require_nonnegative_integer("tp_rank", self.tp_rank)
        if not isinstance(self.kind, DsaLocalResultKind):
            raise TypeError("kind must be DsaLocalResultKind")
        if self.failure_phase is not None and not isinstance(
            self.failure_phase,
            DsaTransferPhase,
        ):
            raise TypeError("failure_phase must be DsaTransferPhase")
        if (
            self.kind is DsaLocalResultKind.TRANSFER_FAILED
            and self.failure_phase is None
        ):
            raise ValueError("TRANSFER_FAILED requires a failure phase")
        if (
            self.kind is DsaLocalResultKind.RECEIVE_COMPLETE
            and self.failure_phase is not None
        ):
            raise ValueError(
                "RECEIVE_COMPLETE must not carry a failure phase"
            )

    @property
    def identity(self) -> tuple[str, int]:
        return self.request_id, self.tp_rank


def _merge_results(
    groups: Iterable[Iterable[DsaLocalResult]],
) -> tuple[DsaLocalResult, ...]:
    merged: dict[tuple[str, int], DsaLocalResult] = {}
    for results in groups:
        for result in results:
            if not isinstance(result, DsaLocalResult):
                raise TypeError(
                    "results must contain DsaLocalResult values"
                )
            existing = merged.get(result.identity)
            if existing is not None and existing != result:
                raise ValueError(
                    f"conflicting DSA local results for {result.identity}"
                )
            merged[result.identity] = result
    return tuple(merged[key] for key in sorted(merged))


@dataclass(frozen=True, slots=True)
class DsaWorkerResultMetadata(KVConnectorWorkerMetadata):
    """Rank-aware completion results for one-shot PD receives."""

    results: tuple[DsaLocalResult, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "results",
            _merge_results((self.results,)),
        )

    def aggregate(
        self,
        other: KVConnectorWorkerMetadata,
    ) -> DsaWorkerResultMetadata:
        if not isinstance(other, DsaWorkerResultMetadata):
            raise TypeError(
                "aggregate expects DsaWorkerResultMetadata"
            )
        return DsaWorkerResultMetadata(
            _merge_results((self.results, other.results))
        )
