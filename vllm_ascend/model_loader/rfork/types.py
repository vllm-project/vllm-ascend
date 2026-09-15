# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class RForkLifecycleState(Enum):
    INITIALIZED = auto()
    LEASED = auto()
    REGISTERED = auto()
    SERVING = auto()
    FINALIZED = auto()


class LeaseReleaseResult(Enum):
    RELEASED = auto()
    RETRYABLE = auto()
    REJECTED = auto()


class RForkSeedServiceStartResult(Enum):
    STARTED = auto()
    DEFERRED = auto()
    FAILED = auto()

    def __bool__(self) -> bool:
        return self is RForkSeedServiceStartResult.STARTED


@dataclass(frozen=True, slots=True)
class RForkFallbackCleanupResult:
    service_stopped: bool
    lease_released: bool
    memory_reset: bool

    def __bool__(self) -> bool:
        return self.service_stopped and self.lease_released and self.memory_reset

    @property
    def can_schedule_seed(self) -> bool:
        """Whether fallback may start now or defer only on lease release."""
        return self.service_stopped and self.memory_reset


@dataclass(frozen=True, slots=True)
class RForkIdentity:
    tp_rank: int
    global_rank: int
    is_draft_model: bool = False
    pp_rank: int | None = None
    ep_rank: int | None = None
    compatibility_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class SeedLease:
    seed_ip: str
    seed_port: int
    user_id: str
    seed_rank: int
    seed_key: str


@dataclass(frozen=True, slots=True)
class SeedAdvertisement:
    seed_ip: str
    seed_port: int
    seed_rank: int


@dataclass(frozen=True, slots=True)
class SeedTransferInfo:
    session_id: str
    weights: Mapping[str, Any]
    shapes: Mapping[str, Any] | None = None
