#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""
Standalone rfork planner mock server used by vLLM rfork seed protocol tests.

Usage:
    python examples/rfork/rfork_planner.py --host 0.0.0.0 --port 1223
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, Request, Response, status


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 1223
    heartbeat_ttl_sec: int = 60
    heartbeat_sweep_sec: int = 5
    default_resource_points: int = 1
    alloc_policy: str = "fifo"

    def __post_init__(self) -> None:
        if self.port <= 0:
            raise ValueError("port must be > 0")
        if self.heartbeat_ttl_sec <= 0:
            raise ValueError("heartbeat_ttl_sec must be > 0")
        if self.heartbeat_sweep_sec <= 0:
            raise ValueError("heartbeat_sweep_sec must be > 0")
        if self.default_resource_points <= 0:
            raise ValueError("default_resource_points must be > 0")
        if self.alloc_policy not in {"fifo", "lru"}:
            raise ValueError("alloc_policy must be one of: fifo, lru")

    @staticmethod
    def from_env() -> Settings:
        return Settings(
            host=os.getenv("RFORK_MOCK_HOST", "0.0.0.0"),
            port=int(os.getenv("RFORK_MOCK_PORT", "1223")),
            heartbeat_ttl_sec=int(os.getenv("RFORK_MOCK_HEARTBEAT_TTL_SEC", "60")),
            heartbeat_sweep_sec=int(os.getenv("RFORK_MOCK_HEARTBEAT_SWEEP_SEC", "5")),
            default_resource_points=int(os.getenv("RFORK_MOCK_DEFAULT_RESOURCE_POINTS", "1")),
            alloc_policy=os.getenv("RFORK_MOCK_ALLOC_POLICY", "fifo").lower(),
        )


@dataclass
class SeedRecord:
    seed_key: str
    seed_ip: str
    seed_port: int
    seed_rank: int
    last_heartbeat_ts: float
    resource_total: int
    resource_used: int = 0

    @property
    def identity(self) -> str:
        return f"{self.seed_ip}:{self.seed_port}:{self.seed_rank}"

    @property
    def available_points(self) -> int:
        return max(self.resource_total - self.resource_used, 0)


@dataclass
class LeaseRecord:
    user_id: str
    seed_key: str
    seed_identity: str
    allocated_points: int
    leased_at: float


class Scheduler:
    def __init__(self, alloc_policy: str = "fifo") -> None:
        if alloc_policy not in {"fifo", "lru"}:
            raise ValueError(f"unsupported alloc policy: {alloc_policy}")
        self.alloc_policy = alloc_policy

    def choose_seed(self, seeds: Iterable[SeedRecord]) -> SeedRecord | None:
        candidates = [seed for seed in seeds if seed.available_points > 0]
        if not candidates:
            return None

        if self.alloc_policy == "fifo":
            return min(candidates, key=lambda s: (s.last_heartbeat_ts, s.identity))

        return max(candidates, key=lambda s: (s.last_heartbeat_ts, s.identity))


class Store:
    def __init__(
        self,
        *,
        heartbeat_ttl_sec: int,
        default_resource_points: int,
        scheduler: Scheduler,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._seeds: dict[str, SeedRecord] = {}
        self._seeds_by_key: dict[str, set[str]] = defaultdict(set)
        self._leases: dict[str, LeaseRecord] = {}
        self._heartbeat_ttl_sec = heartbeat_ttl_sec
        self._default_resource_points = default_resource_points
        self._scheduler = scheduler
        self._time = time_fn or time.time

    @staticmethod
    def _seed_identity(seed_ip: str, seed_port: int, seed_rank: int) -> str:
        return f"{seed_ip}:{seed_port}:{seed_rank}"

    def add_seed(
        self,
        *,
        seed_key: str,
        seed_ip: str,
        seed_port: int,
        seed_rank: int,
        resource_total: int | None = None,
    ) -> SeedRecord:
        identity = self._seed_identity(seed_ip, seed_port, seed_rank)
        now = self._time()
        total = self._default_resource_points if resource_total is None else max(resource_total, 1)

        with self._lock:
            current = self._seeds.get(identity)
            if current is None:
                current = SeedRecord(
                    seed_key=seed_key,
                    seed_ip=seed_ip,
                    seed_port=seed_port,
                    seed_rank=seed_rank,
                    last_heartbeat_ts=now,
                    resource_total=total,
                    resource_used=0,
                )
                self._seeds[identity] = current
                self._seeds_by_key[seed_key].add(identity)
                return current

            if current.seed_key != seed_key:
                self._seeds_by_key[current.seed_key].discard(identity)
                self._seeds_by_key[seed_key].add(identity)
                current.seed_key = seed_key
            current.last_heartbeat_ts = now
            if resource_total is not None:
                current.resource_total = max(resource_total, 1)
                if current.resource_used > current.resource_total:
                    current.resource_used = current.resource_total
            return current

    def get_seed(self, *, seed_key: str) -> tuple[SeedRecord, LeaseRecord] | None:
        with self._lock:
            self.gc_stale_seeds_locked()
            seed_identities = self._seeds_by_key.get(seed_key, set())
            seeds = [self._seeds[sid] for sid in seed_identities if sid in self._seeds]
            selected = self._scheduler.choose_seed(seeds)
            if selected is None:
                return None

            selected.resource_used += 1
            user_id = uuid.uuid4().hex
            lease = LeaseRecord(
                user_id=user_id,
                seed_key=seed_key,
                seed_identity=selected.identity,
                allocated_points=1,
                leased_at=self._time(),
            )
            self._leases[user_id] = lease
            return selected, lease

    def put_seed(self, *, seed_ip: str, seed_port: int, seed_rank: int, user_id: str) -> bool:
        identity = self._seed_identity(seed_ip, seed_port, seed_rank)
        with self._lock:
            lease = self._leases.get(user_id)
            if lease is None:
                return False
            if lease.seed_identity != identity:
                return False

            seed = self._seeds.get(identity)
            if seed is not None:
                seed.resource_used = max(0, seed.resource_used - lease.allocated_points)
            del self._leases[user_id]
            return True

    def gc_stale_seeds(self) -> int:
        with self._lock:
            return self.gc_stale_seeds_locked()

    def gc_stale_seeds_locked(self) -> int:
        now = self._time()
        stale_ids = [
            sid for sid, seed in self._seeds.items() if (now - seed.last_heartbeat_ts) > self._heartbeat_ttl_sec
        ]
        if not stale_ids:
            return 0

        stale_set = set(stale_ids)
        for sid in stale_ids:
            seed = self._seeds.pop(sid)
            self._seeds_by_key[seed.seed_key].discard(sid)
            if not self._seeds_by_key[seed.seed_key]:
                del self._seeds_by_key[seed.seed_key]

        lease_ids = [uid for uid, lease in self._leases.items() if lease.seed_identity in stale_set]
        for uid in lease_ids:
            del self._leases[uid]

        return len(stale_ids)

    def debug_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "seed_count": len(self._seeds),
                "lease_count": len(self._leases),
                "seeds": {
                    sid: {
                        "seed_key": s.seed_key,
                        "resource_total": s.resource_total,
                        "resource_used": s.resource_used,
                        "last_heartbeat_ts": s.last_heartbeat_ts,
                    }
                    for sid, s in self._seeds.items()
                },
                "leases": {
                    uid: {
                        "seed_identity": lease.seed_identity,
                        "seed_key": lease.seed_key,
                        "allocated_points": lease.allocated_points,
                    }
                    for uid, lease in self._leases.items()
                },
            }


class HeartbeatGc:
    def __init__(self, store: Store, sweep_interval_sec: int) -> None:
        self._store = store
        self._sweep_interval_sec = max(sweep_interval_sec, 1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rfork-heartbeat-gc", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._store.gc_stale_seeds()
            time.sleep(self._sweep_interval_sec)


class HeaderError(ValueError):
    pass


@dataclass(frozen=True)
class AddSeedHeaders:
    seed_key: str
    seed_ip: str
    seed_port: int
    seed_rank: int
    seed_refcnt: int


@dataclass(frozen=True)
class GetSeedHeaders:
    seed_key: str


@dataclass(frozen=True)
class PutSeedHeaders:
    seed_ip: str
    seed_port: int
    seed_rank: int
    user_id: str


def _required(headers: Mapping[str, str], key: str) -> str:
    value = headers.get(key)
    if value is None or value == "":
        raise HeaderError(f"missing required header: {key}")
    return value


def _parse_int(value: str, key: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HeaderError(f"invalid integer header {key}: {value}") from exc
    if parsed < minimum:
        raise HeaderError(f"header {key} must be >= {minimum}, got {parsed}")
    return parsed


def parse_add_seed_headers(headers: Mapping[str, str]) -> AddSeedHeaders:
    return AddSeedHeaders(
        seed_key=_required(headers, "SEED_KEY"),
        seed_ip=_required(headers, "SEED_IP"),
        seed_port=_parse_int(_required(headers, "SEED_PORT"), "SEED_PORT", minimum=1),
        seed_rank=_parse_int(_required(headers, "SEED_RANK"), "SEED_RANK", minimum=0),
        seed_refcnt=_parse_int(_required(headers, "SEED_REFCNT"), "SEED_REFCNT", minimum=0),
    )


def parse_get_seed_headers(headers: Mapping[str, str]) -> GetSeedHeaders:
    return GetSeedHeaders(seed_key=_required(headers, "SEED_KEY"))


def parse_put_seed_headers(headers: Mapping[str, str]) -> PutSeedHeaders:
    return PutSeedHeaders(
        seed_ip=_required(headers, "SEED_IP"),
        seed_port=_parse_int(_required(headers, "SEED_PORT"), "SEED_PORT", minimum=1),
        seed_rank=_parse_int(_required(headers, "SEED_RANK"), "SEED_RANK", minimum=0),
        user_id=_required(headers, "USER_ID"),
    )


def build_router(store: Store):
    router = APIRouter()

    @router.post("/add_seed")
    def add_seed(request: Request) -> Response:
        try:
            parsed = parse_add_seed_headers(request.headers)
        except HeaderError as err:
            return Response(content=str(err), status_code=status.HTTP_400_BAD_REQUEST)

        store.add_seed(
            seed_key=parsed.seed_key,
            seed_ip=parsed.seed_ip,
            seed_port=parsed.seed_port,
            seed_rank=parsed.seed_rank,
            # vLLM currently sends SEED_REFCNT=0 as heartbeat metadata.
            # Capacity is controlled by planner config, not by this field.
            resource_total=None,
        )
        return Response(status_code=status.HTTP_200_OK)

    @router.get("/get_seed")
    def get_seed(request: Request) -> Response:
        try:
            parsed = parse_get_seed_headers(request.headers)
        except HeaderError as err:
            return Response(content=str(err), status_code=status.HTTP_400_BAD_REQUEST)

        result = store.get_seed(seed_key=parsed.seed_key)
        if result is None:
            return Response(content="no available seed", status_code=status.HTTP_404_NOT_FOUND)

        seed, lease = result
        response = Response(status_code=status.HTTP_200_OK)
        response.headers["SEED_IP"] = seed.seed_ip
        response.headers["SEED_PORT"] = str(seed.seed_port)
        response.headers["SEED_RANK"] = str(seed.seed_rank)
        response.headers["USER_ID"] = lease.user_id
        return response

    @router.post("/put_seed")
    def put_seed(request: Request) -> Response:
        try:
            parsed = parse_put_seed_headers(request.headers)
        except HeaderError as err:
            return Response(content=str(err), status_code=status.HTTP_400_BAD_REQUEST)

        released = store.put_seed(
            seed_ip=parsed.seed_ip,
            seed_port=parsed.seed_port,
            seed_rank=parsed.seed_rank,
            user_id=parsed.user_id,
        )
        if not released:
            return Response(content="lease not found", status_code=status.HTTP_404_NOT_FOUND)

        return Response(status_code=status.HTTP_200_OK)

    @router.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/debug/snapshot")
    def debug_snapshot() -> dict[str, object]:
        return store.debug_snapshot()

    return router


def create_app(settings: Settings):
    scheduler = Scheduler(settings.alloc_policy)
    store = Store(
        heartbeat_ttl_sec=settings.heartbeat_ttl_sec,
        default_resource_points=settings.default_resource_points,
        scheduler=scheduler,
    )
    gc_runner = HeartbeatGc(store, settings.heartbeat_sweep_sec)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        gc_runner.start()
        try:
            yield
        finally:
            gc_runner.stop()

    app = FastAPI(title="rfork planner mock", version="0.1.0", lifespan=lifespan)
    app.include_router(build_router(store))

    app.state.settings = settings
    app.state.store = store
    app.state.gc_runner = gc_runner
    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    defaults = Settings.from_env()
    parser = argparse.ArgumentParser(description="Standalone rfork planner server")
    parser.add_argument("--host", default=defaults.host, help="bind host (default: env RFORK_MOCK_HOST or 0.0.0.0)")
    parser.add_argument(
        "--port",
        type=int,
        default=defaults.port,
        help="bind port (default: env RFORK_MOCK_PORT or 1223)",
    )
    parser.add_argument(
        "--heartbeat-ttl-sec",
        type=int,
        default=defaults.heartbeat_ttl_sec,
        help="seed heartbeat ttl in seconds (default: env RFORK_MOCK_HEARTBEAT_TTL_SEC or 60)",
    )
    parser.add_argument(
        "--heartbeat-sweep-sec",
        type=int,
        default=defaults.heartbeat_sweep_sec,
        help="gc sweep interval in seconds (default: env RFORK_MOCK_HEARTBEAT_SWEEP_SEC or 5)",
    )
    parser.add_argument(
        "--default-resource-points",
        type=int,
        default=defaults.default_resource_points,
        help="default seed capacity points (default: env RFORK_MOCK_DEFAULT_RESOURCE_POINTS or 1)",
    )
    parser.add_argument(
        "--alloc-policy",
        choices=["fifo", "lru"],
        default=defaults.alloc_policy,
        help="seed allocation policy (default: env RFORK_MOCK_ALLOC_POLICY or fifo)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    settings = Settings(
        host=args.host,
        port=args.port,
        heartbeat_ttl_sec=args.heartbeat_ttl_sec,
        heartbeat_sweep_sec=args.heartbeat_sweep_sec,
        default_resource_points=args.default_resource_points,
        alloc_policy=args.alloc_policy,
    )
    app = create_app(settings)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit("missing dependency: uvicorn. Install it with: python -m pip install uvicorn") from exc
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
