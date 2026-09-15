# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import math
import threading
import time

import requests
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.identity import build_seed_key
from vllm_ascend.model_loader.rfork.types import LeaseReleaseResult, RForkIdentity, SeedAdvertisement, SeedLease

HEARTBEAT_LOG_EVERY_N = 4
SEED_REMOVAL_MAX_ATTEMPTS = 3
SEED_REMOVAL_RETRY_BACKOFF_SEC = 0.1
RESPONSE_LOG_MAX_CHARS = 256


def lease_log_id(lease: SeedLease) -> str:
    """Correlate attempts without logging the lease credential itself."""
    return hashlib.sha256(lease.user_id.encode()).hexdigest()[:12]


class RForkPlannerClient:
    def __init__(
        self,
        config: RForkConfig,
        identity: RForkIdentity,
    ) -> None:
        request_timeout_sec = config.request_timeout_sec
        if isinstance(request_timeout_sec, bool) or not isinstance(request_timeout_sec, (int, float)):
            raise ValueError("request_timeout_sec must be a finite positive number")
        if not math.isfinite(float(request_timeout_sec)) or float(request_timeout_sec) <= 0:
            raise ValueError("request_timeout_sec must be a finite positive number")

        self.planner_url = config.planner_url
        self.tp_rank = identity.tp_rank
        self.request_timeout_sec = float(request_timeout_sec)
        self.config = config
        self.last_advertisement: SeedAdvertisement | None = None
        compatibility_fingerprint = identity.compatibility_fingerprint
        if compatibility_fingerprint is None:
            raise RuntimeError(
                "RFork requires a compatibility fingerprint for the seed key; "
                "build one with build_compatibility_fingerprint()."
            )
        self.seed_key = build_seed_key(
            tp_rank=identity.tp_rank,
            model_url=config.model_url,
            model_deploy_strategy_name=config.model_deploy_strategy_name,
            compatibility_fingerprint=compatibility_fingerprint,
            is_draft_model=identity.is_draft_model,
            pp_rank=identity.pp_rank,
            ep_rank=identity.ep_rank,
        )

    def _require_planner(self) -> None:
        if not self.planner_url:
            raise RuntimeError(
                "rfork_scheduler_url is not set. Configure it through model_loader_extra_config or RFORK_SCHEDULER_URL."
            )

    def acquire_seed(self) -> SeedLease | None:
        try:
            self._require_planner()
            response = requests.get(
                f"{self.planner_url}/get_seed",
                headers={"SEED_KEY": self.seed_key},
                timeout=self.request_timeout_sec,
            )
            if response.status_code == 404:
                logger.debug("RFork planner has no available seed for seed_key=%s", self.seed_key)
                return None
            if response.status_code != 200:
                raise RuntimeError(f"planner get_seed returned status={response.status_code}")
            seed_ip = response.headers.get("SEED_IP")
            seed_port = response.headers.get("SEED_PORT")
            user_id = response.headers.get("USER_ID")
            seed_rank = response.headers.get("SEED_RANK")
            if not seed_ip or not seed_port or not user_id or seed_rank is None:
                raise RuntimeError("planner returned incomplete seed lease headers")
            parsed_port = int(seed_port)
            parsed_rank = int(seed_rank)
            if parsed_port <= 0 or parsed_rank < 0:
                raise ValueError
            return SeedLease(
                seed_ip=seed_ip,
                seed_port=parsed_port,
                user_id=user_id,
                seed_rank=parsed_rank,
                seed_key=self.seed_key,
            )
        except Exception as exc:
            logger.warning("RFork planner seed acquisition failed: %s", exc)
            return None

    def release_seed_once(self, lease: SeedLease) -> LeaseReleaseResult:
        """Send one bounded-timeout request; the session owns background retries."""
        try:
            self._require_planner()
        except RuntimeError as exc:
            logger.warning("RFork planner lease release setup failed: %s", exc)
            return LeaseReleaseResult.REJECTED

        headers = {
            "SEED_IP": lease.seed_ip,
            "SEED_PORT": str(lease.seed_port),
            "USER_ID": lease.user_id,
            "SEED_RANK": str(lease.seed_rank),
        }
        try:
            response = requests.post(
                f"{self.planner_url}/put_seed",
                headers=headers,
                timeout=self.request_timeout_sec,
                allow_redirects=False,
            )
            if response.status_code in (200, 404):
                logger.info(
                    "RFork lease release acknowledged: lease=%s status=%s "
                    "(404 means absent, not verified timely release)",
                    lease_log_id(lease),
                    response.status_code,
                )
                return LeaseReleaseResult.RELEASED
            body = response.text.replace(lease.user_id, "<lease-id>") if lease.user_id else response.text
            body = "".join(char if char.isprintable() else " " for char in body[:RESPONSE_LOG_MAX_CHARS])
            logger.warning(
                "RFork planner lease release rejected: lease=%s status=%s response=%r",
                lease_log_id(lease),
                response.status_code,
                body,
            )
            if response.status_code in (408, 429) or 500 <= response.status_code < 600:
                return LeaseReleaseResult.RETRYABLE
            return LeaseReleaseResult.REJECTED
        except requests.RequestException as exc:
            logger.warning(
                "RFork lease release request failed: lease=%s error=%s", lease_log_id(lease), type(exc).__name__
            )
            return LeaseReleaseResult.RETRYABLE

    def release_seed(self, lease: SeedLease) -> bool:
        """Synchronous bounded retry helper; startup uses release_seed_once in a worker."""
        for attempt in range(self.config.lease_release_max_attempts):
            result = self.release_seed_once(lease)
            if result is LeaseReleaseResult.RELEASED:
                return True
            if result is LeaseReleaseResult.REJECTED:
                return False
            if attempt + 1 < self.config.lease_release_max_attempts:
                time.sleep(self.config.lease_release_retry_interval_sec)
        return False

    def remove_seed(self, advertisement: SeedAdvertisement | None = None) -> bool:
        try:
            self._require_planner()
        except Exception as exc:
            logger.warning("RFork planner seed removal setup failed: %s", exc)
            return False

        target = advertisement or self.last_advertisement
        if target is None:
            return True
        headers = {
            "SEED_KEY": self.seed_key,
            "SEED_IP": target.seed_ip,
            "SEED_PORT": str(target.seed_port),
            "SEED_RANK": str(target.seed_rank),
        }
        for attempt in range(SEED_REMOVAL_MAX_ATTEMPTS):
            try:
                response = requests.post(
                    f"{self.planner_url}/remove_seed",
                    headers=headers,
                    timeout=self.request_timeout_sec,
                )
                if response.status_code in (200, 404):
                    if target == self.last_advertisement:
                        self.last_advertisement = None
                    return True
                logger.warning(
                    "RFork planner seed removal attempt %d/%d returned status=%s",
                    attempt + 1,
                    SEED_REMOVAL_MAX_ATTEMPTS,
                    response.status_code,
                )
            except Exception as exc:
                logger.warning(
                    "RFork planner seed removal attempt %d/%d failed: %s",
                    attempt + 1,
                    SEED_REMOVAL_MAX_ATTEMPTS,
                    exc,
                )
            if attempt + 1 < SEED_REMOVAL_MAX_ATTEMPTS:
                time.sleep(SEED_REMOVAL_RETRY_BACKOFF_SEC * (attempt + 1))
        return False

    def report_seed_once(self, port: int, seed_ip: str | None = None) -> bool:
        try:
            self._require_planner()
            advertisement = SeedAdvertisement(seed_ip or get_ip(), port, self.tp_rank)
            response = requests.post(
                f"{self.planner_url}/add_seed",
                headers={
                    "SEED_KEY": self.seed_key,
                    "SEED_IP": advertisement.seed_ip,
                    "SEED_PORT": str(advertisement.seed_port),
                    "SEED_RANK": str(advertisement.seed_rank),
                    "SEED_REFCNT": "0",
                },
                timeout=self.request_timeout_sec,
            )
            if response.status_code != 200:
                logger.warning("RFork planner seed report returned status=%s", response.status_code)
                return False
            self.last_advertisement = advertisement
            return True
        except Exception as exc:
            logger.warning("RFork planner seed report failed: %s", exc)
            return False

    def run_seed_heartbeat(
        self,
        port: int,
        sleep_interval: float | None = None,
        stop_event: threading.Event | None = None,
        seed_ip: str | None = None,
        initial_delay: bool = False,
    ) -> None:
        if sleep_interval is None:
            sleep_interval = self.config.heartbeat_interval_sec
        if (
            isinstance(sleep_interval, bool)
            or not isinstance(sleep_interval, (int, float))
            or not math.isfinite(sleep_interval)
            or sleep_interval <= 0
        ):
            raise ValueError("heartbeat sleep_interval must be a finite positive number")
        if initial_delay and stop_event is not None and stop_event.wait(sleep_interval):
            return
        heartbeat_index = 0
        while stop_event is None or not stop_event.is_set():
            heartbeat_index += 1
            reported = self.report_seed_once(port, seed_ip=seed_ip)
            if not reported:
                logger.warning("RFork heartbeat failed for seed_key=%s", self.seed_key)
            elif heartbeat_index % HEARTBEAT_LOG_EVERY_N == 0:
                logger.debug("RFork heartbeat accepted for seed_key=%s", self.seed_key)
            if stop_event is None:
                time.sleep(sleep_interval)
            else:
                stop_event.wait(sleep_interval)
