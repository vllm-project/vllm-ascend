# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import math
import threading
import time
from dataclasses import replace

import requests
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.model_loader.rfork.config import RForkConfig, RForkConfigurationError
from vllm_ascend.model_loader.rfork.identity import build_seed_key
from vllm_ascend.model_loader.rfork.types import (
    LeaseReleaseResult,
    RForkIdentity,
    SeedAdvertisement,
    SeedLease,
    SeedRemovalResult,
    SeedReportResult,
    SeedReportStatus,
)

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
            raise RForkConfigurationError("request_timeout_sec must be a finite positive number")
        if not math.isfinite(float(request_timeout_sec)) or float(request_timeout_sec) <= 0:
            raise RForkConfigurationError("request_timeout_sec must be a finite positive number")

        self.planner_url = config.planner_url
        self.tp_rank = identity.tp_rank
        self.request_timeout_sec = float(request_timeout_sec)
        self.config = config
        self.last_advertisement: SeedAdvertisement | None = None
        self._advertisement_lock = threading.Lock()
        # None requires evidence from this endpoint before a 404 can acknowledge withdrawal.
        self._seed_removal_supported: bool | None = getattr(config, "seed_removal_supported", False)
        self._rejected_seed_lease: SeedLease | None = None
        compatibility_fingerprint = identity.compatibility_fingerprint
        if compatibility_fingerprint is None:
            raise RuntimeError(
                "RFork requires a compatibility fingerprint for the seed key; "
                "build one with build_compatibility_fingerprint()."
            )
        self._compatibility_fingerprint = compatibility_fingerprint
        self._is_draft_model = identity.is_draft_model
        self._pp_rank = identity.pp_rank
        self._ep_rank = identity.ep_rank
        self._sharded_dp_rank = getattr(identity, "sharded_dp_rank", None)
        self._structural_digest: str | None = None
        self._seed_key: str | None = None

    def bind_structural_digest(self, digest: str) -> None:
        if self._structural_digest is not None and self._structural_digest != digest:
            raise RuntimeError("RFork structural digest changed after binding")
        self._structural_digest = digest

    @property
    def structural_digest(self) -> str | None:
        return self._structural_digest

    def verify_structural_digest(self, digest: str) -> bool:
        """Bind on first use; log when the live structure drifted from registration."""
        if self._structural_digest is None:
            self.bind_structural_digest(digest)
            return True
        if self._structural_digest != digest:
            logger.error(
                "RFork structural digest changed between registration and seed advertisement: "
                "bound=%s current=%s; retaining the registered digest.",
                self._structural_digest,
                digest,
            )
            return False
        return True

    @property
    def seed_key(self) -> str:
        if self._seed_key is None:
            if self._structural_digest is None:
                raise RuntimeError("RFork seed key requires a bound structural digest")
            self._seed_key = build_seed_key(
                tp_rank=self.tp_rank,
                model_url=self.config.model_url,
                model_deploy_strategy_name=self.config.model_deploy_strategy_name,
                compatibility_fingerprint=self._compatibility_fingerprint,
                structural_digest=self._structural_digest,
                is_draft_model=self._is_draft_model,
                pp_rank=self._pp_rank,
                ep_rank=self._ep_rank,
                sharded_dp_rank=self._sharded_dp_rank,
            )
        return self._seed_key

    def _require_planner(self) -> None:
        if not self.planner_url:
            raise RuntimeError(
                "rfork_scheduler_url is not set. Configure it through model_loader_extra_config or RFORK_SCHEDULER_URL."
            )

    def _with_model_identity_headers(self, headers: dict[str, str]) -> dict[str, str]:
        if not self.config.send_model_identity_headers:
            return headers
        return {
            **headers,
            "MODEL_URL": self.config.model_url,
            "DEPLOY_STRATEGY": self.config.model_deploy_strategy_name,
            "IS_DRAFT": str(self._is_draft_model).lower(),
        }

    def take_rejected_lease(self) -> SeedLease | None:
        """Hand a rejected allocation to the session for release retries, never for transfer."""
        lease = self._rejected_seed_lease
        self._rejected_seed_lease = None
        return lease

    def acquire_seed(self) -> SeedLease | None:
        if self._rejected_seed_lease is not None:
            result = self.release_seed_once(self._rejected_seed_lease)
            if result is not LeaseReleaseResult.RELEASED:
                return None
            self._rejected_seed_lease = None
        lease_for_release: SeedLease | None = None
        try:
            self._require_planner()
            response = requests.get(
                f"{self.planner_url}/get_seed",
                headers=self._with_model_identity_headers({"SEED_KEY": self.seed_key}),
                timeout=self.request_timeout_sec,
                allow_redirects=False,
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
            lease_ttl_sec = response.headers.get("LEASE_TTL_SEC")
            if not seed_ip or not seed_port or not user_id or seed_rank is None:
                raise RuntimeError("planner returned incomplete seed lease headers")
            try:
                parsed_port = int(seed_port)
                parsed_rank = int(seed_rank)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"planner returned malformed seed lease values: seed_port={seed_port!r}, seed_rank={seed_rank!r}"
                ) from exc
            if parsed_port <= 0 or parsed_port > 65535 or parsed_rank < 0:
                raise ValueError(
                    f"planner returned invalid seed lease values: seed_port={parsed_port}, seed_rank={parsed_rank}"
                )
            # Keep a valid allocation identity for cleanup even if its optional TTL is malformed.
            lease_for_release = SeedLease(
                seed_ip=seed_ip,
                seed_port=parsed_port,
                user_id=user_id,
                seed_rank=parsed_rank,
                seed_key=self.seed_key,
            )
            parsed_lease_ttl_sec = None
            if lease_ttl_sec is not None:
                try:
                    parsed_lease_ttl_sec = float(lease_ttl_sec)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"planner returned malformed LEASE_TTL_SEC={lease_ttl_sec!r}") from exc
                if not math.isfinite(parsed_lease_ttl_sec) or parsed_lease_ttl_sec <= 0:
                    raise ValueError(f"planner returned invalid LEASE_TTL_SEC={lease_ttl_sec!r}")
            return replace(lease_for_release, lease_ttl_sec=parsed_lease_ttl_sec)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            logger.warning("RFork planner seed acquisition failed: %s", exc)
            if lease_for_release is not None:
                self._rejected_seed_lease = lease_for_release
                result = self.release_seed_once(lease_for_release)
                if result is LeaseReleaseResult.RELEASED:
                    self._rejected_seed_lease = None
                else:
                    logger.warning(
                        "RFork rejected a seed lease but could not return its planner allocation: lease=%s result=%s",
                        lease_log_id(lease_for_release),
                        result.name,
                    )
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
                headers=self._with_model_identity_headers(headers),
                timeout=self.request_timeout_sec,
                allow_redirects=False,
            )
            if response.status_code in (200, 404):
                log_release = logger.debug if response.status_code == 200 else logger.info
                log_release(
                    "RFork lease release acknowledged: lease=%s status=%s "
                    "(404 means absent, not verified timely release)",
                    lease_log_id(lease),
                    response.status_code,
                )
                return LeaseReleaseResult.RELEASED
            body = response.text.replace(lease.user_id, "<lease-id>") if lease.user_id else response.text
            body = "".join(char if char.isprintable() else " " for char in body[:RESPONSE_LOG_MAX_CHARS])
            retryable = response.status_code in (408, 429) or 500 <= response.status_code < 600
            log_failure = logger.debug if retryable else logger.warning
            log_failure(
                "RFork planner lease release rejected: lease=%s status=%s response=%r",
                lease_log_id(lease),
                response.status_code,
                body,
            )
            if retryable:
                return LeaseReleaseResult.RETRYABLE
            return LeaseReleaseResult.REJECTED
        except requests.RequestException as exc:
            logger.debug(
                "RFork lease release request failed: lease=%s error=%s", lease_log_id(lease), type(exc).__name__
            )
            return LeaseReleaseResult.RETRYABLE

    def renew_seed_once(self, lease: SeedLease) -> bool:
        if lease.lease_ttl_sec is None:
            return True
        try:
            self._require_planner()
            response = requests.post(
                f"{self.planner_url}/renew_seed_lease",
                headers=self._with_model_identity_headers(
                    {
                        "SEED_IP": lease.seed_ip,
                        "SEED_PORT": str(lease.seed_port),
                        "USER_ID": lease.user_id,
                        "SEED_RANK": str(lease.seed_rank),
                    }
                ),
                timeout=self.request_timeout_sec,
                allow_redirects=False,
            )
            if response.status_code == 200:
                return True
            logger.warning(
                "RFork planner lease renewal rejected: lease=%s status=%s",
                lease_log_id(lease),
                response.status_code,
            )
        except (requests.RequestException, RuntimeError) as exc:
            logger.warning(
                "RFork planner lease renewal failed: lease=%s error=%s",
                lease_log_id(lease),
                type(exc).__name__,
            )
        return False

    def remove_seed(self, advertisement: SeedAdvertisement | None = None) -> SeedRemovalResult:
        try:
            self._require_planner()
        except RuntimeError as exc:
            logger.warning("RFork planner seed removal setup failed: %s", exc)
            return SeedRemovalResult.REJECTED

        # Protect only the snapshot; session shutdown already joins heartbeat I/O.
        with self._advertisement_lock:
            target = advertisement or self.last_advertisement
            supported = self._seed_removal_supported
        if target is None:
            return SeedRemovalResult.ABSENT
        if supported is False:
            return SeedRemovalResult.UNSUPPORTED
        headers = {
            "SEED_KEY": self.seed_key,
            "SEED_IP": target.seed_ip,
            "SEED_PORT": str(target.seed_port),
            "SEED_RANK": str(target.seed_rank),
        }
        result = SeedRemovalResult.RETRYABLE
        for attempt in range(SEED_REMOVAL_MAX_ATTEMPTS):
            try:
                response = requests.post(
                    f"{self.planner_url}/remove_seed",
                    headers=self._with_model_identity_headers(headers),
                    timeout=self.request_timeout_sec,
                    allow_redirects=False,
                )
                status = response.status_code
                if status == 200 or (status == 404 and supported is True):
                    with self._advertisement_lock:
                        self._seed_removal_supported = True
                        if target is self.last_advertisement:
                            self.last_advertisement = None
                    return SeedRemovalResult.REMOVED if status == 200 else SeedRemovalResult.ABSENT
                if status == 404:
                    logger.info("RFork planner seed removal returned an unconfirmed 404; retaining the advertisement.")
                    return SeedRemovalResult.UNCONFIRMED
                if status in (405, 501):
                    with self._advertisement_lock:
                        self._seed_removal_supported = False
                    return SeedRemovalResult.UNSUPPORTED
                if status == 409:
                    with self._advertisement_lock:
                        self._seed_removal_supported = True
                    supported = True
                    result = SeedRemovalResult.DRAINING
                elif status in (408, 429) or 500 <= status < 600:
                    result = SeedRemovalResult.RETRYABLE
                else:
                    result = SeedRemovalResult.REJECTED
                logger.warning(
                    "RFork planner seed removal attempt %d/%d returned status=%s",
                    attempt + 1,
                    SEED_REMOVAL_MAX_ATTEMPTS,
                    status,
                )
                if result is SeedRemovalResult.REJECTED:
                    return result
            except requests.RequestException as exc:
                result = SeedRemovalResult.RETRYABLE
                logger.warning(
                    "RFork planner seed removal attempt %d/%d failed: %s",
                    attempt + 1,
                    SEED_REMOVAL_MAX_ATTEMPTS,
                    exc,
                )
            if attempt + 1 < SEED_REMOVAL_MAX_ATTEMPTS:
                time.sleep(SEED_REMOVAL_RETRY_BACKOFF_SEC * (attempt + 1))
        return result

    def report_seed_once(self, port: int, seed_ip: str | None = None) -> SeedReportResult:
        try:
            self._require_planner()
            advertisement = SeedAdvertisement(seed_ip or get_ip(), port, self.tp_rank)
            # Reserve identity before I/O so cleanup can remove a seed even when the response is lost.
            with self._advertisement_lock:
                self.last_advertisement = advertisement
            response = requests.post(
                f"{self.planner_url}/add_seed",
                headers=self._with_model_identity_headers(
                    {
                        "SEED_KEY": self.seed_key,
                        "SEED_IP": advertisement.seed_ip,
                        "SEED_PORT": str(advertisement.seed_port),
                        "SEED_RANK": str(advertisement.seed_rank),
                        "SEED_REFCNT": "0",
                    }
                ),
                timeout=self.request_timeout_sec,
                allow_redirects=False,
            )
            if response.status_code == 200:
                return SeedReportResult(SeedReportStatus.ACCEPTED)
            if response.status_code in (408, 429) or 500 <= response.status_code < 600:
                return SeedReportResult(SeedReportStatus.RETRYABLE, f"status={response.status_code}")
            return SeedReportResult(SeedReportStatus.REJECTED, f"status={response.status_code}")
        except requests.RequestException as exc:
            return SeedReportResult(SeedReportStatus.RETRYABLE, type(exc).__name__)
        except (RuntimeError, ValueError, OSError) as exc:
            return SeedReportResult(SeedReportStatus.REJECTED, type(exc).__name__)
