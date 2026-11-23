# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared similarity metadata server, normally embedded in the DP0 scheduler.

Optional external mode can still run a separately managed server:
python -m vllm_ascend.distributed.ec_transfer.similarity_index_service \
    --endpoint tcp://127.0.0.1:5558 --namespace my-model
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import OrderedDict
from contextlib import suppress
from threading import Event, Lock, Thread
from typing import Any

import msgpack
import zmq

from vllm_ascend.distributed.ec_transfer.cache_statistics import CacheStatistics, CacheStatisticsSnapshot
from vllm_ascend.distributed.ec_transfer.phash_index import PHASH_NUM_BANDS
from vllm_ascend.distributed.ec_transfer.similarity_cache_config import SimilarityCacheConfig
from vllm_ascend.distributed.ec_transfer.similarity_index import (
    LocalSimilarityIndex,
    SimilarityCacheItem,
    SimilarityIndex,
)

_PROTOCOL_VERSION = 2
# Bump when pHash preprocessing, resized-key semantics, or SSIM scoring changes.
_ALGORITHM_VERSION = "phash64-ssim-v1"
_SERVER_POLL_MS = 100
_SERVER_SEND_TIMEOUT_MS = 1000
_SERVER_START_TIMEOUT_SECONDS = 5
_SERVER_STOP_TIMEOUT_SECONDS = 5
_INDEX_CONFIG_FIELDS = (
    "max_hamming",
    "ssim_threshold",
    "ssim_data_range",
    "max_capacity",
    "eviction_threshold",
    "lower_watermark",
)


class SimilarityIndexUnavailable(RuntimeError):
    """Optional shared metadata cannot be reached; fall back to a cache miss."""


class SimilarityIndexConfigurationError(ValueError):
    """Clients disagree on the namespace, protocol, or matching semantics."""


class SimilarityIndexClient:
    """Scheduler-owned client with no cached positive or negative responses.

    Calls belong on the scheduler thread; pHash worker threads return metadata
    for one batched publication. Timed-out sockets are discarded so late replies
    cannot be mistaken for another request. Mutations are idempotent on retry.
    """

    def __init__(
        self,
        config: SimilarityCacheConfig,
        *,
        model_id: str,
        merge_size: int,
        owned_server: EmbeddedSimilarityIndexServer | None = None,
    ) -> None:
        if config.shared_index_endpoint is None:
            raise ValueError("shared_index_endpoint is required for a remote index")
        self.config = config
        self._owned_server = owned_server
        self._settings = {
            "model_id": model_id,
            "merge_size": merge_size,
            "algorithm": _ALGORITHM_VERSION,
            "num_bands": PHASH_NUM_BANDS,
            "config": {key: getattr(config, key) for key in _INDEX_CONFIG_FIELDS},
        }
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._lock = Lock()
        self._retry_after = 0.0
        self._closed = False
        try:
            self._request("ping", [])
        except SimilarityIndexUnavailable:
            logging.getLogger(__name__).warning("Shared similarity index unavailable at startup; using cache misses")
        except Exception:
            self.close()
            raise

    def _request(self, operation: str, args: list[Any]) -> Any:
        with self._lock:
            if self._closed or time.monotonic() < self._retry_after:
                raise SimilarityIndexUnavailable("Shared similarity index is closed or reconnecting")
            request = {
                "version": _PROTOCOL_VERSION,
                "namespace": self.config.shared_index_namespace,
                "settings": self._settings,
                "operation": operation,
                "args": args,
            }
            payload = msgpack.packb(request, use_bin_type=True)
            try:
                if self._socket is None:
                    self._socket = self._context.socket(zmq.REQ)
                    self._socket.setsockopt(zmq.LINGER, 0)
                    self._socket.setsockopt(zmq.IMMEDIATE, 1)
                    self._socket.setsockopt(zmq.SNDTIMEO, self.config.shared_index_timeout_ms)
                    self._socket.setsockopt(zmq.RCVTIMEO, self.config.shared_index_timeout_ms)
                    self._socket.connect(self.config.shared_index_endpoint)
                self._socket.send(payload)
                response = msgpack.unpackb(self._socket.recv(), raw=False)
            except (zmq.ZMQError, ValueError) as exc:
                self._reset_socket()
                self._retry_after = time.monotonic() + self.config.shared_index_timeout_ms / 1000
                logging.getLogger(__name__).warning(
                    "Shared similarity index request failed; retrying after %d ms: %s",
                    self.config.shared_index_timeout_ms,
                    exc,
                )
                raise SimilarityIndexUnavailable("Shared similarity index request failed") from exc
            if response.get("error") is not None:
                raise SimilarityIndexConfigurationError(response["error"])
            return response["result"]

    def batch_get_entries(self, mm_hashes: list[str]) -> list[SimilarityCacheItem | None]:
        if not mm_hashes:
            return []
        entries: list[SimilarityCacheItem | None] = []
        for item in self._request("get", [mm_hashes]):
            if item is None:
                entries.append(None)
            else:
                entry = SimilarityCacheItem(item[0])
                entry.ssim_similar_items = OrderedDict(item[1])
                entry.resized_keys = item[2]
                entries.append(entry)
        return entries

    def batch_register_phashes(self, entries: list[tuple[str, int, str]]) -> None:
        if entries:
            self._request("register", [entries])

    def batch_recall(self, phashes: list[int]) -> list[list[str]]:
        return self._request("recall", [phashes]) if phashes else []

    def record_similarity_scores(self, mm_hash: str, scores: list[tuple[float, str]]) -> None:
        if scores:
            self._request("scores", [mm_hash, scores])

    def report_cache_statistics(self, scheduler_id: str, counts: CacheStatistics) -> CacheStatisticsSnapshot:
        values, reporting_schedulers, epoch = self._request("statistics", [scheduler_id, counts.as_tuple()])
        return CacheStatisticsSnapshot(CacheStatistics(*values), reporting_schedulers, epoch)

    def _reset_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def close(self) -> None:
        try:
            with self._lock:
                if not self._closed:
                    self._closed = True
                    self._reset_socket()
                    self._context.term()
        finally:
            if self._owned_server is not None:
                self._owned_server.close()


class SimilarityIndexServer:
    """One namespace and one atomic index, owned by the service process.

    The first client establishes compatible model and index settings. Every
    subsequent RPC checks them, including after reconnects. The sequential REP
    loop executes only metadata operations, never tensor transfers or SSIM.
    """

    def __init__(self, endpoint: str, namespace: str) -> None:
        SimilarityCacheConfig(shared_index_endpoint=endpoint, shared_index_namespace=namespace)
        self._namespace = namespace
        self._settings: dict[str, Any] | None = None
        self._index: LocalSimilarityIndex | None = None
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, _SERVER_SEND_TIMEOUT_MS)
        self._closed = False
        try:
            self._socket.bind(endpoint)
            self.endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        except Exception:
            self.close()
            raise

    def _handle(self, request: dict[str, Any]) -> Any:
        if request["version"] != _PROTOCOL_VERSION:
            raise SimilarityIndexConfigurationError("Shared index protocol version mismatch")
        if request["namespace"] != self._namespace:
            raise SimilarityIndexConfigurationError("Shared index namespace mismatch")
        settings = request["settings"]
        if settings["algorithm"] != _ALGORITHM_VERSION or settings["num_bands"] != PHASH_NUM_BANDS:
            raise SimilarityIndexConfigurationError("Shared index algorithm compatibility mismatch")
        if self._settings is None:
            if not isinstance(settings["model_id"], str) or not settings["model_id"]:
                raise SimilarityIndexConfigurationError("Shared index requires a model identity")
            if type(settings["merge_size"]) is not int or settings["merge_size"] <= 0:
                raise SimilarityIndexConfigurationError("Shared index requires a positive merge size")
            config = SimilarityCacheConfig(**settings["config"])
            self._index = LocalSimilarityIndex(config)
            self._settings = settings
        elif settings != self._settings:
            raise SimilarityIndexConfigurationError("Shared index model/config compatibility mismatch")
        assert self._index is not None
        args = request["args"]
        operation = request["operation"]
        if operation == "ping":
            return None
        if operation == "get":
            return [
                None if item is None else [item.phash, list(item.ssim_similar_items.items()), item.resized_keys]
                for item in self._index.batch_get_entries(*args)
            ]
        if operation == "register":
            return self._index.batch_register_phashes(*args)
        if operation == "recall":
            return self._index.batch_recall(*args)
        if operation == "statistics":
            scheduler_id, counts = args
            snapshot = self._index.report_cache_statistics(scheduler_id, CacheStatistics(*counts))
            return [snapshot.counts.as_tuple(), snapshot.reporting_schedulers, snapshot.epoch]
        if operation == "scores":
            return self._index.record_similarity_scores(*args)
        raise SimilarityIndexConfigurationError(f"Unsupported shared index operation: {operation}")

    def serve_forever(self, stop_event: Any = None) -> None:
        while stop_event is None or not stop_event.is_set():
            if not self._socket.poll(_SERVER_POLL_MS):
                continue
            try:
                request = msgpack.unpackb(self._socket.recv(), raw=False)
                response = {"result": self._handle(request)}
            except Exception as exc:
                logging.getLogger(__name__).warning("Shared similarity index rejected request: %s", exc)
                response = {"error": str(exc)}
            self._socket.send(msgpack.packb(response, use_bin_type=True))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._socket.close(linger=0)
            self._context.term()

    def __enter__(self) -> SimilarityIndexServer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class EmbeddedSimilarityIndexServer:
    """Own a server thread for the lifetime of the DP0 scheduler connector.

    The thread creates, uses, and closes its ZMQ socket. The scheduler only
    signals readiness/shutdown through events; it never touches that socket.
    No child process, device context, or independent service launcher is needed.
    """

    def __init__(self, endpoint: str, namespace: str) -> None:
        self._stop = Event()
        self._ready = Event()
        self._error: Exception | None = None
        self._thread = Thread(
            target=self._run,
            args=(endpoint, namespace),
            name="ec-similarity-index",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(_SERVER_START_TIMEOUT_SECONDS):
            self.close()
            raise RuntimeError("Timed out starting the embedded similarity index")
        if self._error is not None:
            self.close()
            raise RuntimeError(f"Failed to start embedded similarity index at {endpoint}") from self._error

    def _run(self, endpoint: str, namespace: str) -> None:
        try:
            with SimilarityIndexServer(endpoint, namespace) as server:
                self._ready.set()
                logging.getLogger(__name__).info("Embedded similarity index listening on %s", server.endpoint)
                server.serve_forever(self._stop)
        except Exception as exc:
            self._error = exc
            logging.getLogger(__name__).exception("Embedded similarity index stopped unexpectedly")
        finally:
            self._ready.set()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logging.getLogger(__name__).warning("Embedded similarity index thread is still finishing a request")


def create_similarity_index(
    config: SimilarityCacheConfig,
    parallel_config: Any,
    *,
    model_id: str,
    merge_size: int,
) -> SimilarityIndex:
    """Create a scheduler-local index or connect to the one shared DP index."""
    if config.shared_index_endpoint is None:
        return LocalSimilarityIndex(config)
    owner = None
    if config.shared_index_mode == "embedded":
        # Dense models reset compute DP rank/size. The serving index survives
        # that reset and uniquely identifies the scheduler that owns the server.
        dp_index = getattr(parallel_config, "data_parallel_index", None)
        if type(dp_index) is not int or dp_index < 0:
            raise ValueError("Embedded similarity index requires parallel_config.data_parallel_index")
        if dp_index == 0:
            owner = EmbeddedSimilarityIndexServer(config.shared_index_endpoint, config.shared_index_namespace)
    try:
        return SimilarityIndexClient(config, model_id=model_id, merge_size=merge_size, owned_server=owner)
    except Exception:
        if owner is not None:
            owner.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Bind address, e.g. tcp://127.0.0.1:5558")
    parser.add_argument("--namespace", default="default", help="Deployment namespace, shared by every DP")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with SimilarityIndexServer(args.endpoint, args.namespace) as server:
        logging.getLogger(__name__).info("Shared similarity index listening on %s", server.endpoint)
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
