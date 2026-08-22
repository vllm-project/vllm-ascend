from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from collections.abc import Callable, Iterable
from typing import Any


class PlannerError(RuntimeError):
    """Raised when the CPU planning process cannot return a plan."""


def _hash_to_string(value: bytes | str) -> str:
    return value.hex() if isinstance(value, bytes) else value


def _planner_main(request_queue: Any, response_queue: Any, ready_queue: Any) -> None:
    try:
        ready_queue.put((True, ""))
        while True:
            request = request_queue.get()
            if request is None:
                return
            command_id = request["command_id"]
            try:
                hashes = request["block_hashes"]
                block_size = int(request["block_size"])
                hash_block_size = int(request["hash_block_size"])
                token_len = int(request["token_len"])
                mask_num = int(request.get("mask_num", 0))
                block_ids = request.get("block_ids")
                allowed = request.get("allowed_chunk_indices")
                allowed_set = set(allowed) if allowed is not None else None
                factor = max(block_size // max(hash_block_size, 1), 1)
                grouped_hashes = [hashes[(idx + 1) * factor - 1] for idx in range(len(hashes) // factor)]
                num_chunks = min(len(grouped_hashes), (token_len + block_size - 1) // block_size)
                block_id_offset = max(num_chunks - len(block_ids), 0) if block_ids is not None else 0
                candidate_index = 0
                entries = []
                for chunk_index in range(num_chunks):
                    start = chunk_index * block_size
                    end = min(start + block_size, token_len)
                    if start < mask_num or (allowed_set is not None and chunk_index not in allowed_set):
                        continue
                    block_id = None
                    if block_ids is not None:
                        block_index = chunk_index - block_id_offset
                        if block_index < 0 or block_index >= len(block_ids):
                            continue
                        block_id = block_ids[block_index]
                        if request.get("skip_null_blocks", False) and block_id <= 0:
                            continue
                    shard_size = request.get("shard_size")
                    shard_rank = request.get("shard_rank")
                    shard_allows = (
                        shard_rank is None
                        or shard_size is None
                        or int(shard_size) <= 1
                        or candidate_index % int(shard_size) == int(shard_rank)
                    )
                    candidate_index += 1
                    if not shard_allows:
                        continue
                    entries.append(
                        (
                            start,
                            end,
                            request["prefix"] + _hash_to_string(grouped_hashes[chunk_index]),
                            grouped_hashes[chunk_index],
                            block_id,
                        )
                    )
                response_queue.put((command_id, True, entries, ""))
            except Exception as exc:
                response_queue.put((command_id, False, [], f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        ready_queue.put((False, f"{type(exc).__name__}: {exc}"))


class TransferPlanner:
    """Spawned pure-CPU planner used by the hybrid transfer backend."""

    def __init__(self, timeout_s: float = 30.0, start_method: str = "spawn") -> None:
        self.timeout_s = max(float(timeout_s), 0.001)
        self._context = mp.get_context(start_method)
        self._requests = self._context.Queue()
        self._responses = self._context.Queue()
        self._ready = self._context.Queue()
        self._process: mp.Process | None = None
        self._command_id = 0
        self._lock = threading.Lock()
        self._disabled = False
        self._shutdown = False

    def start(self) -> None:
        if self._shutdown:
            raise PlannerError("planner process has been shut down")
        if self._disabled:
            raise PlannerError("planner process is disabled after an initialization failure")
        if self._process is not None and self._process.is_alive():
            return
        if self._process is not None:
            self._process.join(timeout=0)
            self._process = None
        self._drain_queue(self._requests)
        self._drain_queue(self._responses)
        self._drain_queue(self._ready)
        self._process = self._context.Process(
            target=_planner_main,
            args=(self._requests, self._responses, self._ready),
            name="AscendStorePlanner",
            daemon=True,
        )
        try:
            self._process.start()
        except PermissionError as exc:
            self._process = None
            self._disabled = True
            raise PlannerError("planner process is not permitted in this environment") from exc
        except Exception:
            self._process = None
            raise
        try:
            ok, message = self._ready.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            process = self._process
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            self._process = None
            raise PlannerError("planner process did not become ready") from exc
        if not ok:
            process = self._process
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            self._process = None
            raise PlannerError(message)

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def enabled(self) -> bool:
        return not self._disabled and not self._shutdown

    @staticmethod
    def _drain_queue(channel: Any) -> None:
        while True:
            try:
                channel.get_nowait()
            except queue.Empty:
                return

    def disable(self) -> None:
        self.close()
        self._disabled = True

    def plan(
        self,
        *,
        prefix: str,
        token_len: int,
        block_hashes: Iterable[bytes | str],
        block_ids: list[int],
        block_size: int,
        hash_block_size: int,
        mask_num: int = 0,
        skip_null_blocks: bool = False,
        allowed_chunk_indices: tuple[int, ...] | None = None,
        shard_rank: int | None = None,
        shard_size: int | None = None,
    ) -> list[tuple[int, int, str, bytes | str, int | None]]:
        if self._shutdown:
            raise PlannerError("planner process has been shut down")
        with self._lock:
            self.start()
            if not self.is_alive:
                raise PlannerError("planner process is not alive")
            self._command_id += 1
            command_id = self._command_id
            self._requests.put(
                {
                    "command_id": command_id,
                    "prefix": prefix,
                    "token_len": token_len,
                    "block_hashes": list(block_hashes),
                    "block_ids": list(block_ids),
                    "block_size": block_size,
                    "hash_block_size": hash_block_size,
                    "mask_num": mask_num,
                    "skip_null_blocks": skip_null_blocks,
                    "allowed_chunk_indices": allowed_chunk_indices,
                    "shard_rank": shard_rank,
                    "shard_size": shard_size,
                }
            )
            try:
                response_id, ok, entries, message = self._responses.get(timeout=self.timeout_s)
            except queue.Empty as exc:
                raise PlannerError("planner process timed out") from exc
            if response_id != command_id:
                raise PlannerError(f"planner response mismatch: expected {command_id}, got {response_id}")
            if not ok:
                raise PlannerError(message)
            return entries

    def close(self) -> None:
        """Stop the child while keeping queues reusable for a restart."""
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.is_alive():
                self._requests.put(None)
                process.join(timeout=min(self.timeout_s, 2.0))
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            else:
                process.join(timeout=0)
            self._process = None

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.close()
        self._requests.close()
        self._responses.close()
        self._ready.close()


def iter_token_key_strings_with_block_ids(
    token_database: Any,
    planner: TransferPlanner | None,
    token_len: int,
    block_hashes: Iterable[bytes | str],
    block_ids: list[int],
    mask_num: int = 0,
    kv_cache_group_id: int = 0,
    skip_null_blocks: bool = False,
    chunk_filter: Callable[[int], bool] | None = None,
    shard_rank: int | None = None,
    shard_size: int | None = None,
) -> Iterable[tuple[int, int, str, bytes | str, int | None]]:
    if planner is None or not planner.enabled:
        yield from token_database.process_token_key_strings_with_block_ids(
            token_len,
            block_hashes,
            block_ids,
            mask_num,
            kv_cache_group_id,
            skip_null_blocks,
            chunk_filter,
            shard_rank,
            shard_size,
        )
        return

    block_hashes = list(block_hashes)
    allowed_chunk_indices = None
    if chunk_filter is not None:
        allowed_chunk_indices = tuple(
            start // token_database.get_block_size(kv_cache_group_id)
            for start, _, _, _ in token_database._iter_token_chunks(
                token_len,
                block_hashes,
                mask_num,
                kv_cache_group_id,
                block_ids=block_ids,
                skip_null_blocks=skip_null_blocks,
                chunk_filter=chunk_filter,
            )
        )
    try:
        yield from planner.plan(
            prefix=token_database._get_key_prefix(kv_cache_group_id),
            token_len=token_len,
            block_hashes=block_hashes,
            block_ids=block_ids,
            block_size=token_database.get_block_size(kv_cache_group_id),
            hash_block_size=token_database.hash_block_size,
            mask_num=mask_num,
            skip_null_blocks=skip_null_blocks,
            allowed_chunk_indices=allowed_chunk_indices,
            shard_rank=shard_rank,
            shard_size=shard_size,
        )
    except Exception:
        planner.disable()
        yield from token_database.process_token_key_strings_with_block_ids(
            token_len,
            block_hashes,
            block_ids,
            mask_num,
            kv_cache_group_id,
            skip_null_blocks,
            chunk_filter,
            shard_rank,
            shard_size,
        )
