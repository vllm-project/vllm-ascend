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

"""A small process-shared, disk-backed cache for multimodal encoder outputs."""

from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


class GlobalMMEmbeddingCache:
    """Store ViT embeddings keyed by the multimodal item hash.

    The cache is intentionally simple: one tensor file per item, LRU eviction
    by mtime, and best-effort behavior on corrupt or concurrently-written
    files. It is meant for repeated-image workloads and for sharing encoder
    results between independent vLLM processes.
    """

    def __init__(self, root: str, max_bytes: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(0, max_bytes)
        self._lock = threading.Lock()
        self._size = self._scan_size()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _scan_size(self) -> int:
        total = 0
        for path in self.root.glob("*.pt"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _path(self, mm_hash: str) -> Path:
        digest = hashlib.sha256(mm_hash.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.pt"

    def get(self, mm_hash: str) -> torch.Tensor | None:
        path = self._path(mm_hash)
        if not path.exists():
            return None
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return None
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return None
        try:
            os.utime(path, None)
        except OSError:
            pass
        return value

    def put(self, mm_hash: str, value: torch.Tensor) -> None:
        if self.max_bytes <= 0 or value.numel() == 0:
            return
        path = self._path(mm_hash)
        if path.exists():
            return
        cpu_value = value.detach().to("cpu")
        self._executor.submit(self._write, path, cpu_value)

    def _write(self, path: Path, cpu_value: torch.Tensor) -> None:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            torch.save(cpu_value, tmp)
        except Exception:
            tmp.unlink(missing_ok=True)
            return
        try:
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            return
        with self._lock:
            try:
                self._size += path.stat().st_size
            except OSError:
                pass
            self._evict_locked()

    def flush(self) -> None:
        """Wait for queued writes to complete (tests and shutdown)."""
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _evict_locked(self) -> None:
        if self.max_bytes <= 0 or self._size <= self.max_bytes:
            return
        entries = []
        for path in self.root.glob("*.pt"):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
        entries.sort()
        for _, size, path in entries:
            if self._size <= self.max_bytes:
                break
            try:
                path.unlink()
            except OSError:
                continue
            self._size -= size
