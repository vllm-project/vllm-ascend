#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Per-step wave index for incident correlation."""

from __future__ import annotations

import threading
from collections import deque


class WaveTracker:
    """Process-local wave counter + per-req sample stamps.

    ``record`` runs on the inference thread; async ``take`` / ``pending`` may
    run on the output copy thread — all stamp mutations share ``_lock``.

    Stamps are a **FIFO per req** (W1-3): under async scheduling, step N+1 may
    ``record`` before step N's ``get_output`` ``take``. A single overwrite slot
    made the later take miss (``missing sample-wave stamp``) and polluted
    ``arm_wave``. Ordered ``deque`` FIFOs keep stamp/take paired.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wave = 0
        self._sample_waves: dict[str, deque[int]] = {}

    def advance(self, *, allow_arm: bool = True) -> None:
        if not allow_arm:
            return
        with self._lock:
            self._wave += 1

    def current_wave(self) -> int:
        with self._lock:
            return self._wave

    def record_sample_waves(self, req_ids: list[str] | None) -> None:
        if not req_ids:
            return
        with self._lock:
            wave = self._wave
            for req_id in req_ids:
                if not req_id:
                    continue
                rid = str(req_id)
                q = self._sample_waves.get(rid)
                if q is None:
                    q = deque()
                    self._sample_waves[rid] = q
                q.append(wave)

    def take_sample_wave(self, req_id: str) -> int | None:
        with self._lock:
            rid = str(req_id)
            q = self._sample_waves.get(rid)
            if not q:
                return None
            wave = q.popleft()
            if not q:
                self._sample_waves.pop(rid, None)
            return wave

    def pending(self, req_id: str) -> bool:
        """True when a stamp is still unconsumed (async output not yet materialized)."""
        with self._lock:
            q = self._sample_waves.get(str(req_id))
            return bool(q)

    def discard(self, req_id: str) -> None:
        with self._lock:
            self._sample_waves.pop(str(req_id), None)

    def discard_many(self, req_ids: list[str] | None) -> None:
        # B2: reap-time cleanup — finished requests must not accumulate stamps.
        if not req_ids:
            return
        with self._lock:
            for req_id in req_ids:
                if req_id:
                    self._sample_waves.pop(str(req_id), None)
