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

"""KV block write integrity detector (wave monotonicity / same-wave writer)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.config_backed import ConfigBackedDetector
from vllm_ascend.dfx.kv_block_meta import KvBlockMetaTracker


class BlockKvDetector(ConfigBackedDetector):
    """Detect inconsistent KV block write metadata (reorder / writer clash)."""

    anomaly_type = "block_kv"
    section_key = "block_kv"

    def __init__(self, *, dfx_config: Any | None = None, runner: Any | None = None) -> None:
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=False)
        self._check_wave_regression = True
        self._check_same_wave_writer = True
        if dfx_config is not None:
            self.refresh_from_config()

    def _apply_detector_values(self, getter: Callable[[str, Any], Any]) -> None:
        self._check_wave_regression = bool(getter("check_wave_regression", True))
        self._check_same_wave_writer = bool(getter("check_same_wave_writer", True))

    def check_writes(
        self,
        req_id: str,
        block_ids: list[int],
        wave: int,
    ) -> list[AnomalyAlert]:
        """Return alerts for violations seen *before* ``record_writes`` applies."""
        if not self._precheck() or not req_id or not block_ids:
            return []
        if not self._passes_input_filter(req_id, log=False):
            return []
        tracker = KvBlockMetaTracker.get()
        violations = tracker.preview_write_checks(
            req_id,
            block_ids,
            int(wave),
            check_wave_regression=self._check_wave_regression,
            check_same_wave_writer=self._check_same_wave_writer,
        )
        if not violations:
            return []
        # One report per request write batch (multi-block → single alert).
        return [
            AnomalyAlert(
                anomaly_type=self.anomaly_type,
                req_id=str(req_id),
                detail={
                    "wave": int(wave),
                    "num_violations": len(violations),
                    "violations": [
                        {
                            "violation": v.violation,
                            "block_id": v.block_id,
                            "prev_wave": v.prev_wave,
                            "new_wave": v.new_wave,
                            "prev_writer_req_id": v.prev_writer,
                            "new_writer_req_id": v.new_writer,
                        }
                        for v in violations
                    ],
                },
            )
        ]
