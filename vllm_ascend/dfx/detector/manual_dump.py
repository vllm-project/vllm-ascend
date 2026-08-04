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

"""Manual one-shot dump trigger via JSON ``dump.dump_once``."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.detector.base import AnomalyDetector
from vllm_ascend.dfx.dfx_types import ILL_TYPE_NONE
from vllm_ascend.dfx.dumper.pending import should_run_anomaly_check_on_rank
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)

# Throttle "dump_once pending" warnings: fire at most once per interval (seconds).
_DUMP_ONCE_WARN_INTERVAL_S = 60.0

# Synthetic req id for dump routing; not a real scheduler request.
MANUAL_DUMP_REQ_ID = "__manual_dump_once__"


class ManualDumpDetector(AnomalyDetector):
    """Watches ``dump.dump_once`` and emits one alert when set to true.

    Requires ``additional_config.dfx_config_reload_interval > 0`` so the JSON
    change is picked up by hot-reload; otherwise ``dump_once`` never arms.

    Must be polled on **every** rank after config sync so ``consume_dump_once``
    clears in-memory state everywhere; only last-PP (pending-OR: TP0) returns an
    alert for the dumper to arm.
    """

    anomaly_type = "manual_dump_once"

    # Throttle "dump_once pending" warnings to once per minute (not per step).
    _last_dump_once_warn_ts: float = 0.0

    def __init__(
        self,
        *,
        dfx_config: DfxRuntimeConfig | None = None,
        runner: Any | None = None,
    ) -> None:
        super().__init__(dfx_config=dfx_config, runner=runner, enabled=True)

    def refresh_from_config(self) -> None:
        # Always enabled when constructed; gate is dump_once + dump.enabled on arm.
        self._enabled = True

    def check_all(self) -> list[AnomalyAlert]:
        """Consume ``dump_once`` if set; return an alert only on arming ranks."""
        if self._dfx_config is None:
            return []
        if not self._dfx_config.dump_once():
            return []
        # Do not consume the JSON flag when dump sink is off — keep true for ops.
        # Warn periodically (not once) so repeated hot-reloads remind operators.
        if not self._dfx_config.dump_enabled():
            now = time.time()
            if now - self._last_dump_once_warn_ts >= _DUMP_ONCE_WARN_INTERVAL_S:
                self._last_dump_once_warn_ts = now
                logger.warning(
                    "[DFX manual_dump] dump.dump_once=true but dump.enabled=false; "
                    "not consuming. Set dump.enabled=true to trigger (detector optional)."
                )
            return []
        if not self._dfx_config.consume_dump_once():
            return []
        if not should_run_anomaly_check_on_rank(self._runner):
            return []
        logger.info(
            "[DFX manual_dump] dump_once consumed → alert anomaly_type=%s (report will snapshot all batch requests)",
            self.anomaly_type,
        )
        return [
            AnomalyAlert(
                anomaly_type=self.anomaly_type,
                req_id=MANUAL_DUMP_REQ_ID,
                is_ill=True,
                ill_type=ILL_TYPE_NONE,
                detail={"source": "dump.dump_once"},
                skip_related_check=True,
                consume_quota=False,
            )
        ]
