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

"""Manual trigger control-plane events (outside anomaly detector pipeline)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm_ascend.dfx.dumper.pending import should_run_anomaly_check_on_rank
from vllm_ascend.logger import init_logger_ascend

if TYPE_CHECKING:
    from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

logger = init_logger_ascend(__name__)

_MANUAL_TRIGGER_WARN_INTERVAL_S = 60.0
MANUAL_TRIGGER_REQ_ID = "__manual_trigger__"
MANUAL_TRIGGER_TYPE = "manual_trigger"


@dataclass(slots=True)
class TriggerEvent:
    """One control-plane trigger consumed from DFX runtime config."""

    trigger_type: str
    req_id: str
    detail: dict[str, Any] = field(default_factory=dict)
    consume_quota: bool = False

    def to_report_detail(self) -> dict[str, Any]:
        out = dict(self.detail)
        out.setdefault("source", self.trigger_type)
        return out


class ManualTriggerManager:
    """Consumes manual trigger flags from config and emits trigger events."""

    _last_manual_trigger_warn_ts: float = 0.0

    def __init__(self, *, dfx_config: DfxRuntimeConfig, runner: Any) -> None:
        self._dfx_config = dfx_config
        self._runner = runner

    def consume_once(self, *, allow_arm: bool) -> TriggerEvent | None:
        if not self._dfx_config.manual_trigger():
            return None
        if not allow_arm:
            logger.debug(
                "[DFX manual_trigger] dump.manual_trigger deferred (allow_arm=False); await execute_model wave"
            )
            return None
        if not self._dfx_config.dump_enabled():
            now = time.time()
            if now - self._last_manual_trigger_warn_ts >= _MANUAL_TRIGGER_WARN_INTERVAL_S:
                self._last_manual_trigger_warn_ts = now
                logger.warning(
                    "[DFX manual_trigger] dump.manual_trigger=true but dump.enabled=false; "
                    "not consuming. Set dump.enabled=true to trigger."
                )
            return None
        if not self._dfx_config.consume_manual_trigger():
            return None
        if not should_run_anomaly_check_on_rank(self._runner):
            return None
        logger.info("[DFX manual_trigger] dump.manual_trigger consumed and armed")
        return TriggerEvent(
            trigger_type=MANUAL_TRIGGER_TYPE,
            req_id=MANUAL_TRIGGER_REQ_ID,
            detail={"source": "dump.manual_trigger"},
            consume_quota=False,
        )
