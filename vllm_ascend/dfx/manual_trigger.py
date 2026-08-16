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


def iter_local_request_rows(
    runner: Any,
    scheduler_output: Any | None = None,
) -> list[tuple[str, int]]:
    """``(req_id, req_idx)`` for local live requests (v1 batch / v2 req_states).

    MRV1 uses ``input_batch.req_ids`` / ``requests``. MRV2 keeps ``input_batch``
    as ``None`` until after ``prepare_inputs``; before that (and across waves)
    fall back to ``execute_model_state.input_batch``, ``req_states``, and
    ``scheduler_output.num_scheduled_tokens`` so manual_trigger can arm on the
    first real prefill wave.
    """
    input_batch = getattr(runner, "input_batch", None)
    req_ids = getattr(input_batch, "req_ids", None) if input_batch is not None else None
    if req_ids:
        rows = [(str(req_id), idx) for idx, req_id in enumerate(req_ids) if req_id]
        if rows:
            return rows

    state = getattr(runner, "execute_model_state", None)
    state_batch = getattr(state, "input_batch", None) if state is not None else None
    state_ids = getattr(state_batch, "req_ids", None) if state_batch is not None else None
    if state_ids:
        rows = [(str(req_id), idx) for idx, req_id in enumerate(state_ids) if req_id]
        if rows:
            return rows

    requests = getattr(runner, "requests", None)
    if isinstance(requests, dict) and requests:
        return [(str(req_id), -1) for req_id in requests if req_id]

    req_states = getattr(runner, "req_states", None)
    id_map = getattr(req_states, "req_id_to_index", None) if req_states is not None else None
    if isinstance(id_map, dict) and id_map:
        return sorted(
            ((str(rid), int(idx)) for rid, idx in id_map.items() if rid),
            key=lambda item: item[1],
        )

    if scheduler_output is not None:
        num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if isinstance(num_scheduled, dict) and num_scheduled:
            return [(str(req_id), -1) for req_id, n_tok in num_scheduled.items() if req_id and int(n_tok or 0) > 0]
    return []


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
    """Consumes manual trigger counts from config and emits trigger events."""

    _last_manual_trigger_warn_ts: float = 0.0

    def __init__(self, *, dfx_config: DfxRuntimeConfig, runner: Any) -> None:
        self._dfx_config = dfx_config
        self._runner = runner

    @staticmethod
    def _local_batch_nonempty(
        runner: Any,
        scheduler_output: Any | None = None,
    ) -> bool:
        """True when this wave has at least one local request to snapshot."""
        return bool(iter_local_request_rows(runner, scheduler_output))

    def consume_once(
        self,
        *,
        allow_arm: bool,
        scheduler_output: Any | None = None,
    ) -> TriggerEvent | None:
        remaining = self._dfx_config.manual_trigger_count()
        if remaining <= 0:
            return None
        if not allow_arm:
            logger.debug(
                "[DFX manual_trigger] dump.manual_trigger deferred (allow_arm=False); "
                "await execute_model wave (remaining=%d)",
                remaining,
            )
            return None
        if not self._dfx_config.dump_enabled():
            now = time.time()
            if now - self._last_manual_trigger_warn_ts >= _MANUAL_TRIGGER_WARN_INTERVAL_S:
                self._last_manual_trigger_warn_ts = now
                logger.warning(
                    "[DFX manual_trigger] dump.manual_trigger=%s but dump.enabled=false; "
                    "not consuming. Set dump.enabled=true to trigger.",
                    remaining,
                )
            return None
        # Keep count until a wave with requests so report/detail is useful and
        # empty idle cleanup steps do not burn remaining dumps.
        if not self._local_batch_nonempty(self._runner, scheduler_output):
            logger.debug(
                "[DFX manual_trigger] dump.manual_trigger deferred (empty batch); remaining=%d",
                remaining,
            )
            return None
        if not self._dfx_config.consume_manual_trigger():
            return None
        left = self._dfx_config.manual_trigger_count()
        if not should_run_anomaly_check_on_rank(self._runner):
            return None
        logger.info(
            "[DFX manual_trigger] dump.manual_trigger armed (remaining_after=%d)",
            left,
        )
        return TriggerEvent(
            trigger_type=MANUAL_TRIGGER_TYPE,
            req_id=MANUAL_TRIGGER_REQ_ID,
            detail={
                "source": "dump.manual_trigger",
                "manual_trigger_remaining_after": left,
            },
            consume_quota=False,
        )
