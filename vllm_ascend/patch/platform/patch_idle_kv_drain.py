# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Drain connector finished-transfer bookkeeping on idle engine steps.

**Sunset note**: this is a transitional fix for vllm-ascend #20755, whose
worker-output ACK routing declares that it covers V2 idle steps. Upstream
vLLM #66295 took the same approach (synthesize an empty
``KVConnectorOutput`` so the scheduler update path runs) but was withdrawn
unmerged, pointing at #20755 as the follow-up. Neither is in the pinned
vLLM release, and the deadlock this patch fixes reproduces on the merged
tree. Once #20755 lands and covers idle steps, this patch must be removed
to avoid double-draining the same completions.

In P/D disaggregation the producer releases the KV blocks of a
already-completed transfer only via
``scheduler._update_from_kv_xfer_finished``, which core calls from
``update_from_output`` under ``if kv_connector_output:``. For
``MooncakeConnectorV2`` the producer-side worker never produces a non-empty
``KVConnectorOutput``: ``get_transfer_results`` only returns receive
results and the receiving thread exists on the decode role alone, while
no code path records transfer stats and the remaining fields stay at their
base-class defaults. ``is_empty()`` is therefore always true, the empty
output is compacted away, and the gate stays closed - not only on idle
steps but under normal traffic too. Once the KV cache fills up, no request
can be scheduled any more, so nothing advances and every request deadlocks
behind ``capacity`` until the process is restarted.

Upstream already knows about this window (the post-step hook yields the GIL
for "delayed KV connector frees") but only sleeps. This patch drains the
connector state on idle steps instead, either way the bookkeeping kept by
the connector can reach ``_free_blocks`` regardless of scheduler activity.

The drain only frees requests upstream would free anyway: ``_free_blocks``
requires a finished request, and an id reaches ``finished_sending`` only
after ``request_finished`` reported ``delay_free``, i.e. the blocks are
held by the connector and not by an in-flight batch. The same upstream
function runs with the same precondition, so this is not a new double-free
path.

The call is gated on ``scheduler.has_requests()``, so a fully idle engine
performs zero drains, and a failure is logged and retried on the next step.
Per-step cost on idle steps was measured in-engine on 4xAscend910 at
mean 12.61us / p99 25.98us through the real connector update path (see the
PR description), i.e. below 0.01% of a step.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorOutput
from vllm.logger import logger
from vllm.v1.engine.core import EngineCoreProc

_OriginalProcessEngineStep = EngineCoreProc._process_engine_step


def _process_engine_step_with_idle_kv_drain(self) -> bool:
    model_executed = _OriginalProcessEngineStep(self)
    if not model_executed and self.scheduler.has_requests():
        connector = getattr(self.scheduler, "connector", None)
        if connector is not None:
            try:
                self.scheduler._update_from_kv_xfer_finished(KVConnectorOutput())
            except Exception:
                # A failed drain only postpones this round's release; the next
                # idle step retries it and the engine keeps running.
                logger.exception("Idle KV transfer drain failed; retrying next step")
    return model_executed


def _apply_patch() -> None:
    if getattr(EngineCoreProc._process_engine_step, "_vllm_ascend_idle_kv_drain", False):
        return
    _process_engine_step_with_idle_kv_drain._vllm_ascend_idle_kv_drain = True  # type: ignore[attr-defined]
    EngineCoreProc._process_engine_step = _process_engine_step_with_idle_kv_drain


_apply_patch()
