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
from __future__ import annotations

from dataclasses import dataclass

import vllm.v1.core.sched.scheduler as _sched_mod  # noqa: F401  (force load)
from vllm.v1.core.sched import output as _output_mod
from vllm.v1.core.sched.output import (
    SchedulerOutput as _UpstreamSchedulerOutput,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

"""Replace ``vllm.v1.core.sched.output.SchedulerOutput`` with a VPP-aware
subclass.

The vLLM upstream ``SchedulerOutput`` (v0.22.1) does **not** carry any VPP
metadata.  vllm-ascend's VPP scheme2 needs every produced
``SchedulerOutput`` to expose ``batch_id`` so that NPU workers can key
their continuation-context dict on it.  Instead of re-typing the full
upstream dataclass (which silently drifts the moment upstream adds,
removes, or reorders a field), this patch subclasses the upstream class
and declares only ``batch_id``.

Because every reference is resolved via Python attribute lookup on the
module (``from vllm.v1.core.sched.output import SchedulerOutput``), both
the engine-core and worker processes see this subclass as long as this
patch is loaded in both.

Field-ordering note: dataclass inheritance appends ``batch_id`` at the
end (after ``new_block_ids_to_zero``).  This is safe because:

  * every call site constructs ``SchedulerOutput(...)`` with keyword
    arguments (verified across vllm upstream and vllm-ascend), so the
    declaration order is irrelevant to construction;
  * dataclass pickle serialises the instance ``__dict__`` and never calls
    ``__init__`` on the receiving side, so order does not affect the
    engine-core -> worker round-trip;
  * no code introspects ``SchedulerOutput`` field order.

``__module__`` is rewritten to the vLLM module path so ``pickle`` locates
the class under ``vllm.v1.core.sched.output`` on the receiving process
even though the file lives in ``vllm_ascend``.
"""


@dataclass
class SchedulerOutput(_UpstreamSchedulerOutput):
    """Upstream ``SchedulerOutput`` + the ``batch_id`` VPP field.

    Inherits the entire upstream field set and ``make_empty()`` so the
    schema tracks upstream automatically; only ``batch_id`` is declared
    here.  ``make_empty()`` is inherited unchanged from upstream and
    leaves ``batch_id`` at its default (``0``).
    """

    # Per-scheduler monotonically increasing id; under VPP scheme2 NPU
    # workers key their continuation-context dict on this value.
    batch_id: int = 0


# ---------------------------------------------------------------------------
# Install the replacement on the vLLM module.
# ---------------------------------------------------------------------------
_output_mod.SchedulerOutput = SchedulerOutput

# ALSO replace the local binding inside ``vllm.v1.core.sched.scheduler``.
# ``scheduler.py`` does ``from .output import SchedulerOutput`` at module
# top, so once that module is loaded its local ``SchedulerOutput`` name is
# frozen to whatever class was current at import time.  If some other
# patch (or ``vllm.v1.outputs`` transitively) loaded the scheduler module
# before this one ran, the local binding is still the vLLM upstream
# class, and ``Scheduler.schedule()`` constructs upstream instances
# instead of ours — causing a class-identity mismatch on pickle
# (``PicklingError: ... it's not the same object as
# vllm.v1.core.sched.output.SchedulerOutput``).
_sched_mod.SchedulerOutput = SchedulerOutput

# Make ``pickle`` locate the class under ``vllm.v1.core.sched.output``
# (matching the symlink above) rather than ``vllm_ascend.patch.platform
# .patch_scheduler_output``.  Both worker and engine-core processes
# execute this module at startup so the symbol is always present in the
# receiving process.
SchedulerOutput.__module__ = "vllm.v1.core.sched.output"


__all__ = ["SchedulerOutput"]

"""Patch ``vllm.v1.core.sched.scheduler.Scheduler`` for VPP.

One responsibility:

1. **VPP per-batch id stamping** (added here): under VPP scheme2 each
   physical Scheduler instance produces ``SchedulerOutput`` objects that
   are passed to NPU workers through pickle, where the workers key their
   continuation context dict on ``scheduler_output.batch_id``.  Upstream
   vLLM wires this in ``Scheduler.schedule`` and ``Scheduler.__init__``;
   we replicate it here so the vllm tree can stay free of VPP residue.

After ``patch_scheduler_output`` runs, the vLLM module's
``SchedulerOutput`` is the VPP-aware class with the ``batch_id`` field.
This patch then wraps the upstream ``Scheduler`` to stamp that field on
every produced ``SchedulerOutput`` instance.

Stamping is done as **plain instance attributes** (not keyword
arguments to ``SchedulerOutput(...)``), because pickle preserves
instance attributes across the worker ↔ engine-core boundary, and
because the upstream ``Scheduler.schedule`` would not pass those kwargs
once we revert vllm to baseline.
"""
# ---------------------------------------------------------------------------
# VPP: per-batch id stamping
# ---------------------------------------------------------------------------
_original_scheduler_init = Scheduler.__init__


def _ascend_vpp_scheduler_init(self, *args, **kwargs) -> None:
    """Wrap ``Scheduler.__init__`` to seed the per-instance batch counter.

    The counter is *not* read by upstream vLLM; vllm-ascend only.  Keeping
    it as a plain instance attribute avoids touching the vllm dataclass
    schema and survives garbage collection naturally with ``self``.
    """
    _original_scheduler_init(self, *args, **kwargs)
    self._vpp_next_batch_id: int = 0


Scheduler.__init__ = _ascend_vpp_scheduler_init


_original_scheduler_schedule = Scheduler.schedule


def _ascend_vpp_scheduler_schedule(self):
    """Wrap ``Scheduler.schedule`` to stamp VPP metadata on the result.

    Stamps the per-batch ``batch_id`` on the returned ``SchedulerOutput``
    so that:

      * pickle round-trips carry it to the worker process;
      * the worker can read ``scheduler_output.batch_id`` without us
        having to thread the value through any other side channel.

    ``batch_id`` is a dataclass field defaulting to ``0``; this wrapper
    overwrites the default with the real per-instance counter on every
    produced output.
    """
    scheduler_output = _original_scheduler_schedule(self)
    if scheduler_output is None:
        return scheduler_output

    # Per-instance counter; never reset across steps so workers see a
    # unique id for every batch even after empty scheduling steps.
    if not hasattr(self, "_vpp_next_batch_id"):
        # Defensive: a Scheduler constructed before the wrapper was
        # installed (e.g. during tests) will not have the counter.
        self._vpp_next_batch_id = 0
    scheduler_output.batch_id = self._vpp_next_batch_id
    self._vpp_next_batch_id += 1

    return scheduler_output


Scheduler.schedule = _ascend_vpp_scheduler_schedule
