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

"""Replace ``vllm.v1.core.sched.output.SchedulerOutput`` with a VPP-aware
subclass.

Because every reference is resolved via Python attribute lookup on the
module (``from vllm.v1.core.sched.output import SchedulerOutput``), both
the engine-core and worker processes see this subclass as long as this
patch is loaded in both.

Field-ordering note: dataclass inheritance appends ``batch_id`` at the
end (after ``new_block_ids_to_zero``).  This is safe because:
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
_sched_mod.SchedulerOutput = SchedulerOutput

# Make ``pickle`` locate the class under ``vllm.v1.core.sched.output``
# (matching the symlink above) rather than ``vllm_ascend.patch.platform
# .patch_scheduler_output``.  Both worker and engine-core processes
# execute this module at startup so the symbol is always present in the
# receiving process.
SchedulerOutput.__module__ = "vllm.v1.core.sched.output"


__all__ = ["SchedulerOutput"]

"""Patch ``vllm.v1.core.sched.scheduler.Scheduler`` for VPP.
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
