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

"""Carry a resumed request's replay start to the V1 model runner (vLLM #56227).

SWA bounded replay keeps the sliding-window KV out of the prefix cache and
rebuilds it after a prefix hit by recomputing the hit's last window. Upstream
ships the whole mechanism, and the part that tells the worker where the replayed
run begins travels in the scheduling payload: ``NewRequestData.replay_start``,
filled from ``Request.replay_start``.

The V2 model runner folds resumed requests into that same new-request list, so
one field covers it. The V1 runner resumes a preempted request through
``CachedRequestData`` instead, and a resumed request is precisely the one that
can be rewound a second time -- preemption clears the computed count, so
readmission runs the prefix lookup again and re-sets the replay start. Without
the field travelling with the payload, the worker would not know to stop writing
the replayed positions, and would overwrite blocks that are still shared with
every other request that hit the same prefix.

So this patch adds the one field upstream does not have, and fills it where
upstream builds that payload. Wrapping ``Scheduler._make_cached_request_data``
rather than subclassing covers every Ascend scheduler at once, including the
ones whose ``schedule()`` is a pinned copy of an older upstream body -- those
call this same method.

It reads ``Request.replay_start``, which upstream assigns at admission. A vLLM
pin bump has to re-check that this still holds: a resumed request whose fresh
admission did not rewind again has to end up with a zero replay start, or the
worker would pad slots that were never replayed.

Nothing here is Ascend-specific beyond the V1 payload it targets, so the patch
is inert on a pin that predates #56227 -- see ``_upstream_carries_replay_start``.
"""

import dataclasses
import functools
import inspect

from vllm.logger import logger
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.sched.scheduler import Scheduler

_EXPECTED_PARAMETERS = (
    "self",
    "running_reqs",
    "resumed_reqs",
    "num_scheduled_tokens",
    "spec_decode_tokens",
    "req_to_new_blocks",
)


def _upstream_carries_replay_start() -> bool:
    """Whether the pinned vLLM ships the feature this patch extends.

    Probed rather than read off a version: this repository supports two lanes
    (main, which grows the feature, and the release tag, which never will), and
    a version comparison would have to be revisited at every pin bump. The
    method is upstream's own record that a prefix hit was rewound, which is the
    thing this patch needs to exist; a pin without it has nothing to carry and
    nothing to fill.
    """
    return hasattr(Scheduler, "_mark_prefix_replay")


def _add_dataclass_field(cls: type, name: str, factory: type, annotation: type) -> None:
    """Declare one more field on an already-processed dataclass.

    ``SchedulerOutput`` and its payloads cross to the worker as msgpack, and
    msgspec encodes a dataclass from its *declared* fields, so an attribute
    written onto an instance never arrives. msgspec caches that field list per
    type, which is why this has to run at import time -- before the first
    payload is encoded -- rather than on first use.

    Re-running ``dataclasses.dataclass`` regenerates ``__init__``, but it only
    sets a method when it is absent from the class ``__dict__``, so the
    generated ``__init__`` has to be dropped first. ``__repr__`` and ``__eq__``
    are deliberately left alone: this payload defines its own (with token ids
    redacted, so they cannot leak into logs) and would otherwise be replaced.
    """
    if name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
        return
    cls.__annotations__[name] = annotation
    # A mutable default must be a factory, or every instance would share it.
    setattr(cls, name, dataclasses.field(default_factory=factory))
    delattr(cls, "__init__")
    dataclasses.dataclass(cls)
    assert name in cls.__dataclass_fields__, f"{cls.__name__}.{name} was not declared"  # type: ignore[attr-defined]
    assert name in inspect.signature(cls.__init__).parameters, (  # type: ignore[misc]
        f"{cls.__name__}.__init__ does not accept {name}; msgpack could not encode it"
    )


def _patch_cached_request_data_replay_start() -> None:
    """Declare ``CachedRequestData.replay_start`` so it survives the trip."""
    _add_dataclass_field(CachedRequestData, "replay_start", dict, dict[str, int])


def _patch_make_cached_request_data() -> None:
    """Fill in the replay start of every resumed request that rewinds."""
    original = Scheduler._make_cached_request_data
    if getattr(original, "_vllm_ascend_swa_replay_patched", False):
        return
    current_parameters = tuple(inspect.signature(original).parameters)
    if current_parameters != _EXPECTED_PARAMETERS:
        raise RuntimeError(
            "Cannot apply the SWA bounded replay resume patch: unexpected "
            f"Scheduler._make_cached_request_data signature {current_parameters}"
        )

    @functools.wraps(original)
    def _make_cached_request_data(
        self,
        running_reqs,
        resumed_reqs,
        num_scheduled_tokens,
        spec_decode_tokens,
        req_to_new_blocks,
    ):
        cached_reqs_data = original(
            self,
            running_reqs,
            resumed_reqs,
            num_scheduled_tokens,
            spec_decode_tokens,
            req_to_new_blocks,
        )
        # A dict keyed by request id rather than a list parallel to req_ids:
        # an absent entry is the common case and has to mean "no replay"
        # without depending on the payload's length.
        cached_reqs_data.replay_start = {
            request.request_id: request.replay_start for request in resumed_reqs if request.replay_start
        }
        return cached_reqs_data

    _make_cached_request_data._vllm_ascend_swa_replay_patched = True  # type: ignore[attr-defined]
    Scheduler._make_cached_request_data = _make_cached_request_data


if _upstream_carries_replay_start():
    # vLLM loads general plugins in the engine-core process and in every worker
    # (``load_general_plugins``), so this module reaches both sides of the
    # msgpack boundary -- the encoder needs the field declared to write it, the
    # decoder needs it declared to accept it.
    _patch_cached_request_data_replay_start()
    _patch_make_cached_request_data()
    logger.debug_once("SWA bounded replay: CachedRequestData.replay_start declared and filled from resumed requests.")
else:
    logger.debug_once(
        "SWA bounded replay: this vLLM pin predates the feature; the resumed-request replay start is not patched."
    )
