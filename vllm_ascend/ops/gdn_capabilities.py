# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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

"""Resolve the recurrent GDN verification window the loaded OPP accepts.

Two independent reasons can force the speculative branch onto vLLM's strided
Triton kernel:

* the installed AscendC kernel has a hard per-sequence window, or
* the profile records an observed numerical failure at a width the kernel does
  execute.

The first belongs to the OPP, not to the Python wheel: ``fla_npu`` publishes
``max_verify_len`` from the vendor tree that actually wins the load order
(``None`` means the kernel consumes a sequence in windows and has no cap).
The second is a stack workaround and stays in the hardware profile.

Keeping the merge here means the builder makes one decision for the whole
batch and ``_forward_core`` does not repeat (or contradict) it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# The pre-V2 arch22 kernel shipped a single eight-token window and returned
# without writing output or advancing state above it. It is the only reason
# the capability API can be missing (old wheel) or unpublished (old overlay),
# so unknown always means "at most eight" rather than "no limit".
_LEGACY_MAX_VERIFY_LEN = 8
_LEGACY_SOURCES = frozenset({"legacy-kernel-header", "unavailable", "unpublished"})


@dataclass(frozen=True)
class RecurrentGdnLimits:
    """One batch-level answer for the recurrent GDN dispatch.

    ``max_verify_len`` is the OPP's hard per-sequence limit; ``None`` means the
    OPP published no limit. ``legacy_fallback_min_query_len`` is the profile's
    numerical workaround, and is ``None`` once the OPP is known to be the V2
    window-loop kernel: the failure was observed on the single-window kernel,
    not on the replacement.
    """

    max_verify_len: int | None
    legacy_fallback_min_query_len: int | None
    published: bool
    source: str

    def requires_upstream_recurrent(self, query_len: int, *, adaptive: bool) -> bool:
        """Whether this query window must use vLLM's strided Triton kernel.

        Adaptive verification varies query lengths inside one graph and must
        retain the previous accepted state offset regardless of width, so it
        always uses the strided path.
        """

        if adaptive:
            return True
        if self.max_verify_len is not None and query_len > self.max_verify_len:
            return True
        return self.legacy_fallback_min_query_len is not None and query_len >= self.legacy_fallback_min_query_len


def build_limits(capability, fallback_min_query_len: int | None) -> RecurrentGdnLimits:
    """Merge an OPP capability record with a profile workaround.

    ``capability`` is a ``fla_npu`` ``RecurrentGdnCapability`` or ``None`` when
    the installed wheel predates the API. A capability that is not published
    is treated exactly like the old wheel: conservative, not optimistic.
    """

    if capability is None:
        return RecurrentGdnLimits(
            _LEGACY_MAX_VERIFY_LEN,
            fallback_min_query_len,
            False,
            "unavailable",
        )
    if not capability.published:
        return RecurrentGdnLimits(
            _LEGACY_MAX_VERIFY_LEN,
            fallback_min_query_len,
            False,
            capability.source,
        )
    # The numerical workaround describes the old single-window kernel. Once the
    # OPP publishes its own capability it is the V2 window-loop kernel, which
    # has been re-validated at the widths that used to fail; do not keep paying
    # for the workaround on it.
    legacy_fallback = fallback_min_query_len if capability.source in _LEGACY_SOURCES else None
    return RecurrentGdnLimits(
        capability.max_verify_len,
        legacy_fallback,
        True,
        capability.source,
    )


def _fla_capability(device_index: int):
    """Read the capability record from the loaded ``fla_npu``, if it can."""

    try:
        from fla_npu.ops import ascendc
    except ImportError:
        return None
    resolve = getattr(ascendc, "recurrent_gdn_capability", None)
    if resolve is None:
        return None
    try:
        return resolve(device_index)
    except Exception:
        # The vendor tree may not describe this device, or may fail while
        # querying it (driver not initialised, mismatched runtime). None of that
        # should take down a worker: the conservative limit is still a correct
        # answer for any kernel that cannot state its own window.
        return None


def _profile_fallback_min_query_len() -> int | None:
    try:
        from vllm_ascend.device.hardware_profile import get_current_hardware_profile

        return get_current_hardware_profile().gdn_fixed_spec_recurrent_fallback_min_query_len
    except Exception:
        # Offline unit tests and platforms without a profile keep the
        # capability decision alone.
        return None


@lru_cache(maxsize=16)
def _cached_limits(device_index: int) -> RecurrentGdnLimits:
    return build_limits(_fla_capability(device_index), _profile_fallback_min_query_len())


def get_recurrent_gdn_limits(device_index: int | None = None) -> RecurrentGdnLimits:
    """Capability limits for the device this worker will launch on."""

    if device_index is None:
        try:
            import torch

            device_index = int(torch.npu.current_device())
        except Exception:
            # Offline unit tests have no initialised NPU. The conservative
            # answer is the one an old wheel produces anyway.
            return build_limits(None, _profile_fallback_min_query_len())
    return _cached_limits(int(device_index))
