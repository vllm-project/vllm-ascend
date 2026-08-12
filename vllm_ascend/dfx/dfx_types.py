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

"""DFX shared types and constants.

- ``DumpPhase``: coarse dump arming / activation phase enum.
- ``DumpFinishMeta``: per-req wave stamps for dump-finish sidecar files.
- ``ILL_TYPE_*``: msprobe ILLDetector anomaly category codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DumpPhase(str, Enum):
    """Coarse dump arming / activation phase.

    Fine-grained forward gating still uses ``_dump_needs_forward`` /
    ``_dump_forward_seen`` inside ``Dumper``.
    """

    IDLE = "idle"
    PENDING = "pending"
    ACTIVE = "active"


@dataclass
class DumpFinishMeta:
    """Wave / dump correlation kept until the request finishes.

    Written to a sidecar file on ``clear_finished`` (not by rewriting the
    immediate anomaly report). ``dump_waves_after_report`` is
    ``activate_wave - arm_wave`` when both are known, else ``None``.
    """

    anomaly_type: str | None = None
    source: str | None = None
    dump_arm_wave: int | None = None
    dump_activate_wave: int | None = None
    dump_waves_after_report: int | None = None
    dump_count: int | None = None


# Align with msprobe response_anomaly ILLDetector ill_type codes.
ILL_TYPE_NONE = 0
ILL_TYPE_RARE = 1
ILL_TYPE_GARBLED = 2
ILL_TYPE_REPEAT = 3
ILL_TYPE_NAN = 4

ILL_TYPE_NAME: dict[int, str] = {
    ILL_TYPE_NONE: "none",
    ILL_TYPE_RARE: "rare",
    ILL_TYPE_GARBLED: "garbled",
    ILL_TYPE_REPEAT: "repetition",
    ILL_TYPE_NAN: "nan",
}
