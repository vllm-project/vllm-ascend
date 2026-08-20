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

"""DFX (design for eXcellence) observability.

Components:
1. Runtime config — JSON + per-DP broadcast or file poll (``DfxRuntimeConfig``)
2. Input filter — detect-time gating (``InputFilterManager``)
3. Detector — anomaly checks returning ``AnomalyAlert`` (``DetectorManager``)
4. Dump / observability — msprobe dump + ``ascend_log`` switch (``Dumper``)
5. Report — short anomaly records under ``dfx/report`` (``DfxReportWriter``)
6. Request state — per-req shared memory + unified clear (``RequestDfxStore``)
7. I/O snapshot — report views over Store (``RequestIoSnapshotManager``)
8. Processor — runner hooks wiring the above (``DfxProcessor``)

Call chain: model runner → ``DfxProcessor`` → ``DetectorManager`` → (input filter
gate) → detector → ``AnomalyAlert`` → ``Dumper.handle_anomaly_alert`` (+ report).
"""

from vllm_ascend.dfx.detector.alert import AnomalyAlert
from vllm_ascend.dfx.dfx_types import (
    ILL_TYPE_GARBLED,
    ILL_TYPE_NAME,
    ILL_TYPE_NAN,
    ILL_TYPE_NONE,
    ILL_TYPE_RARE,
    ILL_TYPE_REPEAT,
    DumpPhase,
)
from vllm_ascend.dfx.dumper import Dumper
from vllm_ascend.dfx.processor import DfxProcessor
from vllm_ascend.dfx.runtime_config import DfxRuntimeConfig

__all__ = [
    "AnomalyAlert",
    "DfxProcessor",
    "DfxRuntimeConfig",
    "DumpPhase",
    "Dumper",
    "ILL_TYPE_GARBLED",
    "ILL_TYPE_NAME",
    "ILL_TYPE_NAN",
    "ILL_TYPE_NONE",
    "ILL_TYPE_RARE",
    "ILL_TYPE_REPEAT",
]
