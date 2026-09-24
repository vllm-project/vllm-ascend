#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Internal runtime_guard constants (not RuntimeConfig / not user-tunable).

Rule of thumb: if an operator would never need to change the value under load,
disk pressure, or batch size, keep it here. Tunables live in
``runtime_config._defaults``.
"""

from __future__ import annotations

# Finished but sample_waves still non-empty this many real-steps later → force
# reap (stuck / dropped AsyncOutput). Prefer wave-empty as the primary signal.
MAX_DEFERRED_REAP_WAVES = 8

# Post-reap late async appends: remember recently cleared ids so a zombie
# recreate in append_output_ids can be stamped finished=True for reap.
REAPED_RING_MAX = 1024

# Same (incident_type, req_id) on-disk write backoff (wave-based, not wall-clock).
# After the 1st write, next needs +64 waves; then +128, +256, … (doubles).
# Cap is ``report.max_per_req`` (RuntimeConfig).
SAME_PAIR_BACKOFF_BASE_WAVES = 64
