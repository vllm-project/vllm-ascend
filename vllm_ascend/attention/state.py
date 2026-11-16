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
# This file is a part of the vllm-ascend project.
#

"""Backend-agnostic attention states shared by the Ascend attention backends.

Historically ``AscendAttentionState`` was defined in the GQA v1 backend module
(``vllm_ascend/attention/attention_v1.py``), which forced the MLA/SFA/DSA
backends to import from the GQA module. It now lives in this leaf module, which
only depends on the Python standard library, so every backend can import it
without a cross-backend dependency. The old location still re-exports the enum
for backward compatibility.
"""

from enum import Enum


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4
