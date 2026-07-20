# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
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
# AscendEagleSpeculator: inherits AscendAutoRegressiveSpeculator (shared Ascend
# mixin) + EagleSpeculator (load_draft_model). All Ascend overrides -- including
# init_cudagraph_manager and _build_draft_attn_metadata -- live in the mixin;
# Eagle overrides nothing.
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


class AscendEagleSpeculator(AscendAutoRegressiveSpeculator, EagleSpeculator):
    """Ascend Eagle speculator. Inherits all Ascend overrides from
    AscendAutoRegressiveSpeculator; no overrides here."""
