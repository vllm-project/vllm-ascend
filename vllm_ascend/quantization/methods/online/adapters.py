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
"""Concrete Ascend online-quantization methods.

Each class pairs vLLM's online method contract (``OnlineLinearBase`` /
``OnlineMoEMethodBase``, which own the layerwise load-time lifecycle) with one
Ascend quantization scheme that performs the quantization and the runtime
layout.

They live in their own module, and are imported by the quantization config
rather than by ``methods/online/__init__.py``: ``method_adapters`` imports the
scheme package, so exporting these classes there would create an import cycle.
"""

from vllm_ascend.quantization.method_adapters import (
    AscendOnlineFusedMoEMethod,
    AscendOnlineLinearMethod,
)

from .int8 import (
    AscendOnlineInt8W8A8FusedMoEMethod,
    AscendOnlineInt8W8A8LinearMethod,
)

__all__ = [
    "AscendOnlineInt8LinearMethod",
    "AscendOnlineInt8MoEMethod",
]


class AscendOnlineInt8LinearMethod(AscendOnlineLinearMethod):
    """Online INT8 W8A8_DYNAMIC method for dense linear layers."""

    scheme_cls = AscendOnlineInt8W8A8LinearMethod


class AscendOnlineInt8MoEMethod(AscendOnlineFusedMoEMethod):
    """Online INT8 W8A8_DYNAMIC method for routed experts."""

    scheme_cls = AscendOnlineInt8W8A8FusedMoEMethod
