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
"""Ascend online (load-time) quantization methods.

vLLM's online quantization is table driven: ``(layer kind, QuantKey)`` selects a
method class and a ``--quantization`` name selects default
``QuantizationConfigArgs``.

This package declares the recipes Ascend supports: ``ONLINE_LINEAR_METHODS`` /
``ONLINE_MOE_METHODS`` for the dispatch and ``ONLINE_SHORTHANDS`` for the
``--quantization`` names. ``patch_online_quant`` applies them on top of vLLM's
tables; vLLM's other entries are left untouched, and a recipe Ascend cannot
execute is rejected explicitly by ``AscendOnlineQuantizationConfig`` when the
config is built.
"""

from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8StaticChannelSym,
)

from .int8 import (
    AscendInt8PerChannelOnlineLinearMethod,
    AscendInt8PerChannelOnlineMoEMethod,
    quantize_last_dim_to_int8,
)

# ``(layer kind, QuantKey) -> Ascend method class``. ``patch_online_quant``
# writes these into vLLM's dispatch tables; their keys are the weight keys
# ``AscendOnlineQuantizationConfig`` accepts, so listing a method here is what
# makes a recipe supported end to end.
ONLINE_LINEAR_METHODS: dict[QuantKey, type] = {
    kInt8StaticChannelSym: AscendInt8PerChannelOnlineLinearMethod,
}
ONLINE_MOE_METHODS: dict[QuantKey, type] = {
    # vLLM maps this key to its CUDA-only Int8OnlineMoEMethod.
    kInt8StaticChannelSym: AscendInt8PerChannelOnlineMoEMethod,
}

# Weight keys Ascend has an execution path for, for either layer kind.
SUPPORTED_ONLINE_WEIGHT_KEYS: frozenset[QuantKey] = frozenset(ONLINE_LINEAR_METHODS) | frozenset(ONLINE_MOE_METHODS)

# The online ``--quantization`` names Ascend supports, with the config we test
# them with. They are *applied on top of* vLLM's table: a name vLLM already
# knows is rewritten to this definition, every other vLLM entry is left
# untouched, and a recipe Ascend cannot execute is rejected by
# ``AscendOnlineQuantizationConfig`` instead of being removed here.
ONLINE_SHORTHANDS: dict[str, QuantizationConfigArgs] = {
    # Ascend's own name: per-channel INT8 weights with dynamically per-token
    # INT8 activations, for dense linear layers and routed experts. vLLM's
    # `int8_per_channel_weight_only` does not describe this path (it is
    # experts-only and its name implies weight-only quantization), so it is not
    # reused.
    "int8_per_channel": QuantizationConfigArgs(
        linear=QuantSpec(weight=kInt8StaticChannelSym),
        moe=QuantSpec(weight=kInt8StaticChannelSym),
    ),
}

__all__ = [
    "AscendInt8PerChannelOnlineLinearMethod",
    "AscendInt8PerChannelOnlineMoEMethod",
    "ONLINE_LINEAR_METHODS",
    "ONLINE_MOE_METHODS",
    "ONLINE_SHORTHANDS",
    "SUPPORTED_ONLINE_WEIGHT_KEYS",
    "quantize_last_dim_to_int8",
]
