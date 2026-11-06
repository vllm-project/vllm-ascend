# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm_ascend._310p.fused_moe.grouped_topk_router import AscendGroupedTopKRouter310
from vllm_ascend.ops.fused_moe.router.pluggable import AscendBiasRouter, AscendGroupedRouter, AscendTopKRouter


class AscendTopKRouter310(AscendTopKRouter, AscendGroupedTopKRouter310):
    _compute_routing = AscendGroupedTopKRouter310._compute_routing


class AscendGroupedRouter310(AscendGroupedRouter, AscendGroupedTopKRouter310):
    _compute_routing = AscendGroupedTopKRouter310._compute_routing


class AscendBiasRouter310(AscendBiasRouter, AscendGroupedTopKRouter310):
    _compute_routing = AscendGroupedTopKRouter310._compute_routing
