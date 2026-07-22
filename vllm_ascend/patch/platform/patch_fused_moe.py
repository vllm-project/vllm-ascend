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
#

# Patch vllm's FusedMoE factory to use AscendMoERunner by default.
#
# vllm's FusedMoE is a factory function (not a class). deepseek_v2 and other
# models do `from vllm.model_executor.layers.fused_moe import FusedMoE` and
# call it directly, so we must patch the binding in the package __init__ as
# well as the layer module before any model is imported.
#
# Import order in worker.__init__:
#   1. adapt_patch()  ->  this file runs  ->  FusedMoE patched
#   2. from vllm_ascend import ops
#   3. model loading  ->  deepseek_v2 imported  ->  gets patched FusedMoE  ✓

import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer
from vllm.model_executor.layers.fused_moe import MoERunner as _UpstreamMoERunner

from vllm_ascend.utils import is_310p


def _has_actual_quantization() -> bool:
    """Check whether the current model has actual quantization targets.

    Models like Kimi-K2-Thinking declare ``quantization=compressed-tensors``
    in their config but have no actual quantization scheme (``quantization_config
    is None``).  For these models we must NOT force AscendMoERunner because
    the Ascend runner triggers NPU-specific weight-layout processing that
    doubles peak memory during weight creation and causes OOM on large MoE
    models.

    Returns:
        True if the model has real quantization that requires AscendMoERunner.
    """
    try:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        quant_config = vllm_config.quantization_config
        if quant_config is None:
            return False
        from vllm_ascend.quantization.compressed_tensors_config import (
            AscendCompressedTensorsConfig,
        )

        if isinstance(quant_config, AscendCompressedTensorsConfig):
            target_scheme_map = getattr(quant_config, "target_scheme_map", None)
            if not target_scheme_map:
                return False
        return True
    except Exception:
        return True

# Capture the real original before fused_moe.py's module-level code runs.
_original_FusedMoE = _fused_moe_layer.FusedMoE

if is_310p():
    from vllm_ascend._310p.fused_moe.fused_moe import AscendMoERunner310 as _DefaultAscendMoERunner
else:
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner as _DefaultAscendMoERunner


def _ascend_FusedMoE(*args, runner_cls=None, runner_args=None, **kwargs):
    if runner_cls is None:
        if _has_actual_quantization():
            runner_cls = _DefaultAscendMoERunner
        else:
            # Use the upstream default MoERunner for unquantized models.
            # This avoids the Ascend-specific weight-layout processing that
            # doubles peak memory during loading.
            runner_cls = _UpstreamMoERunner
    # 'hash' is a DeepSeek V4 flag already consumed before FusedMoE is called;
    # 'tid2eid' is Ascend-specific and must reach AscendMoERunner via runner_args.
    kwargs.pop("hash", None)
    tid2eid = kwargs.pop("tid2eid", None)
    if tid2eid is not None:
        runner_args = dict(runner_args) if runner_args is not None else {}
        runner_args["tid2eid"] = tid2eid
    return _original_FusedMoE(*args, runner_cls=runner_cls, runner_args=runner_args, **kwargs)


_fused_moe_layer.FusedMoE = _ascend_FusedMoE
_fused_moe_pkg.FusedMoE = _ascend_FusedMoE
