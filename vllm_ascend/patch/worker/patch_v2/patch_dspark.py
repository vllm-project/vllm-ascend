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
"""Patch DSpark draft loading to inherit the target quant config.

DSpark supports two kinds of drafts:

* Self-contained drafts (e.g. Qwen3 DSpark speculators) ship their own weights
  and ``quantization_config``; ``get_draft_quant_config`` resolves their own
  quant scheme and must be left alone.
* Same-checkpoint drafts (DeepSeek-V4 DSpark, weights under ``mtp.*``) declare
  no quantization of their own, but reuse the target weights. Upstream
  ``load_dspark_model`` derives the draft quant config via
  ``get_draft_quant_config``, which returns ``None`` here. That builds an
  unquantized draft, and a W4A8/W8A8 target checkpoint cannot be loaded into it
  (the draft linear layers lack the ``weight_offset``/``weight_scale``/
  ``scale_bias`` params the checkpoint ships), failing with a KeyError.

The same-checkpoint fix is a general one; a long-term plan exists to contribute
it upstream to ``load_dspark_model``.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import cast
from unittest.mock import patch

import vllm.envs as vllm_envs
import vllm.model_executor.models.utils as model_utils
import vllm.v1.worker.gpu.spec_decode.dspark.speculator as speculator_module
import vllm.v1.worker.gpu.spec_decode.dspark.utils as dspark_utils

from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support

_original_get_draft_quant_config = model_utils.get_draft_quant_config
_original_load_dspark_model = dspark_utils.load_dspark_model


def _load_dspark_model_with_target_quant(target_model, vllm_config):
    # ``load_dspark_model`` imports ``get_draft_quant_config`` from inside the
    # function body, so overriding the module attribute before the call and
    # restoring it afterwards is enough to redirect it. Only alias it for
    # same-checkpoint drafts; self-contained drafts keep their own quant.
    speculative_config = vllm_config.speculative_config
    draft_model_config = speculative_config.draft_model_config
    inherits_target_quant = draft_model_config.model == vllm_config.model_config.model
    spec_pp_support = resolve_spec_pp_support(vllm_config)
    if inherits_target_quant:
        model_utils.get_draft_quant_config = lambda _vllm_config: vllm_config.quant_config
    try:
        # Native draft loading already sets PP=1, but still reads the target's
        # manual layer partition. Keep it scoped to the target model.
        partition_mask = cast(AbstractContextManager[None], nullcontext())
        if spec_pp_support is not None:
            partition_mask = patch.object(vllm_envs, "VLLM_PP_LAYER_PARTITION", None)
        with partition_mask:
            return _original_load_dspark_model(target_model, vllm_config)
    finally:
        if inherits_target_quant:
            model_utils.get_draft_quant_config = _original_get_draft_quant_config


# The speculator binds ``load_dspark_model`` by name at import time, so both
# the utils module and the speculator module must point at the wrapper.
dspark_utils.load_dspark_model = _load_dspark_model_with_target_quant
speculator_module.load_dspark_model = _load_dspark_model_with_target_quant
