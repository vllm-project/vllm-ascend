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

Mark draft reconstruction so disabled drafts can inherit the target collection phase.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from typing import Any, cast
from unittest.mock import patch

import vllm.envs as vllm_envs
import vllm.model_executor.models.utils as model_utils
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator_module
import vllm.v1.worker.gpu.spec_decode.dspark.speculator as speculator_module
import vllm.v1.worker.gpu.spec_decode.dspark.utils as dspark_utils
import vllm.v1.worker.gpu.spec_decode.eagle.utils as eagle_utils
from vllm.config import ParallelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.ascend_config import draft_config_context
from vllm_ascend.worker.v2.pp_transport import resolve_spec_pp_support

_original_dspark_init = speculator_module.DSparkSpeculator.__init__
_original_get_draft_quant_config = model_utils.get_draft_quant_config
_original_load_dspark_model = dspark_utils.load_dspark_model
_original_maybe_share_target_embed = eagle_utils.maybe_share_target_embed

# Isolate the physical KV topology of each in-flight DSpark construction or load.
_DSPARK_PHYSICAL_PARALLEL_CONFIG: ContextVar[ParallelConfig | None] = ContextVar(
    "dspark_physical_parallel_config", default=None
)


@contextmanager
def dspark_context_parallel_config(vllm_config: VllmConfig) -> Iterator[None]:
    """Distinguish Kimi's replicated draft queries from its physical KV group."""
    spec = getattr(vllm_config, "speculative_config", None)
    draft = getattr(spec, "draft_model_config", None)
    target = getattr(spec, "target_parallel_config", None)
    physical_config = None
    if (
        getattr(spec, "method", None) == "dspark"
        and draft is not None
        and getattr(getattr(draft, "hf_config", None), "model_type", None) == "k3_dspark"
        and draft.use_mla
        and target is not None
        and target.prefill_context_parallel_size > 1
        and target.decode_context_parallel_size > 1
    ):
        physical_config = target
    token = _DSPARK_PHYSICAL_PARALLEL_CONFIG.set(physical_config)
    try:
        yield
    finally:
        _DSPARK_PHYSICAL_PARALLEL_CONFIG.reset(token)


def replace_dspark_config(config: Any, **changes: Any) -> Any:
    """Validate the physical topology before exposing the PCP=1 query view.

    Called only through the two upstream DSpark/DFlash module-local aliases.
    An ordinary ParallelConfig must still reject TP=1/PCP=1/DCP=8. Kimi's
    draft query view addresses an already-created PCP=8/DCP=8 KV group; it
    does not request a new worker group or another PCP query gather.
    """
    physical = _DSPARK_PHYSICAL_PARALLEL_CONFIG.get()
    if physical is None:
        return replace(config, **changes)
    if isinstance(config, ParallelConfig):
        pcp_size = changes.get("prefill_context_parallel_size", config.prefill_context_parallel_size)
        dcp_size = changes.get("decode_context_parallel_size", config.decode_context_parallel_size)
        if pcp_size == 1 and dcp_size == physical.decode_context_parallel_size:
            validated = replace(
                config,
                **{**changes, "prefill_context_parallel_size": physical.prefill_context_parallel_size},
            )
            # Keep the validated physical world-size information. Only the
            # attention query layout is replicated across the existing group.
            validated.prefill_context_parallel_size = 1
            return validated
    elif isinstance(config, VllmConfig):
        query_parallel = changes.get("parallel_config", config.parallel_config)
        if (
            query_parallel.prefill_context_parallel_size == 1
            and query_parallel.decode_context_parallel_size == physical.decode_context_parallel_size
        ):
            load_parallel = replace(
                query_parallel, prefill_context_parallel_size=physical.prefill_context_parallel_size
            )
            validated = replace(config, **{**changes, "parallel_config": load_parallel})
            validated.parallel_config = query_parallel
            return validated
    return replace(config, **changes)


@wraps(_original_dspark_init)
def _dspark_init(self, vllm_config, device):
    with dspark_context_parallel_config(vllm_config):
        _original_dspark_init(self, vllm_config, device)


def _pp_safe_maybe_share_target_embed(draft_model, draft_inner, target_inner):
    # PP stages past the first have no target embedding to alias. Drafts
    # whose post_process loads this stage's own vocab shard (e.g. the
    # Ascend DSpark align step) run after the share check, so skip the
    # share here instead of letting upstream raise; every other case keeps
    # the original behavior.
    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(target_inner, "embedding", None)
    if isinstance(target_embed, model_utils.PPMissingLayer):
        target_embed = None
    if target_embed is None and get_pp_group().world_size > 1 and hasattr(draft_model, "post_process"):
        return
    return _original_maybe_share_target_embed(draft_model, draft_inner, target_inner)


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
    eagle_utils.maybe_share_target_embed = _pp_safe_maybe_share_target_embed
    try:
        # Native draft loading already sets PP=1, but still reads the target's
        # manual layer partition. Keep it scoped to the target model.
        partition_mask = cast(AbstractContextManager[None], nullcontext())
        if spec_pp_support is not None:
            partition_mask = patch.object(vllm_envs, "VLLM_PP_LAYER_PARTITION", None)
        with partition_mask, draft_config_context(), dspark_context_parallel_config(vllm_config):
            return _original_load_dspark_model(target_model, vllm_config)
    finally:
        eagle_utils.maybe_share_target_embed = _original_maybe_share_target_embed
        if inherits_target_quant:
            model_utils.get_draft_quant_config = _original_get_draft_quant_config


# The speculator binds ``load_dspark_model`` by name at import time, so both
# the utils module and the speculator module must point at the wrapper.
dspark_utils.load_dspark_model = _load_dspark_model_with_target_quant
speculator_module.load_dspark_model = _load_dspark_model_with_target_quant

# Both the upstream runner and the Ascend speculator construct this class.
speculator_module.DSparkSpeculator.__init__ = _dspark_init
dflash_speculator_module.replace = replace_dspark_config
dspark_utils.replace = replace_dspark_config
