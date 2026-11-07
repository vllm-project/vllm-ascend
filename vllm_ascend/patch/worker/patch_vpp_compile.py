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
"""Per-virtual-stage torch.compile artifacts for virtual pipeline parallelism.

Under VPP the runner points ``inner_model.start_layer`` / ``end_layer`` at a
different layer slice for every virtual stage before the wrapped forward runs.
``@support_torch_compile`` drops all Dynamo guards, so the second stage never
re-traces: it replays the first stage's compiled subgraph and computes the
wrong layers. The cache keys have the same blind spot -- the AOT artifact path
is derived from the env/config/model hash, and the backend artifact directory
from ``rank/prefix`` -- neither of which changes with the virtual stage.

This module makes the compile state a property of the *virtual stage*:

* ``bind_vpp_compile_stage`` installs (or builds) the compile state of the
  stage that is about to run, and activates the stage for the duration of the
  forward so the two cache keys below can see it;
* ``capture_vpp_compile_stage`` stores whatever the forward produced, in a
  ``finally``, so the next visit to the stage is a cheap state swap.

Two upstream cache keys are extended while a stage is active. Both return the
original value verbatim when no stage is active, which is always the case
without VPP -- no cache key, directory or file name changes on that path.

* ``vllm.compilation.caching.aot_compile_hash_factors`` gains one
  ``vpp_stage=<n>`` factor, which moves the AOT artifact, its rank directory
  and its ``TORCHINDUCTOR_CACHE_DIR`` into a per-stage subtree.
* ``VllmBackend.prefix`` gains an ``__vp<n>`` suffix. The AOT hash directory is
  not enough on its own: ``compilation_config.cache_dir`` (the inductor
  artifact root keyed by env/code/compiler hashes) is computed once and
  reused, so without a per-stage prefix the stages would share one
  ``artifact_compile_range_*`` directory.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from vllm.config import set_current_vllm_config
from vllm.logger import logger

# The virtual stage whose compile state is currently installed, or None.
# Consulted by the two cache-key patches below.
_CURRENT_VPP_STAGE: int | None = None

# Compile state owned by one virtual stage. The upstream wrapper distinguishes
# AOT (`aot_compiled_fn`), bytecode hook (`_compiled_bytecode`) and plain
# `torch.compile` (`_compiled_callable`) dispatch, so all three move together
# with their bookkeeping flags.
_STATE_FIELDS = (
    "aot_compiled_fn",
    "_compiled_bytecode",
    "_compiled_callable",
    "compiled",
    "was_aot_compile_fn_loaded_from_disk",
)

_ARTIFACTS_ATTR = "_vpp_compile_artifacts"

# The cache keys these patches wrap, kept for tests and for a clean identity
# check: with no stage active the patched callables delegate unchanged.
_ORIGINAL_AOT_HASH_FACTORS: Any = None
_ORIGINAL_BACKEND_INIT: Any = None


def current_vpp_compile_stage() -> int | None:
    """Return the virtual stage whose compile state is installed, if any."""
    return _CURRENT_VPP_STAGE


class CompileStageBinding:
    """Handle for one virtual-stage forward, undone by ``capture``."""

    __slots__ = ("model", "stage", "prev_stage")

    def __init__(self, model: Any, stage: int, prev_stage: int | None) -> None:
        self.model = model
        self.stage = stage
        self.prev_stage = prev_stage


def bind_vpp_compile_stage(model: Any, stage: int) -> CompileStageBinding | None:
    """Select the compile state for ``stage`` before its forward runs.

    Returns ``None`` when the model is not compiled at all, in which case the
    caller has nothing to capture and the stage-aware cache keys stay inert.
    """
    if getattr(model, "do_not_compile", True):
        return None

    global _CURRENT_VPP_STAGE
    binding = CompileStageBinding(model, stage, _CURRENT_VPP_STAGE)
    _CURRENT_VPP_STAGE = stage

    artifacts = model.__dict__.setdefault(_ARTIFACTS_ATTR, {})
    artifact = artifacts.get(stage)
    if artifact is not None:
        for name, value in artifact.items():
            setattr(model, name, value)
        return binding
    if stage == 0 and not artifacts:
        # The wrapper built at model construction has not compiled anything
        # yet, so it is this stage's compiler already.
        return binding

    try:
        if torch.npu.is_current_stream_capturing():
            # Compilation has to finish before the ACL graph capture starts:
            # the captured graph must record the compiled callable, not a trace.
            raise RuntimeError(
                f"[vllm-ascend][VPP compile] virtual stage {stage} is being compiled while an ACL graph is capturing"
            )
        _reinitialize_wrapper(model)
    except BaseException:
        # No binding is returned, so nothing else will restore the stage.
        _CURRENT_VPP_STAGE = binding.prev_stage
        raise
    return binding


def capture_vpp_compile_stage(binding: CompileStageBinding | None) -> None:
    """Store what the stage's forward compiled and leave the stage context."""
    if binding is None:
        return
    model = binding.model
    artifacts = model.__dict__.setdefault(_ARTIFACTS_ATTR, {})
    artifacts[binding.stage] = {name: getattr(model, name, None) for name in _STATE_FIELDS}
    global _CURRENT_VPP_STAGE
    _CURRENT_VPP_STAGE = binding.prev_stage


def _reinitialize_wrapper(model: Any) -> None:
    """Give ``model`` a fresh, empty compile wrapper for the active stage."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    # Drop the previous stage's bytecode hook before a new one is registered.
    model.cleanup()

    for name in _STATE_FIELDS:
        setattr(model, name, None if name != "compiled" else False)

    # Dropping all guards means Dynamo would answer the re-trace of the same
    # code object straight from its cache, staged on the previous stage.
    torch._dynamo.eval_frame.remove_from_cache(model.original_code_object())

    # ``__init__`` reads the config from the ambient context and rebuilds the
    # backend, whose ``prefix`` picks up the active stage.
    with set_current_vllm_config(model.vllm_config):
        TorchCompileWithNoGuardsWrapper.__init__(
            model,
            compile_prefix=model._compile_prefix,
            is_encoder=model._is_encoder,
        )
    logger.info(
        "[vllm-ascend][VPP compile] compiling a separate artifact for virtual stage %s",
        _CURRENT_VPP_STAGE,
    )


def _stage_prefixed_prefix(prefix: str, stage: int) -> str:
    """Suffix a backend prefix with the virtual stage (idempotently)."""
    suffix = f"__vp{stage}"
    if prefix.endswith(suffix):
        # Deserializing a per-stage AOT artifact rebuilds the backend with the
        # already-suffixed prefix.
        return prefix
    return f"{prefix}{suffix}"


def _patch_cache_keys() -> None:
    from vllm.compilation import caching as _caching
    from vllm.compilation.backends import VllmBackend

    global _ORIGINAL_AOT_HASH_FACTORS, _ORIGINAL_BACKEND_INIT
    original_factors = _caching.aot_compile_hash_factors
    _ORIGINAL_AOT_HASH_FACTORS = original_factors

    def aot_compile_hash_factors(vllm_config: Any) -> list[str]:
        factors = original_factors(vllm_config)
        stage = _CURRENT_VPP_STAGE
        if stage is None:
            return factors
        return [*factors, f"vpp_stage={stage}"]

    _caching.aot_compile_hash_factors = aot_compile_hash_factors

    original_init = VllmBackend.__init__
    _ORIGINAL_BACKEND_INIT = original_init

    def __init__(self: VllmBackend, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        stage = _CURRENT_VPP_STAGE
        if stage is not None:
            self.prefix = _stage_prefixed_prefix(self.prefix, stage)

    VllmBackend.__init__ = __init__


_patch_cache_keys()


@contextlib.contextmanager
def vpp_compile_stage(model: Any, stage: int):
    """Bind, run and capture one virtual-stage forward."""
    binding = bind_vpp_compile_stage(model, stage)
    try:
        yield binding
    finally:
        capture_vpp_compile_stage(binding)
