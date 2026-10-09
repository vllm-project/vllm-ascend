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
"""
VPP patch for the shared pipeline-parallel layer allocator ``make_layers``.

Previously every VPP-enabled model needed its own ``__init__`` patch that
reimplemented the *whole* constructor just to swap
``make_layers`` -> ``setup_vpp_layers`` (see the now-removed
``patch_vpp_deepseek``).  ``make_layers`` is the single entry point
*every* model uses to build its decoder stack under pipeline parallelism,
so patching it here makes V-shaped fold-back layer assignment available
to all of them automatically — no per-model shim required.

When ``vp_size > 1`` this drops in ``make_vpp_layers`` and attaches the
per-virtual-stage ranges to the returned ``ModuleList`` as
``modules.vpp_layer_ranges``, so a model's forward pass can pick the right
slice for the current virtual stage::

    vpp_ranges = getattr(self.layers, "vpp_layer_ranges", None)
    if vpp_ranges is None:                 # vanilla PP
        start, end = self.start_layer, self.end_layer
    else:                                  # VPP
        start, end = vpp_ranges[get_virtual_pipeline_parallel_rank()]

Scope
-----
This covers layer *construction* only.  The two remaining VPP concerns are
handled centrally:

* ``norm`` / ``lm_head`` placement -- upstream gates this on
  ``is_last_rank``, which under VPP is phase-latched in
  ``patch_distributed``: at construction time it returns the static
  fold-back topology, so the upstream ``__init__`` already places these
  layers on the correct rank without a per-model patch.
* Forward-pass range selection -- under ``ENABLE_VPP`` the model runner
  sets ``start_layer`` / ``end_layer`` from ``vpp_layer_ranges`` per
  virtual stage, so the standard upstream forward
  (``islice(self.layers, self.start_layer, self.end_layer)``) needs no
  change.
"""
from __future__ import annotations

import sys

from vllm.model_executor.models import utils as _model_utils

from vllm_ascend.distributed.vpp_utils import (
    get_custom_layer_ranges_for_rank,
    get_vp_size,
    make_vpp_layers,
)


_original_make_layers = _model_utils.make_layers


def _vpp_make_layers(
    num_hidden_layers: int,
    layer_fn,
    prefix: str,
):
    """Drop-in replacement for ``make_layers`` that is VPP-aware."""
    vp_size = get_vp_size()
    if vp_size <= 1:
        return _original_make_layers(num_hidden_layers, layer_fn, prefix)

    custom_ranges = get_custom_layer_ranges_for_rank()
    layer_ranges, modules = make_vpp_layers(
        num_hidden_layers,
        layer_fn,
        prefix,
        vp_size,
        custom_layer_ranges=custom_ranges,
    )
    # Expose the per-virtual-stage ranges so a model's forward can select
    # the right slice for the current vp_stage.
    modules.vpp_layer_ranges = layer_ranges

    # start_layer / end_layer span every virtual stage owned by this rank
    # (min start .. max end), mirroring what setup_vpp_layers sets.
    start_layer = layer_ranges[0][0]
    end_layer = layer_ranges[-1][1]
    return start_layer, end_layer, modules


# Replace the function in its defining module ...
_model_utils.make_layers = _vpp_make_layers

# ... and rebind every consumer that did ``from .utils import make_layers``.
# That import form binds ``make_layers`` into the importing module's own
# namespace, so patching ``utils.make_layers`` alone would leave already-
# imported model modules pointing at the original function.  Modules that
# are imported lazily *after* this patch pick up the new version
# automatically (they read ``utils.make_layers`` at import time); this loop
# fixes the modules that were imported eagerly, before the patch ran.
#
# Scoped to ``vllm.*`` modules: ``make_layers`` is only re-exported by vLLM
# model modules, and limiting the scan avoids touching unrelated modules
# (e.g. transformers ships its own unrelated ``make_layers`` and emits a
# deprecation warning merely on attribute access).  The identity check
# (``is _original_make_layers``) guarantees we never overwrite a different
# function that merely shares the name.
for _name, _mod in list(sys.modules.items()):
    if _mod is None or not _name.startswith("vllm"):
        continue
    if getattr(_mod, "make_layers", None) is _original_make_layers:
        _mod.make_layers = _vpp_make_layers
