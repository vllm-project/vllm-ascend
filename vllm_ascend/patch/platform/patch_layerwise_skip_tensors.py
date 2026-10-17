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
"""Make vLLM's layerwise weight-reload accounting agree with Ascend's models.

Why this patch exists
---------------------
``vllm.model_executor.model_loader.reload`` drives its per-module completion
trigger off ``LayerReloadingInfo.load_numel`` vs ``load_numel_total``:

* ``load_numel_total`` is ``get_layer_size(layer)``, i.e. the element count of
  every parameter/buffer of the module **except** the names listed in
  ``reload.meta.SKIP_LOAD_TENSORS``;
* a tensor named in that set is also left unwrapped by
  ``_wrap_parameters_weight_loader``, so it never contributes to
  ``load_numel``.

DeepSeek-V4 (and the Ascend runtimes around it) register a number of *runtime
derived* tensors that no checkpoint and no weight-update payload ever carries:
the hash-router ``tid2eid`` table, the VL ``bias_vl``, the attention
``attn_sink``, the DSA RoPE tables, the vision ``image_*`` marker vectors and
the EPLB/EP state.  With the stock six-entry skip set, those tensors are counted
into ``load_numel_total`` but never arrive, so the trigger is unreachable, the
buffered weights are never applied, and the worker logs::

    WARNING [layerwise.py] <Module>: Failed to load weights

for every such module on every reload.  The served model then keeps a mix of
pre-update and zero-filled storage, and its output no longer matches a normal
startup load of the same payload.

Two smaller holes are closed here as well:

* ``materialize_meta_tensor`` returns ``torch.empty_strided`` storage.  A module
  may legally receive fewer elements than ``load_numel_total`` (deliberate
  loader skips, padding), so region copies would propagate uninitialised garbage
  into the woken kernel tensors -- e.g. a garbage expert id read by the MoE
  dispatch.  Zero-filling makes uncovered regions defined.
A completeness gate is deliberately *not* added here: the Ascend weight-transfer
engines fail loudly on their own (``VLLM_IPC_STRICT_UPDATE=1``) and the test
oracle already compares the served weights against an independent startup load
of the same payload, which is a stronger check than a coverage report.

Every change is guarded so that a future vLLM which already ships the behaviour
(a larger skip set, a zero-filling materialiser) is left untouched.
"""

import contextlib

import vllm.model_executor.model_loader.reload.meta as _meta

# --------------------------------------------------------------------------
# 1. Extend SKIP_LOAD_TENSORS in place.
#
# Mutating the existing set (instead of rebinding the module attribute) matters:
# ``reload.layerwise`` does ``from .meta import SKIP_LOAD_TENSORS`` at import
# time and therefore holds a reference to *this* object, while
# ``reload.utils.get_layer_size`` imports the name lazily.  An in-place update is
# seen by both paths regardless of import order.
# --------------------------------------------------------------------------
_ASCEND_RUNTIME_TENSORS = {
    # Ascend EPLB / expert-parallel state promoted to named buffers. Created at
    # model init, never present in a checkpoint, never sent by the trainer;
    # ``expert_ids_per_ep_rank`` alone contributes num_experts elements per
    # layer and would make the completion trigger unreachable.
    "expert_ids_per_ep_rank",
    "log2phy",
    "moe_load",
    "load_counter",
    # DeepSeek-V4 served-side routing / vision tensors read by the forward
    # kernels. After a level-2 sleep their freshly-woken storage is
    # uninitialised; garbage expert ids fed to MoeDistributeDispatchV2 fault the
    # whole worker set (AI Core 507035). The checkpoint ships trained values for
    # these, so keeping the loaded values alive across a reload is what makes the
    # first post-update forward correct.
    "tid2eid",
    "bias_vl",
    # ``attn_sink`` is ONE nn.Parameter registered on two modules
    # (``DeepseekV4Attention`` and the DSA attention that receives it as a
    # keyword), so ``get_layer_size`` counts it twice while the write only
    # credits one of them and the other never reaches its trigger.
    "attn_sink",
    # RoPE positional-embedding caches (generic ``cos_sin_cache`` plus the DSA
    # tables the Ascend ``ComplexExpRotaryEmbedding`` owns as non-persistent
    # buffers). Runtime-derived from max_position_embeddings and never present in
    # a checkpoint; zero-filling them destroys positional encoding for every
    # layer, while counting them makes the trigger unreachable.
    "cos_sin_cache",
    "cos_cached",
    "sin_cached",
    "full_rope_cos",
    "full_rope_sin",
    # Vision-tower marker vectors and the lazily built sentinel table of
    # ``AscendDeepseekV4ForConditionalGeneration``.
    "image_start",
    "image_end",
    "image_newline",
    "image_pad",
    "_sentinel_table",
}

if not _ASCEND_RUNTIME_TENSORS <= _meta.SKIP_LOAD_TENSORS:
    _meta.SKIP_LOAD_TENSORS.update(_ASCEND_RUNTIME_TENSORS)

# ``SKIP_TENSORS`` is derived at import time from the old contents, so refresh it
# too when it exists (it gates which tensors are moved to the meta device).
if hasattr(_meta, "SKIP_TENSORS"):
    _meta.SKIP_TENSORS.update(_ASCEND_RUNTIME_TENSORS)


# --------------------------------------------------------------------------
# 2. Zero-fill materialised meta tensors when vLLM does not already do so.
#
# ``SKIP_LOAD_TENSORS``/``SKIP_TENSORS`` may legitimately point at an aliased
# function, so resolve the current implementation instead of assuming a module
# level name.
# --------------------------------------------------------------------------
def _patch_materialize_zero_fill() -> None:
    materialize = getattr(_meta, "materialize_meta_tensor", None)
    if materialize is None:
        return
    source = None
    try:
        import inspect

        source = inspect.getsource(materialize)
    except (OSError, TypeError):
        source = None
    if source is not None and "zero_()" in source:
        # Already zero-fills (newer vLLM).
        return

    def materialize_meta_tensor_zero_filled(meta_tensor):
        tensor = materialize(meta_tensor)
        # Uncovered regions of a partially covered module must be defined (0)
        # rather than uninitialised storage that can corrupt runtime kernels.
        with contextlib.suppress(Exception):  # best effort, never break loading
            tensor.zero_()
        return tensor

    _meta.materialize_meta_tensor = materialize_meta_tensor_zero_filled


_patch_materialize_zero_fill()
