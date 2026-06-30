#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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

import sys

import torch
import vllm
from vllm.model_executor.models.utils import _embedding_count_expression, _flatten_embeddings
from vllm.multimodal import NestedTensors


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    flattened = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype
    try:
        inputs_embeds[is_multimodal] = flattened.to(dtype=input_dtype)
    except RuntimeError as e:
        num_expected_tokens = is_multimodal.sum().item()
        assert isinstance(num_expected_tokens, int)

        if flattened.shape[0] != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {flattened.shape[0]} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e
        else:
            raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


def _apply_patch() -> None:
    """
    Replace ``utils._merge_multimodal_embeddings`` and propagate the replacement
    to all already-imported model modules.

    Background:
        Model modules under ``vllm.model_executor.models`` (e.g. ``qwen3_vl``,
        ``qwen3_5``, ``phi3v``) bind this function locally via
        ``from .utils import _merge_multimodal_embeddings`` at module top.
        That ``from ... import name`` statement copies the current attribute
        value into a module-level name in the importing module, NOT a live
        reference to the utils attribute. So patching only
        ``utils._merge_multimodal_embeddings`` does not propagate to model
        modules that have already been imported.

        By the time ``NPUWorker.__init__`` triggers ``adapt_patch()``, model
        modules are typically already imported (during model registry
        initialization, VllmConfig setup, etc.), making the original patch
        silently ineffective on the actual call sites.

    Fix:
        After replacing the utils attribute, sweep ``sys.modules`` and
        update any local binding in modules under
        ``vllm.model_executor.models`` whose ``_merge_multimodal_embeddings``
        attribute is still identical to the original function reference.
    """
    utils_mod = vllm.model_executor.models.utils
    orig_merge_mm = utils_mod._merge_multimodal_embeddings

    # 1) Replace utils module attribute (legacy behavior, kept for
    #    new model modules that are imported AFTER this patch runs).
    utils_mod._merge_multimodal_embeddings = _merge_multimodal_embeddings

    # 2) Sweep sys.modules to update local bindings in already-imported
    #    model modules. Use ``is`` for strict identity check so we never
    #    overwrite a binding that was deliberately replaced by another patch.
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not mod_name.startswith("vllm.model_executor.models"):
            continue
        # Skip the utils module itself; we already replaced it above.
        if mod is utils_mod:
            continue
        # Accessing or assigning attributes on arbitrary modules can raise
        # (e.g. lazy-loaded or partially initialized modules). Guard the whole
        # check-and-assign so a single misbehaving module never crashes worker
        # startup.
        try:
            if getattr(mod, "_merge_multimodal_embeddings", None) is orig_merge_mm:
                # mypy can't see the dynamically-assigned attribute on a generic
                # ModuleType obtained from sys.modules; this monkey patch is
                # intentional and the attribute is guaranteed to exist by the
                # getattr check above.
                mod._merge_multimodal_embeddings = (  # type: ignore[attr-defined]
                    _merge_multimodal_embeddings
                )
        except Exception:
            pass


_apply_patch()
