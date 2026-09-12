# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter must be built from the vllm-ascend subclasses.

vLLM main resolves the inner model and the decoder layer through the
``model_cls`` / ``decoder_layer_cls`` class attributes (upstream PR 52816), so
overriding the module globals no longer reaches them. Without these hooks the
drafter is assembled from the DFlash1 classes and weight loading dies on the
checkpoint's ``candidate_selector.*`` tensors.
"""

import pytest
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

from vllm_ascend.models.qwen3_dflash2 import (
    DFlash2Qwen3DecoderLayer,
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)

# 0.27.1 and some main snapshots still hardcode DFlashQwen3Model /
# DFlashQwen3DecoderLayer in the parent ctors; the hooks only exist once
# upstream PR 52816 is on the tree. The CI pin is one of those snapshots.
_HAS_UPSTREAM_HOOKS = hasattr(DFlashQwen3ForCausalLM, "model_cls") and hasattr(DFlashQwen3Model, "decoder_layer_cls")


def test_inner_model_hook_points_at_the_dflash2_model():
    assert DFlash2Qwen3ForCausalLM.model_cls is DFlash2Qwen3Model


def test_decoder_layer_hook_points_at_the_dflash2_layer():
    assert DFlash2Qwen3Model.decoder_layer_cls is DFlash2Qwen3DecoderLayer


def test_dflash2_subclasses_the_upstream_dflash1_types():
    assert issubclass(DFlash2Qwen3Model, DFlashQwen3Model)
    assert issubclass(DFlash2Qwen3DecoderLayer, DFlashQwen3DecoderLayer)


@pytest.mark.skipif(
    not _HAS_UPSTREAM_HOOKS,
    reason="this vLLM pin has no model_cls / decoder_layer_cls hooks",
)
def test_hooks_override_the_upstream_dflash1_defaults():
    assert DFlashQwen3ForCausalLM.model_cls is DFlashQwen3Model
    assert DFlashQwen3Model.decoder_layer_cls is DFlashQwen3DecoderLayer
