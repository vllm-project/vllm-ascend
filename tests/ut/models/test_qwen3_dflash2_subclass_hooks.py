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
from vllm_ascend.utils import vllm_version_is

# 0.27.1 predates the hooks and reads the module globals, which the ctors swap.
requires_upstream_hooks = pytest.mark.skipif(
    vllm_version_is("0.27.1"),
    reason="vLLM 0.27.1 has no model_cls / decoder_layer_cls hooks",
)


def test_inner_model_hook_points_at_the_dflash2_model():
    assert DFlash2Qwen3ForCausalLM.model_cls is DFlash2Qwen3Model


def test_decoder_layer_hook_points_at_the_dflash2_layer():
    assert DFlash2Qwen3Model.decoder_layer_cls is DFlash2Qwen3DecoderLayer


@requires_upstream_hooks
def test_hooks_override_the_upstream_dflash1_defaults():
    assert DFlashQwen3ForCausalLM.model_cls is DFlashQwen3Model
    assert DFlashQwen3Model.decoder_layer_cls is DFlashQwen3DecoderLayer
    assert issubclass(DFlash2Qwen3Model, DFlashQwen3Model)
    assert issubclass(DFlash2Qwen3DecoderLayer, DFlashQwen3DecoderLayer)
