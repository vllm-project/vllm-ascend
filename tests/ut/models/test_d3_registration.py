# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib.util
from unittest.mock import patch

import pytest

from vllm_ascend import models

# vllm_ascend.models.d3 subclasses upstream vLLM's own D3ForSequenceClassification, which is in
# neither v0.31.0 nor v0.32.0 (see tests/e2e/.../pooling/test_d3.py for the same probe). Only the
# test below that actually imports vllm_ascend.models.d3 needs this; registration itself (below) is
# just a string mapping and runs on every vLLM version.
requires_native_d3 = pytest.mark.skipif(
    importlib.util.find_spec("vllm.model_executor.models.d3") is None,
    reason="D3 needs a vLLM revision with native D3 support",
)


def test_d3_sequence_classifier_uses_ascend_adapter():
    registrations = []
    with patch.object(
        models.ModelRegistry,
        "register_model",
        side_effect=lambda name, path: registrations.append((name, path)),
    ):
        models.register_model()

    assert (
        "D3ForSequenceClassification",
        "vllm_ascend.models.d3:AscendD3ForSequenceClassification",
    ) in registrations


@requires_native_d3
def test_d3_skips_readout_weight_in_backbone_pass():
    # Imported here, not at module level: it pulls in upstream vLLM's own D3ForSequenceClassification,
    # which collection would otherwise fail to import on a vLLM without native D3 support, regardless
    # of this skip marker (that only governs whether the test runs, not whether the module imports).
    from vllm_ascend.models.d3 import AscendD3ForSequenceClassification

    assert AscendD3ForSequenceClassification.is_unused_checkpoint_weight("weight")
    assert not AscendD3ForSequenceClassification.is_unused_checkpoint_weight("language_model.model.embed_tokens.weight")
