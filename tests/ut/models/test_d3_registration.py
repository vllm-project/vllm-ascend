# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

from vllm_ascend import models
from vllm_ascend.models.d3 import AscendD3ForSequenceClassification


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


def test_d3_skips_readout_weight_in_backbone_pass():
    assert AscendD3ForSequenceClassification.is_unused_checkpoint_weight("weight")
    assert not AscendD3ForSequenceClassification.is_unused_checkpoint_weight("language_model.model.embed_tokens.weight")
