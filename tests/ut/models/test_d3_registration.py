# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

from vllm_ascend import models


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
