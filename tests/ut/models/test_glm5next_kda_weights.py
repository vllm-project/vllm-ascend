# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.model_executor.model_loader import utils as loader_utils

from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention


def test_loader_prepares_conv_weights_before_capture_and_refreshes_on_reload(monkeypatch):
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.register_buffer("_merged_conv_weight", None, persistent=False)
    for offset, name in zip((0, 10, 20), ("q_conv1d", "k_conv1d", "v_conv1d")):
        projection = torch.nn.Module()
        projection.weight = torch.nn.Parameter(torch.arange(8, dtype=torch.float32).reshape(2, 1, 4) + offset)
        setattr(layer, name, projection)
    model = torch.nn.Module()
    model.layer = layer
    monkeypatch.setattr(loader_utils, "maybe_retie_word_embeddings", lambda *args: None)
    config = SimpleNamespace(dtype=torch.bfloat16, quantization=None)

    # Use the actual loader lifecycle. Profiling without attention metadata
    # must not leave packing deferred until the first graph capture.
    loader_utils.process_weights_after_loading(model, config, torch.device("cpu"))
    expected = torch.tensor(
        [[0, 4, 10, 14, 20, 24], [1, 5, 11, 15, 21, 25], [2, 6, 12, 16, 22, 26], [3, 7, 13, 17, 23, 27]],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(layer._merged_conv_weight, expected)
    assert layer._merged_conv_weight.is_contiguous()
    assert not layer._merged_conv_weight.requires_grad
    assert "_merged_conv_weight" in dict(layer.named_buffers())
    assert "_merged_conv_weight" not in layer.state_dict()
    assert layer.q_conv1d.weight.dtype == torch.float32

    with torch.no_grad():
        layer.q_conv1d.weight.add_(100)
    loader_utils.process_weights_after_loading(model, config, torch.device("cpu"))
    expected[:, :2] += 100
    torch.testing.assert_close(layer._merged_conv_weight, expected)
