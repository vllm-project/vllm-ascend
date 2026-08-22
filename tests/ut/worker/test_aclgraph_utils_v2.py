from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.worker.v2.aclgraph_utils import (
    ModelWithContext,
    model_capture_wrapper,
)


def test_model_capture_wrapper_forwards_multimodal_embed_input_ids():
    original_model = torch.nn.Module()
    embedded_inputs = torch.randn(2, 4)
    original_model.embed_input_ids = MagicMock(return_value=embedded_inputs)
    speculator = SimpleNamespace(model=original_model)
    input_ids = torch.tensor([1, 2])
    multimodal_embeddings = [torch.randn(1, 4)]
    is_multimodal = torch.tensor([True, False])

    with model_capture_wrapper(speculator, is_draft_model_prefill=True):
        assert isinstance(speculator.model, ModelWithContext)
        result = speculator.model.embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    assert result is embedded_inputs
    original_model.embed_input_ids.assert_called_once_with(
        input_ids,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )
    assert speculator.model is original_model
