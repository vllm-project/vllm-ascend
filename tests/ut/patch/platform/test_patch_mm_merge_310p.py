# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch
import vllm.model_executor.models.utils as model_utils

_original_merge_multimodal_embeddings = model_utils._merge_multimodal_embeddings
patch_mm_merge_310p = importlib.import_module("vllm_ascend.patch.platform.patch_mm_merge_310p")
_merge_multimodal_embeddings_310p = patch_mm_merge_310p._merge_multimodal_embeddings_310p

# Importing the patch module performs the production monkey patch. Restore the
# original here so collecting this CPU unit test does not affect unrelated tests.
model_utils._merge_multimodal_embeddings = _original_merge_multimodal_embeddings


def test_merge_multimodal_embeddings_310p():
    inputs_embeds = torch.zeros(4, 2)
    multimodal_embeddings = [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
    ]
    is_multimodal = torch.tensor([False, True, False, True])

    result = _merge_multimodal_embeddings_310p(
        inputs_embeds,
        multimodal_embeddings,
        is_multimodal,
    )

    assert result is inputs_embeds
    torch.testing.assert_close(
        result,
        torch.tensor([[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 4.0]]),
    )


def test_merge_multimodal_embeddings_310p_moves_source_to_target_device(
    monkeypatch,
):
    class FakeEmbeddings:
        def __init__(self):
            self.to_kwargs = None

        def to(self, **kwargs):
            self.to_kwargs = kwargs
            return self

    class FakeInputsEmbeds:
        dtype = torch.float16
        device = torch.device("cpu")

        def index_copy_(self, dim, index, source):
            assert dim == 0
            assert index.tolist() == [1]
            assert source is flattened

    flattened = FakeEmbeddings()
    monkeypatch.setattr(
        patch_mm_merge_310p,
        "_flatten_embeddings",
        lambda _embeddings: flattened,
    )

    inputs_embeds = FakeInputsEmbeds()
    result = _merge_multimodal_embeddings_310p(
        inputs_embeds,
        [torch.ones(1, 2)],
        torch.tensor([False, True]),
    )

    assert result is inputs_embeds
    assert flattened.to_kwargs == {
        "device": inputs_embeds.device,
        "dtype": inputs_embeds.dtype,
    }


def test_merge_multimodal_embeddings_310p_rejects_token_count_mismatch():
    inputs_embeds = torch.zeros(3, 2)
    is_multimodal = torch.tensor([False, True, False])

    with pytest.raises(
        ValueError,
        match=r"Attempted to assign 2 = 2 multimodal tokens to 1 placeholders",
    ):
        _merge_multimodal_embeddings_310p(
            inputs_embeds,
            torch.ones(2, 2),
            is_multimodal,
        )


def test_merge_multimodal_embeddings_310p_empty_input_is_unchanged():
    inputs_embeds = torch.randn(3, 2)

    result = _merge_multimodal_embeddings_310p(
        inputs_embeds,
        [],
        torch.zeros(3, dtype=torch.bool),
    )

    assert result is inputs_embeds
