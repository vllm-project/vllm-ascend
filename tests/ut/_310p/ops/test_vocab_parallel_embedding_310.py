from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend._310p.ops.vocab_parallel_embedding as embedding_310
from vllm_ascend._310p.ops.vocab_parallel_embedding import (
    AscendUnquantizedEmbeddingMethod310,
    AscendVocabParallelEmbedding310,
)


def test_private_gather_elements_embedding_is_exact_for_valid_rows():
    layer = object.__new__(AscendVocabParallelEmbedding310)
    torch.nn.Module.__init__(layer)
    layer.tp_size = 1
    layer.embedding_dim = 4
    layer.forward_type = None
    layer.quant_method = AscendUnquantizedEmbeddingMethod310()
    layer.weight = torch.nn.Parameter(
        torch.arange(80, dtype=torch.float32).view(20, 4),
        requires_grad=False,
    )
    input_ids = torch.tensor([7, 1, 19, 7], dtype=torch.int64)
    expected = torch.nn.functional.embedding(input_ids, layer.weight)

    with patch.object(
        torch.ops.vllm,
        "maybe_pad_and_reduce",
        side_effect=lambda output: output,
    ):
        actual = layer.embedding_gather_elements_310(input_ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("forward_type", "quant_method", "message"),
    [
        ("embed_tp", AscendUnquantizedEmbeddingMethod310(), "embedding TP"),
        (None, object(), "unquantized embedding"),
    ],
)
def test_private_embedding_rejects_unsupported_public_contracts(
    forward_type,
    quant_method,
    message,
):
    layer = object.__new__(AscendVocabParallelEmbedding310)
    torch.nn.Module.__init__(layer)
    layer.tp_size = 1
    layer.embedding_dim = 4
    layer.forward_type = forward_type
    layer.quant_method = quant_method
    layer.weight = torch.nn.Parameter(
        torch.arange(80, dtype=torch.float32).view(20, 4),
        requires_grad=False,
    )

    with pytest.raises(NotImplementedError, match=message):
        layer.embedding_gather_elements_310(
            torch.tensor([1, 2], dtype=torch.int64),
        )


@pytest.mark.parametrize(
    ("hybrid", "is_draft", "runtime_mode", "expected"),
    [
        (True, True, CUDAGraphMode.PIECEWISE, True),
        (True, True, CUDAGraphMode.FULL, True),
        (True, True, CUDAGraphMode.NONE, True),
        (True, False, CUDAGraphMode.PIECEWISE, False),
        (False, True, CUDAGraphMode.PIECEWISE, False),
    ],
)
def test_private_gather_elements_route_is_stable_for_hybrid_draft(
    hybrid,
    is_draft,
    runtime_mode,
    expected,
):
    context = SimpleNamespace(
        is_draft_model=is_draft,
        cudagraph_runtime_mode=runtime_mode,
        vllm_config=object(),
    )
    with patch.object(
        embedding_310,
        "is_310p_dflash_full_and_piecewise",
        return_value=hybrid,
    ):
        assert (
            embedding_310._uses_private_draft_embedding_310(context)
            is expected
        )


def test_draft_piecewise_forward_uses_private_gather_elements():
    layer = object.__new__(AscendVocabParallelEmbedding310)
    torch.nn.Module.__init__(layer)
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    private_forward = MagicMock(return_value=expected)
    context = SimpleNamespace(
        is_draft_model=True,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        vllm_config=object(),
    )

    with (
        patch.object(
            embedding_310,
            "is_forward_context_available",
            return_value=True,
        ),
        patch.object(
            embedding_310,
            "get_forward_context",
            return_value=context,
        ),
        patch.object(
            embedding_310,
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch.object(
            layer,
            "embedding_gather_elements_310",
            private_forward,
        ),
        patch.object(
            embedding_310.AscendVocabParallelEmbedding,
            "forward",
            return_value=torch.full_like(expected, -1),
        ) as public_forward,
    ):
        actual = layer.forward(input_ids)

    assert actual is expected
    private_forward.assert_called_once_with(input_ids)
    public_forward.assert_not_called()


def test_non_draft_piecewise_forward_keeps_public_embedding():
    layer = object.__new__(AscendVocabParallelEmbedding310)
    torch.nn.Module.__init__(layer)
    input_ids = torch.tensor([1], dtype=torch.int32)
    expected = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    context = SimpleNamespace(
        is_draft_model=False,
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        vllm_config=object(),
    )

    with (
        patch.object(
            embedding_310,
            "is_forward_context_available",
            return_value=True,
        ),
        patch.object(
            embedding_310,
            "get_forward_context",
            return_value=context,
        ),
        patch.object(
            embedding_310,
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch.object(
            layer,
            "embedding_gather_elements_310",
        ) as private_forward,
        patch.object(
            embedding_310.AscendVocabParallelEmbedding,
            "forward",
            return_value=expected,
        ) as public_forward,
    ):
        actual = layer.forward(input_ids)

    assert actual is expected
    private_forward.assert_not_called()
    public_forward.assert_called_once_with(input_ids)
