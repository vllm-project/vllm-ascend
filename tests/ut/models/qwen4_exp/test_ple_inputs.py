# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.models.qwen4_exp.input_adapter import (
    Qwen4ExpInputContext,
    Qwen4ExpPLEInputAdapter,
    build_qwen4_exp_input_adapter,
    pad_ple_graph_inputs,
)


def _make_adapter() -> Qwen4ExpPLEInputAdapter:
    return Qwen4ExpPLEInputAdapter(
        SimpleNamespace(ngram_size=3, eos_token_id=999),
        max_num_reqs=2,
        device=torch.device("cpu"),
    )


def _make_context(*, is_dummy: bool = False) -> Qwen4ExpInputContext:
    return Qwen4ExpInputContext(
        num_reqs=1,
        num_reqs_padded=1,
        is_dummy=is_dummy,
        query_start_loc=torch.tensor([0, 1, 1], dtype=torch.int32),
        num_computed_tokens=np.array([3], dtype=np.int32),
        token_ids=np.array([[7, 8, 9, 0]], dtype=np.int32),
    )


def test_graph_padding_uses_fixed_buffers() -> None:
    context_buffer = torch.empty((2, 2), dtype=torch.int32)
    query_buffer = torch.empty(3, dtype=torch.int32)
    context, query = pad_ple_graph_inputs(
        context_buffer,
        query_buffer,
        torch.tensor([[8, 9]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        1,
        999,
    )
    assert context.tolist() == [[8, 9], [999, 999]]
    assert query.tolist() == [0, 1, 1]


def test_factory_ignores_models_without_ple() -> None:
    assert (
        build_qwen4_exp_input_adapter(
            SimpleNamespace(),
            max_num_reqs=2,
            device=torch.device("cpu"),
        )
        is None
    )


def test_ngram_context_keeps_address_and_refreshes_values() -> None:
    adapter = _make_adapter()

    dummy_kwargs: dict = {}
    adapter.prepare(dummy_kwargs, _make_context(is_dummy=True))
    dummy_context = dummy_kwargs["ngram_context"]
    stable_context_ptr = dummy_context.data_ptr()
    stable_query_ptr = dummy_kwargs["query_start_loc"].data_ptr()
    assert dummy_context.tolist() == [[999, 999], [999, 999]]
    assert dummy_kwargs["query_start_loc"].tolist() == [0, 1, 1]

    real_kwargs: dict = {}
    adapter.prepare(real_kwargs, _make_context())
    assert real_kwargs["ngram_context"].data_ptr() == stable_context_ptr
    assert real_kwargs["query_start_loc"].data_ptr() == stable_query_ptr
    assert real_kwargs["ngram_context"].tolist() == [
        [8, 9],
        [999, 999],
    ]

    external_kwargs = {
        "ngram_context": torch.tensor([[33, 44]], dtype=torch.int32),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
    }
    adapter.prepare(external_kwargs, _make_context())
    assert external_kwargs["ngram_context"].data_ptr() == stable_context_ptr
    assert external_kwargs["query_start_loc"].data_ptr() == stable_query_ptr
    assert external_kwargs["ngram_context"].tolist() == [
        [33, 44],
        [999, 999],
    ]
