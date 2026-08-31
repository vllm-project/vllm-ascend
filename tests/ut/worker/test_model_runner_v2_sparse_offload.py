from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


def _make_speculator():
    return SimpleNamespace(
        model_state=SimpleNamespace(
            _offload_req_ids_tensor=torch.tensor([11, 22, 0, 0], dtype=torch.int64),
        ),
        _offload_draft_token_to_req=torch.arange(8, dtype=torch.int32),
        _offload_draft_metadata_logged=False,
    )


def test_populates_sparse_kv_offload_mtp_draft_metadata():
    speculator = _make_speculator()
    metadata = SimpleNamespace(req_ids_tensor=None, token_to_req=None)

    AscendAutoRegressiveSpeculator._populate_sparse_kv_offload_draft_metadata(
        speculator,
        {"layer": metadata},
        num_reqs_padded=2,
        num_tokens_padded=2,
        num_query_per_req=1,
    )

    assert metadata.req_ids_tensor.tolist() == [11, 22]
    assert metadata.token_to_req.tolist() == [0, 1]


def test_rejects_non_unit_sparse_kv_offload_mtp_draft_width():
    speculator = _make_speculator()

    with pytest.raises(RuntimeError, match="requires one token per request"):
        AscendAutoRegressiveSpeculator._populate_sparse_kv_offload_draft_metadata(
            speculator,
            {"layer": SimpleNamespace()},
            num_reqs_padded=2,
            num_tokens_padded=4,
            num_query_per_req=2,
        )
