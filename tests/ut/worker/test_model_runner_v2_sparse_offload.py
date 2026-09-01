from types import SimpleNamespace
from zlib import adler32

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.input_batch import prepare_sparse_kv_offload_metadata
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


def _make_speculator():
    return SimpleNamespace(
        target_input_buffers=SimpleNamespace(
            offload_req_ids=torch.tensor([11, 22, 0, 0], dtype=torch.int64),
        ),
        _offload_draft_token_to_req=torch.arange(8, dtype=torch.int32),
        _offload_draft_metadata_logged=False,
    )


def test_stages_sparse_kv_offload_metadata_in_input_buffers(monkeypatch):
    def copy_to_cpu(value, out=None, device=None):
        value = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        if out is None:
            return value.to(device=device)
        return out.copy_(value)

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.async_copy_to_gpu",
        copy_to_cpu,
    )
    input_buffers = SimpleNamespace(
        offload_req_ids=torch.zeros(4, dtype=torch.int64),
        offload_token_to_req=torch.zeros(8, dtype=torch.int32),
    )
    input_batch = SimpleNamespace(
        req_ids=["request-a", "request-b"],
        num_reqs=2,
        num_reqs_after_padding=4,
        num_tokens=3,
        num_tokens_after_padding=5,
        query_start_loc_np=np.array([0, 2, 3], dtype=np.int32),
        offload_req_ids=None,
        offload_token_to_req=None,
    )

    result = prepare_sparse_kv_offload_metadata(input_batch, input_buffers)

    assert result is input_batch
    assert result.offload_req_ids.tolist() == [
        adler32(b"request-a"),
        adler32(b"request-b"),
        0,
        0,
    ]
    assert result.offload_token_to_req.tolist() == [0, 0, 1, 0, 0]


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
