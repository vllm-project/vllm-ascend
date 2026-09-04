from types import SimpleNamespace
from zlib import adler32

import numpy as np
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    prepare_sparse_kv_offload_metadata,
)
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


def _make_speculator():
    return SimpleNamespace(
        target_input_buffers=SimpleNamespace(
            offload_req_ids=torch.tensor([11, 22, 0, 0], dtype=torch.int64),
        ),
    )


def test_stages_sparse_kv_offload_request_metadata_in_input_buffers(monkeypatch):
    def copy_to_cpu(value, out=None, device=None):
        value = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        if out is None:
            return value.to(device=device)
        return out.copy_(value)

    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager.async_copy_to_gpu",
        copy_to_cpu,
    )
    input_buffers = SimpleNamespace(
        offload_req_ids=torch.zeros(4, dtype=torch.int64),
    )
    input_batch = SimpleNamespace(
        req_ids=["request-a", "request-b"],
        num_reqs=2,
        num_reqs_after_padding=4,
        offload_req_ids=None,
    )

    result = prepare_sparse_kv_offload_metadata(input_batch, input_buffers)

    assert result is input_batch
    assert result.offload_req_ids.tolist() == [
        adler32(b"request-a"),
        adler32(b"request-b"),
        0,
        0,
    ]


def test_populates_only_sparse_kv_offload_mtp_draft_request_metadata():
    speculator = _make_speculator()
    token_to_req = torch.tensor([0, 1], dtype=torch.int32)
    metadata = SimpleNamespace(req_ids_tensor=None, token_to_req=token_to_req)

    AscendAutoRegressiveSpeculator._populate_sparse_kv_offload_draft_metadata(
        speculator,
        {"layer": metadata},
        num_reqs_padded=2,
    )

    assert metadata.req_ids_tensor.tolist() == [11, 22]
    assert metadata.token_to_req is token_to_req
