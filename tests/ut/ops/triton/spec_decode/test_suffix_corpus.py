# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.ops.triton.spec_decode import suffix_corpus as ops


def test_ingestion_launcher_uses_tensor_stride_and_document_count(monkeypatch):
    kernel = MagicMock()
    monkeypatch.setattr(ops, "_suffix_ingest_kernel", kernel)
    token_ids = SimpleNamespace(shape=(4, 2048), stride=lambda axis: 2056)
    args = (token_ids, 2048, object(), 2, *[object() for _ in range(8)])
    options = dict(HASH_MUL=-7046029254386353131, NUM_BUCKETS=64, SLOTS=8, BLOCK=1024, num_warps=4)

    ops.launch_suffix_ingest(*args, **options)

    kernel.__getitem__.assert_called_once_with((2, 2))
    kernel.__getitem__.return_value.assert_called_once_with(token_ids, 2056, *args[2:], **options)
