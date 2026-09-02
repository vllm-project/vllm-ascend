# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.attention_cp import (
    AscendAttentionDCPImpl,
    AscendAttentionDCPMetadata,
    AscendAttentionDCPMetadataBuilder,
    AscendMetadataForDecode,
)
from vllm_ascend.attention.context_parallel.common_cp import (
    _update_out_and_lse,
)


def test_gqa_dcp_extends_v1_backend_without_polluting_base_metadata() -> None:
    assert issubclass(AscendAttentionDCPImpl, AscendAttentionBackendImpl)
    assert issubclass(
        AscendAttentionDCPMetadataBuilder,
        AscendAttentionMetadataBuilder,
    )
    assert AscendAttentionDCPMetadataBuilder.metadata_cls is (AscendAttentionDCPMetadata)
    assert not hasattr(AscendMetadata(), "decode_meta")
    assert not hasattr(AscendMetadata(), "prefill")


def test_dcp_chunked_request_mask_marks_nonempty_contexts() -> None:
    local_context_lens = torch.tensor(
        [
            [0, 0],
            [4, 0],
            [0, 7],
        ],
        dtype=torch.int32,
    )

    assert AscendAttentionDCPMetadataBuilder._get_chunked_req_mask(local_context_lens) == [
        False,
        True,
        True,
    ]


def test_dcp_decode_metadata_keeps_rank_local_context_lengths() -> None:
    local_context_lens = np.array([[11, 12], [21, 22]], dtype=np.int32)
    block_tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    metadata = AscendMetadataForDecode(
        num_computed_tokens_of_dcp=local_context_lens,
        block_tables=block_tables,
    )

    np.testing.assert_array_equal(metadata.num_computed_tokens_of_dcp[:, 1], [12, 22])
    assert metadata.block_tables is block_tables


def test_dcp_chunked_prefill_unpacks_standardized_kv_cache() -> None:
    impl = object.__new__(AscendAttentionDCPImpl)
    impl.dcp_rank = 0
    impl.dcp_size = 1
    impl.num_heads = 2
    impl.head_size = 4
    standardized_cache = torch.empty((2, 1, 1, 4))
    cache_pair = (torch.empty((1, 1, 4)), torch.empty((1, 1, 4)))
    chunked_context = SimpleNamespace(
        local_context_lens_allranks=torch.zeros((1, 1), dtype=torch.int32),
        local_total_toks=0,
    )
    metadata = MagicMock(spec=AscendAttentionDCPMetadata)
    metadata.prefill = SimpleNamespace(chunked_context=chunked_context)

    with (
        patch.object(impl, "_unpack_kv_cache", return_value=cache_pair) as unpack_cache,
        patch.object(impl, "_load_kv_for_chunk", return_value=cache_pair) as load_chunk,
    ):
        output, lse = impl._compute_prefill_context(torch.empty((1, 2, 4)), standardized_cache, metadata)

    unpack_cache.assert_called_once_with(standardized_cache)
    assert load_chunk.call_args.args[1] is cache_pair
    assert output.shape == (1, 2, 4)
    assert lse.shape == (1, 2, 1)


def test_dcp_partial_attention_merge_matches_weighted_reference() -> None:
    outputs = torch.tensor(
        [
            [[[[1.0, 3.0]]]],
            [[[[5.0, 7.0]]]],
        ]
    ).reshape(2, 1, 1, 2)
    lse = torch.tensor([0.0, np.log(3.0)], dtype=torch.float32).reshape(2, 1, 1, 1)

    output, merged_lse = _update_out_and_lse(outputs, lse)

    torch.testing.assert_close(output, torch.tensor([[[4.0, 6.0]]]))
    torch.testing.assert_close(merged_lse, torch.tensor([[[np.log(4.0)]]], dtype=torch.float32))
