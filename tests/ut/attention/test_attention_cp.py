# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
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


@pytest.mark.parametrize("dcp_size", [1, 2])
def test_gqa_dcp_extends_v1_backend_without_polluting_base_metadata(dcp_size) -> None:
    assert issubclass(AscendAttentionDCPImpl, AscendAttentionBackendImpl)
    dcp_group = SimpleNamespace(world_size=dcp_size, rank_in_group=0, device_group=Mock())
    with (
        patch.object(AscendAttentionBackendImpl, "__init__", return_value=None),
        patch("vllm_ascend.attention.context_parallel.common_cp.get_dcp_group", return_value=dcp_group),
    ):
        impl = AscendAttentionDCPImpl()
    assert impl.can_return_lse_for_decode
    assert impl.need_to_return_lse_for_decode is (dcp_size > 1)
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


@pytest.mark.parametrize(
    "is_consumer,is_producer,recompute", [(True, False, True), (True, False, False), (False, True, True)]
)
@pytest.mark.parametrize("query_lens", [[1, 1], [3, 3], [3, 5]])
def test_dcp_split_uses_builder_config_without_current_context(is_consumer, is_producer, recompute, query_lens):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=is_consumer, is_kv_producer=is_producer),
    )
    with (
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.DCPMetadataBuilderMixin.__init__", return_value=None
        ),
        patch("vllm_ascend.attention.context_parallel.attention_cp.enable_dcp", return_value=True) as dcp,
    ):
        builder = AscendAttentionDCPMetadataBuilder()
    dcp.assert_called_once_with()
    builder.vllm_config = config
    builder.decode_threshold = 3
    query_start_loc = torch.tensor([0, query_lens[0], sum(query_lens)], dtype=torch.int32)
    common = SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=max(query_lens),
        num_reqs=2,
        num_actual_tokens=sum(query_lens),
        query_start_loc_cpu=query_start_loc,
        is_prefilling=torch.ones(2, dtype=torch.bool),
    )
    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=None),
        patch(
            "vllm_ascend.utils.get_ascend_config",
            return_value=SimpleNamespace(scheduler_config=SimpleNamespace(recompute_scheduler_enable=recompute)),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.attention_cp.enable_dcp",
            side_effect=AssertionError("use cached DCP state"),
        ),
    ):
        actual = builder._split_decodes_and_prefills(common)
    num_decodes = sum(q <= 3 for q in query_lens) if is_consumer and not is_producer and recompute else 0
    num_decode_tokens = sum(query_lens[:num_decodes])
    assert actual == (num_decodes, 2 - num_decodes, num_decode_tokens, sum(query_lens) - num_decode_tokens)
