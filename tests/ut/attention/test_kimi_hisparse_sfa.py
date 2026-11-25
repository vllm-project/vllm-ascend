# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.v1.hisparse.backend")

from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl  # noqa: E402


def test_cold_mixed_prefill_masks_padded_topk():
    block_size, topk = 128, 2048
    num_input_tokens = 5
    query_start_loc = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    physical_decode = torch.full((2, topk), -1, dtype=torch.int32)
    physical_decode[:, 0] = torch.tensor([5, 6], dtype=torch.int32)
    slab = torch.empty((2, block_size, 656), dtype=torch.float8_e4m3fn)
    staged_table = torch.tensor([[1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        num_decode_tokens=2,
        num_decodes=2,
        num_prefills=1,
        num_reqs=3,
        block_size=block_size,
        hisparse_for_capture=False,
        query_start_loc=query_start_loc,
        prefill_valid_token_mask=torch.arange(num_input_tokens, dtype=torch.int32)[:, None] < query_start_loc[-1],
        # The metadata builder maps padding to the last request.
        req_id_per_token=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
        prefill=SimpleNamespace(host_staging_plan=SimpleNamespace(block_size=block_size)),
    )

    def stage_prefill_rows(layer_index, host, metadata):
        reqs = metadata.req_id_per_token[metadata.num_decode_tokens :] - metadata.num_decodes
        return slab, staged_table, reqs

    def read_physical_indices(query, query_rope, packed_cache, physical_topk, **kwargs):
        # Treat the first selected address as the attention result so the test
        # can check the actual address translation without launching QSFA.
        first = physical_topk[:, 0].to(query.dtype)
        return first[:, None, None].expand_as(query)

    impl = AscendHiSparseImpl.__new__(AscendHiSparseImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.index_group_index = 0
    impl.index_group = SimpleNamespace(
        set_logical_topk_ready=lambda layer_index: None,
        physical_kv_cache=lambda layer_index: slab,
        stage_prefill_rows=stage_prefill_rows,
    )
    impl.hisparse_cache = SimpleNamespace(
        source_block_table=torch.zeros((3, 1), dtype=torch.int32),
        swap_in=lambda *args, **kwargs: physical_decode,
        runtime=SimpleNamespace(
            host_cache=slab.view(-1, 656),
            cache_ops=SimpleNamespace(
                attention_lengths=(torch.arange(3, dtype=torch.int32), torch.tensor([256], dtype=torch.int32)),
                attention_capacity=256,
            ),
        ),
    )
    impl._attention_physical = read_physical_indices

    query = torch.empty((num_input_tokens, 12, 512), dtype=torch.bfloat16)
    query_tail = torch.empty((num_input_tokens, 12, 64), dtype=torch.bfloat16)
    logical = torch.full((num_input_tokens, 1, topk), -1, dtype=torch.int32)
    logical[2:, 0, 0] = torch.tensor([1, 2, 127], dtype=torch.int32)
    result = impl._execute_sparse_flash_attention_process(
        query, query_tail, (), logical, metadata, metadata.query_start_loc[1:], torch.ones(3, dtype=torch.int32)
    )

    torch.testing.assert_close(result[:, 0, 0], torch.tensor([5, 6, 129, 130, -1], dtype=torch.bfloat16))
