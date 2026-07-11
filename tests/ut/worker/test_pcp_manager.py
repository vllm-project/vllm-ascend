# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, split_decodes_and_prefills
from vllm_ascend.worker.pcp_attention_backend import (
    PCPBackendMetadata,
    PCPMetadataBuildContext,
)
from vllm_ascend.worker.pcp_utils import PCPManager


def _make_vllm_config(
    *,
    model_type="qwen2",
    use_mla=False,
    num_speculative_tokens=0,
    max_model_len=128,
    max_num_batched_tokens=10000,
    max_num_seqs=1000,
    cp_kv_cache_interleave_size=64,
):
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.use_mla = use_mla
    vllm_config.model_config.hf_config.model_type = model_type
    vllm_config.model_config.max_model_len = max_model_len
    vllm_config.parallel_config.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size
    vllm_config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    vllm_config.speculative_config.num_speculative_tokens = num_speculative_tokens
    vllm_config.kv_transfer_config = None
    return vllm_config


def _make_pcp_manager(
    *,
    pcp_world_size=2,
    pcp_rank=0,
    dcp_world_size=1,
    dcp_rank=0,
    model_type="qwen2",
    use_mla=False,
    num_speculative_tokens=0,
    max_model_len=128,
    max_buffer_num_tokens=10000,
    max_num_reqs=1000,
    cp_kv_cache_interleave_size=64,
):
    return PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        max_buffer_num_tokens=max_buffer_num_tokens,
        max_num_reqs=max_num_reqs,
        device="cpu",
        vllm_config=_make_vllm_config(
            model_type=model_type,
            use_mla=use_mla,
            num_speculative_tokens=num_speculative_tokens,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_buffer_num_tokens,
            max_num_seqs=max_num_reqs,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        ),
        use_async_scheduling=False,
        pin_memory=False,
    )


def test_generate_pcp_metadata_uses_selected_attention_backend():
    class FakePCPAttentionBackend:
        name = "fake"

        def __init__(self) -> None:
            self.ctx: PCPMetadataBuildContext | None = None

        def build_metadata(self, ctx: PCPMetadataBuildContext) -> PCPBackendMetadata:
            self.ctx = ctx
            return PCPBackendMetadata(
                q_head_idx_tensor=torch.tensor([42], dtype=torch.int32),
                q_tail_idx_tensor=torch.tensor([43], dtype=torch.int32),
                q_full_idx=torch.tensor([0, 1], dtype=torch.int32),
                attn_chunk_seqlens=torch.tensor([4], dtype=torch.int32),
                extra_long_seq_kwargs={"backend": [1]},
            )

        def apply_metadata(self, long_seq_metadata, metadata: PCPBackendMetadata) -> None:
            long_seq_metadata.fake_backend_name = self.name
            long_seq_metadata.fake_backend_q_head = metadata.q_head_idx_tensor

    pcp_manager = _make_pcp_manager(pcp_world_size=2, pcp_rank=1)
    num_scheduled_tokens = np.array([8], dtype=np.int32)
    num_computed_tokens = np.array([0], dtype=np.int32)
    num_prompt_tokens = np.array([8], dtype=np.int32)
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=1,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )
    pcp_manager.update_tokens_for_pcp(
        num_scheduled_tokens,
        np.arange(10000, dtype=np.int32),
    )

    input_batch = MagicMock()
    input_batch.num_reqs = 1
    input_batch.num_computed_tokens_cpu = num_computed_tokens
    input_batch.num_prompt_tokens = torch.tensor(num_prompt_tokens)
    input_batch.num_tokens = torch.tensor(num_scheduled_tokens)

    fake_backend = FakePCPAttentionBackend()
    with patch("vllm_ascend.worker.pcp_utils.select_pcp_attention_backend", return_value=fake_backend):
        metadata, _ = pcp_manager.generate_pcp_metadata(
            int(num_scheduled_tokens.sum()),
            torch.tensor(num_scheduled_tokens, dtype=torch.int32),
            input_batch,
            num_scheduled_tokens,
            torch.zeros((1, 1), dtype=torch.int32),
            num_reqs_padded=1,
            num_reqs=1,
        )

    assert metadata is not None
    assert metadata.fake_backend_name == "fake"
    assert metadata.fake_backend_q_head.tolist() == [42]
    ctx = fake_backend.ctx
    assert ctx is not None
    assert ctx.num_decode_reqs == 0
    assert ctx.pcp_world_size == 2
    assert ctx.pcp_world_rank == 1
    assert ctx.use_mla is False
    assert pcp_manager.q_head_idx_tensor.tolist() == [42]
    assert pcp_manager.extra_long_seq_kwargs == {"backend": [1]}


@pytest.mark.parametrize(
    "pcp_size, dcp_size, num_reqs, query_lens, num_decodes, use_mla, total_tokens, expect_not_none",
    [
        (1, 1, 5, [10, 20, 30, 40, 50], 2, False, 100, False),
        (1, 2, 3, [20, 30, 40], 1, False, 50, True),
        (2, 1, 4, [5, 10, 40, 60], 2, False, 100, True),
        (2, 1, 4, [5, 10, 40, 60], 2, True, 100, True),
        (2, 1, 3, [5, 10, 15], 3, False, 50, True),
        (2, 1, 3, [40, 50, 60], 0, False, 150, True),
    ],
)
def test_generate_pcp_metadata_basic(
    pcp_size, dcp_size, num_reqs, query_lens, num_decodes, use_mla, total_tokens, expect_not_none
):
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.use_mla = use_mla
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 64
    vllm_config.speculative_config.num_speculative_tokens = 0

    pcp_manager = PCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=0,
        dcp_world_size=dcp_size,
        dcp_rank=0,
        max_buffer_num_tokens=10000,
        max_num_reqs=1000,
        device="cpu",
        vllm_config=vllm_config,
        use_async_scheduling=False,
        pin_memory=False,
    )
    input_batch = MagicMock()
    input_batch.num_reqs = num_reqs

    num_computed_tokens = []
    num_prompt_tokens = []
    num_tokens = []

    for i in range(num_reqs):
        if i < num_decodes:
            num_computed_tokens.append(query_lens[i])
            num_prompt_tokens.append(query_lens[i] // 2)
            num_tokens.append(query_lens[i])
        else:
            num_computed_tokens.append(0)
            num_prompt_tokens.append(query_lens[i])
            num_tokens.append(query_lens[i])

    input_batch.num_computed_tokens_cpu = np.array(num_computed_tokens)
    input_batch.num_prompt_tokens = torch.tensor(num_prompt_tokens)
    input_batch.num_tokens = torch.tensor(num_tokens)
    num_scheduled_tokens = np.array(query_lens) - input_batch.num_computed_tokens_cpu

    query_lens = torch.tensor(query_lens)
    result, _ = pcp_manager.generate_pcp_metadata(
        total_tokens,
        query_lens,
        input_batch,
        num_scheduled_tokens,
        torch.tensor([]),
        num_reqs_padded=num_reqs,
        num_reqs=num_reqs,
    )

    if not expect_not_none:
        assert result is None, f"Expected to return None, but got {type(result)}"
    else:
        assert result is not None, "Expected to return a metadata object, but got None."

        assert hasattr(result, "num_actual_tokens_pcp_padded")
        assert hasattr(result, "num_computed_tokens_of_pcp_dcp")

        if pcp_size > 1:
            assert hasattr(result, "pcp_allgather_restore_idx")

            has_prefill_requests = (num_reqs - num_decodes) > 0
            if has_prefill_requests:
                assert hasattr(result, "q_head_idx_tensor")
                assert hasattr(result, "q_tail_idx_tensor")
                assert hasattr(result, "q_full_idx")
                assert hasattr(result, "kv_with_q_head_nomask_idx_tensor")
                assert hasattr(result, "kv_with_q_head_mask_idx_tensor")
                assert hasattr(result, "kv_with_q_tail_nomask_idx_tensor")
                assert hasattr(result, "kv_with_q_tail_mask_idx_tensor")
                assert hasattr(result, "kv_tail_proj_idx_tensor")
                assert hasattr(result, "kv_with_q_head_attn_idx_in_tail_tensor")
                assert hasattr(result, "kv_with_q_tail_attn_idx_in_tail_tensor")
                assert hasattr(result, "attn_mask_seqlens")
                assert hasattr(result, "head_attn_nomask_seqlens")
                assert hasattr(result, "tail_attn_nomask_seqlens")
                assert hasattr(result, "head_actual_seq_lengths_kv")
                assert hasattr(result, "tail_actual_seq_lengths_kv")


def test_generate_pcp_metadata_combines_pcp_and_dcp_rank_layout():
    pcp_manager = _make_pcp_manager(
        pcp_world_size=2,
        dcp_world_size=2,
        max_model_len=512,
    )
    pcp_manager.speculative_config = None
    num_scheduled_tokens = np.array([1, 1, 8, 12], dtype=np.int32)
    num_computed_tokens = np.array([255, 128, 0, 0], dtype=np.int32)
    num_prompt_tokens = np.array([255, 128, 8, 12], dtype=np.int32)
    num_reqs = len(num_scheduled_tokens)

    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=num_reqs,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )
    pcp_tokens, _ = pcp_manager.update_tokens_for_pcp(
        num_scheduled_tokens,
        np.arange(10000, dtype=np.int32),
    )

    input_batch = MagicMock()
    input_batch.num_reqs = num_reqs
    input_batch.num_computed_tokens_cpu = num_computed_tokens
    input_batch.num_prompt_tokens = torch.tensor(num_prompt_tokens)
    input_batch.num_tokens = torch.tensor(num_scheduled_tokens)

    metadata, _ = pcp_manager.generate_pcp_metadata(
        int(num_scheduled_tokens.sum()),
        torch.tensor(num_scheduled_tokens, dtype=torch.int32),
        input_batch,
        num_scheduled_tokens,
        torch.zeros((num_reqs, 1), dtype=torch.int32),
        num_reqs_padded=num_reqs,
        num_reqs=num_reqs,
    )

    assert pcp_tokens.tolist() == [1, 1, 4, 6]
    assert metadata is not None
    assert metadata.pcp_use_hybrid_attn is False
    assert metadata.num_computed_tokens_of_pcp_dcp.tolist() == [
        [[64, 64], [64, 64]],
        [[64, 64], [1, 0]],
        [[0, 0], [0, 0]],
        [[0, 0], [0, 0]],
    ]
    assert metadata.q_head_idx_tensor is not None
    assert metadata.q_tail_idx_tensor is not None
    assert metadata.pcp_allgather_restore_idx is not None
    assert metadata.dcp_mtp_attn_mask is None


@pytest.mark.parametrize(
    "pcp_size, pcp_rank, query_lens",
    [
        (2, 0, [8]),
        (2, 1, [8]),
        (4, 0, [8, 12]),
        (4, 3, [8, 12]),
    ],
)
def test_generate_pcp_metadata_mla_tail_projection_indices(pcp_size, pcp_rank, query_lens):
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.use_mla = True
    vllm_config.model_config.hf_config.model_type = "deepseek_v2"
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 64
    vllm_config.scheduler_config.max_num_batched_tokens = 10000
    vllm_config.scheduler_config.max_num_seqs = 1000
    vllm_config.speculative_config.num_speculative_tokens = 0

    pcp_manager = PCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=pcp_rank,
        dcp_world_size=1,
        dcp_rank=0,
        max_buffer_num_tokens=10000,
        max_num_reqs=1000,
        device="cpu",
        vllm_config=vllm_config,
        use_async_scheduling=False,
        pin_memory=False,
    )

    num_reqs = len(query_lens)
    num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
    num_computed_tokens = np.zeros(num_reqs, dtype=np.int32)
    num_prompt_tokens = np.array(query_lens, dtype=np.int32)
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs,
        num_computed_tokens,
        num_prompt_tokens,
    )

    input_batch = MagicMock()
    input_batch.num_reqs = num_reqs
    input_batch.num_computed_tokens_cpu = np.zeros(num_reqs, dtype=np.int32)
    input_batch.num_prompt_tokens = torch.tensor(query_lens)
    input_batch.num_tokens = torch.tensor(query_lens)

    result, _ = pcp_manager.generate_pcp_metadata(
        int(num_scheduled_tokens.sum()),
        torch.tensor(query_lens, dtype=torch.int32),
        input_batch,
        num_scheduled_tokens,
        torch.zeros((num_reqs, 1), dtype=torch.int32),
        num_reqs_padded=num_reqs,
        num_reqs=num_reqs,
    )

    assert result is not None
    tail_idx = result.kv_tail_proj_idx_tensor
    full_kv_len = int(num_scheduled_tokens.sum()) * pcp_size
    assert tail_idx.numel() <= full_kv_len
    assert tail_idx.numel() > 0
    assert tail_idx.min().item() >= 0
    assert tail_idx.max().item() < full_kv_len

    expected_tail_idx: list[int] = []
    expected_head_attn_idx_in_tail = []
    expected_tail_attn_idx_in_tail = []
    expected_head_actual_seq_lengths_kv = []
    expected_tail_actual_seq_lengths_kv = []
    kv_req_offset = 0
    q_head_chunk_id = pcp_rank
    q_tail_chunk_id = pcp_size * 2 - 1 - pcp_rank
    for seq_len in query_lens:
        chunk_len = seq_len // 2
        tail_proj_offset = len(expected_tail_idx)
        tail_proj_len = chunk_len * (q_tail_chunk_id + 1)
        expected_tail_idx.extend(list(range(kv_req_offset, kv_req_offset + tail_proj_len)))
        expected_head_attn_idx_in_tail.extend(
            list(range(tail_proj_offset, tail_proj_offset + chunk_len * (q_head_chunk_id + 1)))
        )
        expected_tail_attn_idx_in_tail.extend(list(range(tail_proj_offset, tail_proj_offset + tail_proj_len)))
        expected_head_actual_seq_lengths_kv.append(len(expected_head_attn_idx_in_tail))
        expected_tail_actual_seq_lengths_kv.append(len(expected_tail_attn_idx_in_tail))
        kv_req_offset += seq_len * pcp_size

    assert torch.equal(tail_idx.cpu(), torch.tensor(expected_tail_idx, dtype=tail_idx.dtype))
    head_attn_idx = result.kv_with_q_head_attn_idx_in_tail_tensor
    tail_attn_idx = result.kv_with_q_tail_attn_idx_in_tail_tensor
    assert torch.equal(
        head_attn_idx.cpu(),
        torch.tensor(expected_head_attn_idx_in_tail, dtype=head_attn_idx.dtype),
    )
    assert torch.equal(
        tail_attn_idx.cpu(),
        torch.tensor(expected_tail_attn_idx_in_tail, dtype=tail_attn_idx.dtype),
    )
    assert result.head_actual_seq_lengths_kv == expected_head_actual_seq_lengths_kv
    assert result.tail_actual_seq_lengths_kv == expected_tail_actual_seq_lengths_kv


@pytest.mark.parametrize(
    "tokens, num_reqs, num_computed_tokens, num_prompt_tokens, pcp_size, pcp_rank, expected_pcp_tokens",
    [
        # Case 1: prefill only
        ([8, 12, 16], 3, [0, 0, 0], [8, 12, 16], 4, 0, [2, 4, 4]),
        # # Case 2: mix prefill and decode
        ([8, 4, 12], 3, [8, 4, 0], [8, 0, 12], 4, 0, [2, 2, 4]),
        # # Case 3: request which need to be padded
        ([3, 7, 9], 3, [0, 0, 0], [3, 7, 9], 4, 0, [2, 2, 4]),
        # Case 4: single request
        ([10], 1, [0], [10], 4, 0, [4]),
    ],
)
def test_update_tokens_for_pcp_basic(
    tokens, num_reqs, num_computed_tokens, num_prompt_tokens, pcp_size, pcp_rank, expected_pcp_tokens
):
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 0
    vllm_config.scheduler_config.max_num_seqs = 1000

    pcp_manager = PCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=0,
        dcp_world_size=1,
        dcp_rank=0,
        max_buffer_num_tokens=10000,
        max_num_reqs=1000,
        device="cpu",
        vllm_config=vllm_config,
        use_async_scheduling=False,
        pin_memory=False,
    )
    input_batch = MagicMock()
    input_batch.num_reqs = num_reqs
    input_batch.num_computed_tokens_cpu = np.array(num_computed_tokens, dtype=np.int32)
    input_batch.num_prompt_tokens = np.array(num_prompt_tokens, dtype=np.int32)
    arange_np = np.arange(10000)
    num_scheduled_tokens = np.array(tokens)
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs,
        input_batch.num_computed_tokens_cpu,
        input_batch.num_prompt_tokens,
    )
    pcp_tokens_result, positions_result = pcp_manager.update_tokens_for_pcp(num_scheduled_tokens, arange_np)

    assert np.array_equal(pcp_tokens_result, expected_pcp_tokens), (
        f"Expected pcp_tokens: {expected_pcp_tokens}, got: {pcp_tokens_result}"
    )

    total_pcp_tokens: int = np.sum(pcp_tokens_result)
    assert positions_result.shape == (total_pcp_tokens,), (
        f"Positions shape mismatch. Expected length {total_pcp_tokens}, got {positions_result.shape}"
    )


def test_split_decodes_short_extend_with_default_false():
    """Short extends should be treated as prefills by default."""
    long_seq_metadata = MagicMock()
    long_seq_metadata.query_lens_pcp_full_cpu = torch.tensor([3], dtype=torch.int32)
    long_seq_metadata.max_query_len_pcp_full = 3

    query_start_loc_cpu = torch.tensor([0, 2], dtype=torch.int32)
    common_attn_metadata = AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc_cpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.tensor([173], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=173,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.arange(2, dtype=torch.int32),
        is_prefilling=torch.tensor([True]),
        prefill_context_parallel_metadata=long_seq_metadata,
    )

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=4,
        treat_short_extends_as_decodes=False,
    )

    assert num_decodes == 0
    assert num_prefills == 1
    assert num_decode_tokens == 0
    assert num_prefill_tokens == 2


# yapf: disable
@pytest.mark.parametrize(
    "seq_lens, pcp_world_size, dcp_world_size, cp_kv_cache_interleave_size, target",
    [
        # without pcp and dcp
        (torch.tensor([1, 2, 128, 129]), 1, 1, 1,
        torch.tensor([[[1]], [[2]], [[128]], [[129]]])),
        # pcp
        (torch.tensor([1, 2, 128, 129]), 2, 1, 1,
        torch.tensor([[[1], [0]], [[1], [1]], [[64], [64]], [[65], [64]]])),
        # dcp
        (torch.tensor([1, 2, 128, 129]), 1, 2, 1,
        torch.tensor([[[1, 0]], [[1, 1]], [[64, 64]], [[65, 64]]])),
        # pcp + dcp
        (torch.tensor([1, 2, 128, 129]), 2, 2, 1,
        torch.tensor([[[1, 0], [0, 0]], [[1, 1], [0, 0]],
                     [[32, 32], [32, 32]], [[33, 32], [32, 32]]])),
        # specify interleave_size
        (torch.tensor([1, 2, 128, 129]), 2, 1, 2,
        torch.tensor([[[1], [0]], [[2], [0]], [[64], [64]], [[65], [64]]])),
        (torch.tensor([1, 2, 128, 129]), 2, 1, 128,
        torch.tensor([[[1], [0]], [[2], [0]], [[128], [0]], [[128], [1]]])),
        (torch.tensor([1, 2, 128, 129, 256, 257]), 2, 2, 128,
        torch.tensor([[[1, 0], [0, 0]], [[2, 0], [0, 0]],
                     [[128, 0], [0, 0]], [[128, 1], [0, 0]],
                     [[128, 128], [0, 0]], [[128, 128], [1, 0]]])),
    ]
)
# yapf: enable
def test_get_cp_local_seq_lens(
    seq_lens,
    pcp_world_size,
    dcp_world_size,
    cp_kv_cache_interleave_size,
    target,
):
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 0
    pcp_manager = PCPManager(pcp_world_size=pcp_world_size,
                             pcp_rank=0,
                             dcp_world_size=dcp_world_size,
                             dcp_rank=0,
                             max_buffer_num_tokens=10000,
                             max_num_reqs=1000,
                             device="cpu",
                             vllm_config=vllm_config,
                             use_async_scheduling=False,
                             pin_memory=False)
    ret = pcp_manager._get_cp_local_seq_lens(seq_lens, pcp_world_size,
                                             dcp_world_size,
                                             cp_kv_cache_interleave_size)
    assert torch.equal(ret, target)


def test_get_cp_local_seq_lens_preserves_input_dtype_and_device():
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 0
    pcp_manager = PCPManager(pcp_world_size=2,
                             pcp_rank=0,
                             dcp_world_size=2,
                             dcp_rank=0,
                             max_buffer_num_tokens=10000,
                             max_num_reqs=1000,
                             device="cpu",
                             vllm_config=vllm_config,
                             use_async_scheduling=False,
                             pin_memory=False)
    seq_lens = torch.empty((2, ), dtype=torch.int64, device="meta")

    ret = pcp_manager._get_cp_local_seq_lens(seq_lens, 2, 2, 128)

    assert ret.shape == (2, 2, 2)
    assert ret.dtype == seq_lens.dtype
    assert ret.device == seq_lens.device


# yapf: disable
@pytest.mark.parametrize(
    "req_ids, num_computed_tokens," \
    "token_ids_tensor_list," \
    "num_reqs, total_num_scheduled_tokens, num_scheduled_tokens," \
    "target_input_ids_pcp_full, target_query_start_loc_pcp_full",
    [
        # prefill
        (
            ['0'], np.array([0]),
            [torch.tensor([0, 671, 6102, 294, 8760, 344])],
            1, 6, {'0': 6},
            torch.tensor([0, 671, 6102, 294, 8760, 344]),
            torch.tensor([0, 6])
        ),
        # decode
        (
            ['0'], np.array([6]),
            [torch.tensor([0, 671, 6102, 294, 8760, 344, 88907, 0])],
            1, 2, {'0': 2},
            torch.tensor([88907, 0]),
            torch.tensor([0, 2])
        ),
        # decode + prefill
        (
            ['0', '1'], np.array([6, 0]),
            [
                torch.tensor([0, 671, 6102, 294, 8760, 344, 88907, 0]),
                torch.tensor([0, 19923, 14, 1026, 2329, 344, 9807, 14, 342, 1030]),
            ],
            2, 12, {'0': 2, '1': 10},
            torch.tensor([88907, 0, 0, 19923, 14, 1026, 2329, 344, 9807, 14, 342, 1030]),
            torch.tensor([0, 2, 12])
        ),
        # decodes + prefills
        (
            ['0', '1', '2', '3'], np.array([6, 8, 0, 0]),
            [
                torch.tensor([0, 671, 6102, 294, 8760, 344, 88907, 0]),
                torch.tensor([0, 19923, 14, 1026, 2329, 344, 9807, 14, 342, 0]),
                torch.tensor([0, 671, 8749, 294, 3702, 4106, 344, 88907]),
                torch.tensor([0, 671, 5335, 1469, 7539, 305, 6397]),
            ],
            4, 19, {'0': 2, '1': 2, '2': 8, '3': 7},
            torch.tensor([88907, 0, 342, 0, 0, 671, 8749, 294, 3702, 4106, 344, 88907,
                          0, 671, 5335, 1469, 7539, 305, 6397]),
            torch.tensor([0, 2, 4, 12, 19])
        ),
    ])
# yapf: enable
def test_generate_pcp_mtp_input(
    req_ids,
    num_computed_tokens,
    token_ids_tensor_list,
    num_reqs,
    total_num_scheduled_tokens,
    num_scheduled_tokens,
    target_input_ids_pcp_full,
    target_query_start_loc_pcp_full,
):
    max_num_reqs = 4
    max_model_len = 4096
    max_num_tokens = 4096
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 1
    vllm_config.scheduler_config.max_num_seqs = max_num_reqs
    vllm_config.scheduler_config.max_num_batched_tokens = max_model_len
    pcp_manager = PCPManager(pcp_world_size=2,
                             pcp_rank=0,
                             dcp_world_size=1,
                             dcp_rank=0,
                             max_buffer_num_tokens=max_num_tokens,
                             max_num_reqs=max_num_reqs,
                             device="cpu",
                             vllm_config=vllm_config,
                             use_async_scheduling=False,
                             pin_memory=False)
    arange_np = np.arange(max_model_len)
    input_batch = MagicMock()
    input_batch.num_computed_tokens_cpu = \
        np.zeros(max_num_reqs, dtype=np.int32)
    token_ids_cpu_tensor = torch.zeros(
        (max_num_reqs, max_model_len),
        device="cpu",
        dtype=torch.int32,
    )
    input_batch.token_ids_cpu_tensor = token_ids_cpu_tensor
    input_batch.token_ids_cpu = token_ids_cpu_tensor.numpy()
    token_ids_cpu_tensor = input_batch.token_ids_cpu_tensor

    # Set input_batch
    input_batch.req_ids = req_ids
    input_batch.num_computed_tokens_cpu[:num_computed_tokens.
                                        size] = num_computed_tokens
    for i, token_ids_tensor in enumerate(token_ids_tensor_list):
        token_ids_cpu_tensor[i][:token_ids_tensor.size(0)] = token_ids_tensor

    num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
    for i, req_id in enumerate(req_ids):
        if num_computed_tokens[i] > 0:
            num_prompt_tokens[i] = num_computed_tokens[i]
        else:
            num_prompt_tokens[i] = num_scheduled_tokens[req_id]
    input_batch.num_prompt_tokens = num_prompt_tokens

    pcp_manager.init_batch_info(
        np.array(list(num_scheduled_tokens.values())),
        num_reqs,
        input_batch.num_computed_tokens_cpu,
        input_batch.num_prompt_tokens,
    )
    pcp_manager.generate_pcp_mtp_input(total_num_scheduled_tokens, num_scheduled_tokens, False,
                                       input_batch, arange_np)
    assert torch.equal(
        pcp_manager.input_ids_pcp_full.cpu[:total_num_scheduled_tokens],
        target_input_ids_pcp_full)
    assert torch.equal(pcp_manager.query_start_loc_pcp_full.cpu[:num_reqs + 1],
                       target_query_start_loc_pcp_full)


# yapf: disable
@pytest.mark.parametrize(
    "pcp_size, num_scheduled_tokens, num_decode_reqs,"
    " expected_cu_num_scheduled_tokens",
    [
        # Case 1: no prefill reqs -> returned unchanged.
        (2, [1, 1], 2, [1, 2]),
        # Case 2: prefill only (num_decode_reqs == 0).
        #   pcp_tokens (local prefill len) = [ceil(3/4)*2, ceil(5/4)*2] = [2, 4]
        #   cu=[3, 8]; pads cumsum=[1, 4]; base=0
        #   prefill_cu=[2, 6]; final = [2*2-1, 6*2-4] = [3, 8]
        (2, [3, 5], 0, [3, 8]),
        # Case 3: mix decode + prefill, pcp_size=2.
        #   cu=[1, 2, 5, 10]; decode part [1, 2] stays unchanged
        #   pcp_tokens[2:] = [2, 4]; pads[2:] cumsum=[1, 4]; base=cu[1]=2
        #   prefill_cu=[4, 8]; final[2:] = [4*2-1, 8*2-4] = [7, 12]
        (2, [1, 1, 3, 5], 2, [1, 2, 7, 12]),
        # Case 4: pcp_size=4, mix decode + prefill with uneven prefill tokens.
        #   cu=[1, 2, 7, 16]; decode part [1, 2] stays unchanged
        #   pcp_tokens[2:] = [ceil(5/8)*2, ceil(9/8)*2] = [2, 4]
        #   pads[2:] cumsum=[3, 10]; base=cu[1]=2
        #   prefill_cu=[4, 8]; final[2:] = [4*4-3, 8*4-10] = [13, 22]
        (4, [1, 1, 5, 9], 2, [1, 2, 13, 22]),
        # Case 5: single prefill req, pcp_size=2.
        #   pcp_tokens = [ceil(7/4)*2] = [4]; pads cumsum=[1]; base=0
        #   prefill_cu=[4]; final = [4*2-1] = [7]
        (2, [7], 0, [7]),
        # Case 6: prefill req already aligned to 2*pcp_size (no pad).
        #   pcp_tokens = [4]; pads=[0]; base=0; final = [4*2-0] = [8]
        (2, [8], 0, [8]),
    ],
)
# yapf: enable
def test_adjust_cu_num_scheduled_tokens_for_pcp(
    pcp_size,
    num_scheduled_tokens,
    num_decode_reqs,
    expected_cu_num_scheduled_tokens,
):
    vllm_config = MagicMock()
    vllm_config.model_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = 0
    vllm_config.scheduler_config.max_num_batched_tokens = 10000
    vllm_config.scheduler_config.max_num_seqs = 1000

    pcp_manager = PCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=0,
        dcp_world_size=1,
        dcp_rank=0,
        max_buffer_num_tokens=10000,
        max_num_reqs=1000,
        device="cpu",
        vllm_config=vllm_config,
        use_async_scheduling=False,
        pin_memory=False,
    )

    num_reqs = len(num_scheduled_tokens)
    num_scheduled_tokens_np = np.array(num_scheduled_tokens, dtype=np.int32)
    num_computed_tokens = np.array(
        [num_scheduled_tokens[i] if i < num_decode_reqs else 0 for i in range(num_reqs)],
        dtype=np.int32,
    )
    num_prompt_tokens = num_scheduled_tokens_np.copy()
    pcp_manager.init_batch_info(
        num_scheduled_tokens_np,
        num_reqs,
        num_computed_tokens,
        num_prompt_tokens,
    )

    cu_num_scheduled_tokens = np.cumsum(num_scheduled_tokens_np)
    pcp_manager.update_tokens_for_pcp(num_scheduled_tokens_np, np.arange(10000))
    assert pcp_manager.num_decode_reqs == num_decode_reqs
    num_pcp_pads = pcp_manager.num_pcp_pads_cpu[:num_reqs].astype(np.int32)

    result = pcp_manager.adjust_cu_num_scheduled_tokens_for_pcp(
        cu_num_scheduled_tokens, num_pcp_pads
    )

    assert np.array_equal(
        result, np.array(expected_cu_num_scheduled_tokens, dtype=np.int32)
    ), (
        f"Expected {expected_cu_num_scheduled_tokens}, got {result.tolist()}"
    )


@pytest.mark.parametrize("model_type", ["qwen3_next", "qwen3_5", "qwen3_5_moe"])
def test_hybrid_model_types_use_hybrid_pcp_layout(model_type):
    pcp_manager = _make_pcp_manager(model_type=model_type)
    num_scheduled_tokens = np.array([3, 5], dtype=np.int32)
    num_computed_tokens = np.zeros(2, dtype=np.int32)
    num_prompt_tokens = num_scheduled_tokens.copy()

    assert pcp_manager.pcp_use_hybrid_attn is True
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=2,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )
    pcp_tokens, positions = pcp_manager.update_tokens_for_pcp(
        num_scheduled_tokens,
        np.arange(10000, dtype=np.int32),
    )

    assert pcp_tokens.tolist() == [2, 3]
    assert positions.tolist() == [0, 1, 0, 1, 2]
    assert pcp_manager.num_scheduled_tokens_padded.tolist() == [2, 4]
    assert pcp_manager.max_num_tokens_across_pcp == 5
    assert pcp_manager.total_pcp_padding_tokens_fla == 2


def test_hybrid_update_tokens_records_fa_reorder_indices():
    pcp_manager = _make_pcp_manager(model_type="qwen3_next")
    num_scheduled_tokens = np.array([3, 5], dtype=np.int32)
    num_computed_tokens = np.zeros(2, dtype=np.int32)
    num_prompt_tokens = num_scheduled_tokens.copy()

    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=2,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )
    pcp_tokens, positions = pcp_manager.update_tokens_for_pcp(
        num_scheduled_tokens,
        np.arange(10000, dtype=np.int32),
    )

    assert pcp_tokens.tolist() == [2, 3]
    assert positions.tolist() == [0, 1, 0, 1, 2]
    assert pcp_manager.num_scheduled_tokens_padded.tolist() == [2, 4]
    assert pcp_manager.max_num_tokens_across_pcp == 5
    assert pcp_manager.total_num_scheduled_tokens == 5
    assert pcp_manager.pcp_padded_tokens_fla == 0
    assert pcp_manager.total_pcp_padding_tokens_fla == 2
    assert pcp_manager.pcp_enter_fa_restore_idx[:8].tolist() == [
        0, 1, 5, 2, 3, 4, 6, 7
    ]
    assert pcp_manager.pcp_fa_padding_restore_idx[:12].tolist() == [
        0, 1, 2, 8, 3, 4, 5, 6, 7, 8, 8, 8
    ]
    assert pcp_manager.pcp_exit_fa_scatter_idx.gpu[:8].tolist() == [
        0, 6, 2, 3, 8, 0, 0, 0
    ]
    assert pcp_manager.pcp_fa_query_idx[:6].tolist() == [0, 3, 4, 5, 10, 11]

    input_batch = MagicMock()
    input_batch.num_reqs = 2
    input_batch.num_computed_tokens_cpu = num_computed_tokens
    input_batch.num_prompt_tokens = torch.tensor(num_prompt_tokens)
    input_batch.num_tokens = torch.tensor(num_scheduled_tokens)
    metadata, _ = pcp_manager.generate_pcp_metadata(
        int(num_scheduled_tokens.sum()),
        torch.tensor(num_scheduled_tokens, dtype=torch.int32),
        input_batch,
        num_scheduled_tokens,
        torch.zeros((2, 1), dtype=torch.int32),
        num_reqs_padded=2,
        num_reqs=2,
    )

    assert metadata.num_actual_tokens_pcp_padded == 12
    assert metadata.pcp_enter_fa_restore_idx.tolist() == [
        0, 1, 5, 2, 3, 4, 6, 7
    ]
    assert metadata.pcp_fa_padding_restore_idx.tolist() == [
        0, 1, 2, 8, 3, 4, 5, 6, 7, 8, 8, 8
    ]
    assert metadata.pcp_exit_fa_scatter_idx.tolist() == [
        0, 6, 2, 3, 8, 0, 0, 0
    ]
    assert metadata.pcp_fa_query_idx.tolist() == [0, 3, 4, 5, 10, 11]
    assert metadata.pcp_allgather_restore_idx.tolist() == [
        0, 6, 7, 1, 2, 3, 8, 9, 10, 11, 4, 5
    ]


def test_hybrid_logits_indices_include_decode_padding():
    pcp_manager = _make_pcp_manager(model_type="qwen3_next")
    num_scheduled_tokens = np.array([1, 1, 3, 5], dtype=np.int32)
    num_computed_tokens = np.array([16, 17, 0, 0], dtype=np.int32)
    num_prompt_tokens = np.array([16, 17, 3, 5], dtype=np.int32)
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=4,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )

    logits_indices = pcp_manager.get_logits_indices(
        np.cumsum(num_scheduled_tokens),
        num_reqs=4,
        tokens_original=num_scheduled_tokens.tolist(),
    )

    assert logits_indices.tolist() == [1, 3, 6, 11]


def test_hybrid_padded_slot_mapping_uses_independent_kv_group_buffers():
    pcp_manager = _make_pcp_manager(model_type="qwen3_next")
    num_scheduled_tokens = np.array([3, 5], dtype=np.int32)
    num_computed_tokens = np.zeros(2, dtype=np.int32)
    num_prompt_tokens = num_scheduled_tokens.copy()
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=2,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )
    pcp_manager.update_tokens_for_pcp(
        num_scheduled_tokens,
        np.arange(10000, dtype=np.int32),
    )
    pcp_manager.initialize_slot_mapping()
    pcp_manager.initialize_slot_mapping()

    group0 = pcp_manager.get_padded_slot_mapping(
        num_tokens=5,
        num_tokens_padded=8,
        slot_mapping=torch.arange(8, dtype=torch.int32),
        kv_cache_group_id=0,
    )
    expected_group0 = [0, 1, 2, -1, 3, 4, 5, 6, 7, -1, -1, -1]
    assert group0.tolist() == expected_group0

    group1 = pcp_manager.get_padded_slot_mapping(
        num_tokens=5,
        num_tokens_padded=8,
        slot_mapping=torch.arange(100, 108, dtype=torch.int32),
        kv_cache_group_id=1,
    )

    assert group1.tolist() == [
        100, 101, 102, -1, 103, 104, 105, 106, 107, -1, -1, -1
    ]
    assert pcp_manager.pcp_padded_slot_mapping_list[0][:12].tolist() == expected_group0


def _local_kv_len(seq_len: int, cp_rank: int, cp_size: int, interleave_size: int) -> int:
    """Per-rank KV length for the first ``seq_len`` global positions [0, seq_len)."""
    base = seq_len // interleave_size // cp_size * interleave_size
    remainder = seq_len - base * cp_size
    return base + max(0, min(remainder - cp_rank * interleave_size, interleave_size))


def _ref_mtp_mask_lens(
    history_len: int,
    num_scheduled: int,
    cp_rank: int,
    cp_size: int,
    interleave_size: int,
) -> tuple[int, list[int]]:
    """Reference: compute per-rank KV length and k_upper per query position.

    ``k_upper`` is the inclusive local index of KV tokens with global
    ``pos <= P`` on this rank (standard causal, including self).
    """
    total_len = history_len + num_scheduled
    k_lens = _local_kv_len(total_len, cp_rank, cp_size, interleave_size)

    context_len = history_len
    k_uppers: list[int] = []
    for qi in range(num_scheduled):
        inclusive_len = context_len + qi + 1
        k_upper = _local_kv_len(inclusive_len, cp_rank, cp_size, interleave_size) - 1
        k_uppers.append(k_upper)

    return k_lens, k_uppers


def _build_ref_mask(
    q_lens: int,
    k_lens: int,
    k_uppers: list[int],
) -> torch.Tensor:
    """Build the reference attention mask with the same logic as the function under test."""
    mask = torch.zeros(q_lens, k_lens, dtype=torch.bool)
    for qi, ku in enumerate(k_uppers):
        if ku >= 0:
            mask[qi, ku + 1 :] = True
        # When ku < 0 the guard (ku >= 0) makes the AND-gate False, so mask stays False.
    return mask


# yapf: disable
@pytest.mark.parametrize(
    "dcp_world_size, dcp_rank, history_len, num_scheduled, interleave_size",
    [
        # interleave_size=1: token-level interleave, all cp_rank see similar counts
        (2, 0, 200, 100, 1),
        (2, 1, 200, 100, 1),
        (4, 0, 300, 50, 1),
        (4, 3, 300, 50, 1),
        # interleave_size=128: block-level interleave
        # total_len < 128: rank-0 owns everything, rank-1 owns nothing
        (2, 0, 50, 10, 128),
        (2, 1, 50, 10, 128),
        # total_len crosses one interleave boundary (128 < L < 256)
        (2, 0, 100, 50, 128),
        (2, 1, 100, 50, 128),
        # total_len crosses multiple interleave boundaries (L > 256)
        (2, 0, 200, 100, 128),
        (2, 1, 200, 100, 128),
        # exactly at interleave boundary
        (2, 0, 128, 10, 128),
        (2, 1, 128, 10, 128),
        # cp_size=4 with interleave_size=128
        (4, 0, 300, 50, 128),
        (4, 2, 300, 50, 128),
        (4, 3, 300, 50, 128),
    ],
)
# yapf: enable
def test_generate_mtp_attention_mask_for_decode(
    dcp_world_size: int,
    dcp_rank: int,
    history_len: int,
    num_scheduled: int,
    interleave_size: int,
):
    """Verify interleave-aware MTP attention masks for decode requests.

    Compares the function output against a reference implementation that
    applies the same formulas.  Covers both token-level (I=1) and
    block-level (I=128) interleave patterns.
    """
    pcp_world_size = 1
    cp_size = dcp_world_size * pcp_world_size
    cp_rank = dcp_rank

    pcp_manager = _make_pcp_manager(
        pcp_world_size=pcp_world_size,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        num_speculative_tokens=num_scheduled - 1,
        max_model_len=4096,
        max_buffer_num_tokens=4096,
        max_num_reqs=8,
        cp_kv_cache_interleave_size=interleave_size,
    )

    num_scheduled_tokens = np.array([num_scheduled], dtype=np.int32)
    num_computed_tokens = np.array([history_len], dtype=np.int32)
    num_prompt_tokens = np.array([history_len], dtype=np.int32)
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=1,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )
    assert pcp_manager.num_decode_reqs == 1

    result = pcp_manager.generate_mtp_attention_mask_for_decode(
        decode_num_computed_tokens=[history_len],
        decode_num_scheduled_tokens=num_scheduled_tokens,
    )
    assert result is not None

    ref_k_lens, ref_k_uppers = _ref_mtp_mask_lens(
        history_len, num_scheduled, cp_rank, cp_size, interleave_size
    )
    ref_mask = _build_ref_mask(num_scheduled, ref_k_lens, ref_k_uppers)

    max_q = num_scheduled
    max_k = ref_k_lens

    if ref_k_lens == 0:
        # k_lens=0 means valid.any() is False; function returns early with
        # zero-filled mask.
        assert result[0, :max_q, :max_k].sum() == 0, (
            f"Expected all-False mask for k_lens=0, "
            f"history={history_len}, scheduled={num_scheduled}, dcp_rank={dcp_rank}"
        )
        return

    actual_mask = result[0, :max_q, :max_k]
    assert actual_mask.shape == ref_mask.shape, (
        f"Shape mismatch: {actual_mask.shape} vs {ref_mask.shape}"
    )
    assert torch.equal(actual_mask, ref_mask), (
        f"Mask mismatch for "
        f"history={history_len}, scheduled={num_scheduled}, "
        f"dcp_rank={dcp_rank}, interleave={interleave_size}\n"
        f"ref_k_lens={ref_k_lens}, ref_k_uppers={ref_k_uppers}\n"
        f"Expected:\n{ref_mask.int()}\nGot:\n{actual_mask.int()}"
    )


@pytest.mark.parametrize(
    "dcp_world_size, dcp_rank, history_len, num_scheduled",
    [
        (2, 0, 5, 4),
        (2, 1, 5, 4),
        (2, 0, 200, 3),
        (2, 1, 200, 3),
        (4, 0, 100, 5),
        (4, 3, 100, 5),
    ],
)
def test_mtp_mask_interleave1_matches_legacy_inclusive_formula(
    dcp_world_size: int,
    dcp_rank: int,
    history_len: int,
    num_scheduled: int,
):
    """For interleave_size=1, k_upper must match the pre-#11492 inclusive formula."""
    _, k_uppers = _ref_mtp_mask_lens(
        history_len, num_scheduled, dcp_rank, dcp_world_size, interleave_size=1
    )
    for qi, k_upper in enumerate(k_uppers):
        pos = history_len + qi
        legacy = (pos - dcp_rank) // dcp_world_size
        assert k_upper == legacy, (
            f"pos={pos}, rank={dcp_rank}: got k_upper={k_upper}, legacy={legacy}"
        )

    pcp_manager = _make_pcp_manager(
        pcp_world_size=1,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=1,
        num_speculative_tokens=num_scheduled - 1,
    )
    num_scheduled_tokens = np.array([num_scheduled], dtype=np.int32)
    num_computed_tokens = np.array([history_len], dtype=np.int32)
    num_prompt_tokens = np.array([history_len], dtype=np.int32)
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=1,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )
    result = pcp_manager.generate_mtp_attention_mask_for_decode(
        decode_num_computed_tokens=[history_len],
        decode_num_scheduled_tokens=num_scheduled_tokens,
    )
    ref_k_lens, ref_k_uppers = _ref_mtp_mask_lens(
        history_len, num_scheduled, dcp_rank, dcp_world_size, interleave_size=1
    )
    ref_mask = _build_ref_mask(num_scheduled, ref_k_lens, ref_k_uppers)
    assert torch.equal(result[0, :num_scheduled, :ref_k_lens], ref_mask)


def test_remap_mrope_positions_for_pcp_uses_local_positions():
    pcp_manager = _make_pcp_manager()
    input_batch = SimpleNamespace(req_ids=["req0", "req1"])
    req0_mrope = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
        ],
        dtype=torch.int64,
    )
    requests = {
        "req0": SimpleNamespace(
            prompt_token_ids=[0, 1, 2, 3, 4],
            prompt_embeds=None,
            mrope_positions=req0_mrope,
            mrope_position_delta=100,
        ),
        "req1": SimpleNamespace(
            prompt_token_ids=[0, 1],
            prompt_embeds=None,
            mrope_positions=None,
            mrope_position_delta=None,
        ),
    }
    mrope_positions = SimpleNamespace(cpu=torch.zeros((3, 5), dtype=torch.int64))

    pcp_manager.remap_mrope_positions_for_pcp(
        positions_np=np.array([1, 3, 5, 2, 4], dtype=np.int64),
        num_scheduled_tokens=np.array([3, 2], dtype=np.int32),
        num_reqs=2,
        input_batch=input_batch,
        requests=requests,
        mrope_positions=mrope_positions,
    )

    expected = torch.tensor(
        [
            [1, 3, 105, 2, 4],
            [11, 13, 105, 2, 4],
            [21, 23, 105, 2, 4],
        ],
        dtype=torch.int64,
    )
    assert torch.equal(mrope_positions.cpu, expected)


def test_get_restore_hidden_states_uses_pcp_allgather_restore_idx():
    pcp_manager = _make_pcp_manager(model_type="qwen2")
    num_scheduled_tokens = np.array([3, 5], dtype=np.int32)
    num_computed_tokens = np.zeros(2, dtype=np.int32)
    num_prompt_tokens = num_scheduled_tokens.copy()
    pcp_manager.init_batch_info(
        num_scheduled_tokens,
        num_reqs=2,
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
    )
    pcp_manager.update_tokens_for_pcp(
        num_scheduled_tokens,
        np.arange(10000, dtype=np.int32),
    )
    pcp_manager.num_actual_tokens_pcp_padded = 12
    gathered_hidden_states = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    local_hidden_states = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    fake_group = MagicMock()
    fake_group.all_gather.return_value = gathered_hidden_states

    with patch("vllm.distributed.parallel_state.get_pcp_group", return_value=fake_group):
        restored = pcp_manager.get_restore_hidden_states(local_hidden_states)

    local_arg, dim_arg = fake_group.all_gather.call_args.args
    assert dim_arg == 0
    assert local_arg.shape == (6, 1)
    assert restored.flatten().tolist() == [
        0, 6, 7, 1, 2, 3, 8, 9, 10, 11, 4, 5
    ]
