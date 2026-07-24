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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.attention.utils import AscendDCPMetadata
from vllm_ascend.worker.dcp_utils import DCPManager


def _make_dcp_manager(
    dcp_world_size: int,
    dcp_rank: int,
    interleave_size: int,
    max_query_len: int = 8,
    max_model_len: int = 1024,
) -> DCPManager:
    manager = object.__new__(DCPManager)
    manager.dcp_world_size = dcp_world_size
    manager.dcp_world_rank = dcp_rank
    manager.num_decode_reqs = 1
    manager.vllm_config = MagicMock()
    manager.vllm_config.parallel_config.cp_kv_cache_interleave_size = interleave_size
    manager.dcp_mtp_attn_mask = MagicMock()
    manager.dcp_mtp_attn_mask.cpu = torch.zeros((1, max_query_len, max_model_len), dtype=torch.bool)
    return manager


@pytest.mark.parametrize(
    "dcp_world_size, interleave_size, expected",
    [
        (
            2,
            1,
            [
                [1, 0],
                [1, 1],
                [64, 64],
                [65, 64],
                [128, 128],
                [129, 128],
            ],
        ),
        (
            2,
            128,
            [
                [1, 0],
                [2, 0],
                [128, 0],
                [128, 1],
                [128, 128],
                [129, 128],
            ],
        ),
        (
            4,
            128,
            [
                [1, 0, 0, 0],
                [2, 0, 0, 0],
                [128, 0, 0, 0],
                [128, 1, 0, 0],
                [128, 128, 0, 0],
                [128, 128, 1, 0],
            ],
        ),
    ],
)
def test_get_dcp_local_seq_lens_interleaves_kv_across_ranks(
    dcp_world_size: int,
    interleave_size: int,
    expected: list[list[int]],
) -> None:
    manager = _make_dcp_manager(
        dcp_world_size=dcp_world_size,
        dcp_rank=0,
        interleave_size=interleave_size,
    )
    seq_lens = torch.tensor([1, 2, 128, 129, 256, 257])

    actual = manager._get_dcp_local_seq_lens(seq_lens)

    assert torch.equal(actual, torch.tensor(expected))


@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_generate_mtp_attention_mask_for_decode(dcp_rank: int) -> None:
    manager = _make_dcp_manager(
        dcp_world_size=2,
        dcp_rank=dcp_rank,
        interleave_size=1,
    )
    history_len = 5
    num_scheduled = 4

    actual = manager.generate_mtp_attention_mask_for_decode(
        decode_num_computed_tokens=[history_len],
        decode_num_scheduled_tokens=np.array([num_scheduled], dtype=np.int32),
    )

    total_len = history_len + num_scheduled
    local_k_len = (total_len + 1 - dcp_rank) // 2
    positions = torch.arange(history_len, history_len + num_scheduled)
    local_visible = (positions + 1 + 1 - dcp_rank) // 2
    expected = torch.arange(local_k_len)[None, :] >= local_visible[:, None]

    assert torch.equal(
        actual[0, :num_scheduled, :local_k_len],
        expected,
    )


def test_generate_dcp_mtp_input_fills_query_start_loc_tail() -> None:
    manager = object.__new__(DCPManager)
    manager.num_reqs = 2
    manager.use_async_scheduling = False
    manager.decode_threshold = 2
    manager.query_start_loc_full = MagicMock()
    manager.query_start_loc_full.np = np.zeros(5, dtype=np.int32)
    input_batch = MagicMock()
    input_batch.req_ids = ["request-0", "request-1"]

    manager.generate_dcp_mtp_input(
        total_num_scheduled_tokens=5,
        num_scheduled_tokens={"request-0": 2, "request-1": 3},
        input_batch=input_batch,
        req_indices=np.arange(5, dtype=np.int32),
        positions_np=np.arange(5, dtype=np.int64),
        cu_num_tokens=np.array([2], dtype=np.int32),
    )

    np.testing.assert_array_equal(
        manager.query_start_loc_full.np,
        np.array([0, 2, 5, -1, -1], dtype=np.int32),
    )
    manager.query_start_loc_full.copy_to_gpu.assert_called_once_with()


def test_update_spec_decode_drafting_metadata_skips_prefill() -> None:
    manager = object.__new__(DCPManager)
    manager.dcp_world_rank = 0
    manager._get_dcp_local_seq_lens = MagicMock(return_value=torch.tensor([[4, 3]], dtype=torch.int32))
    attn_metadata = MagicMock()
    attn_metadata.decode_meta = None

    with (
        patch.object(DCPManager, "_is_mla_kv_cache_spec", return_value=False),
        patch.object(DCPManager, "_is_sfa_dcp_metadata_builder", return_value=False),
    ):
        manager.update_spec_decode_drafting_cp_metadata(
            attn_metadata=attn_metadata,
            kv_cache_spec=object(),
            seq_lens=torch.tensor([3]),
            draft_index=0,
        )

    assert attn_metadata.decode_meta is None


def test_prepare_spec_decode_drafting_metadata_transitions_to_decode() -> None:
    manager = object.__new__(DCPManager)
    manager.dcp_world_rank = 1
    local_seq_lens = torch.tensor([[4, 3], [6, 5]], dtype=torch.int32)
    manager._get_dcp_local_seq_lens = MagicMock(return_value=local_seq_lens)
    mtp_mask = torch.ones((2, 4, 16), dtype=torch.bool)
    original_dcp_metadata = AscendDCPMetadata(
        num_computed_tokens_of_dcp=[[3, 2], [5, 4]],
        query_lens_cpu=torch.tensor([8, 4], dtype=torch.int32),
        max_query_len=8,
        dcp_mtp_attn_mask=mtp_mask,
    )
    common_attn_metadata = SimpleNamespace(
        context_parallel_metadata=original_dcp_metadata,
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling=torch.tensor([True, True]),
    )
    seq_lens = torch.tensor([7, 11], dtype=torch.int32)
    seq_lens_cpu = torch.tensor([6, 10], dtype=torch.int32)

    with patch.object(DCPManager, "_is_mla_kv_cache_spec", return_value=True):
        manager.prepare_spec_decode_drafting_cp_metadata(
            common_attn_metadata=common_attn_metadata,
            kv_cache_spec=object(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            draft_index=1,
        )

    draft_dcp_metadata = common_attn_metadata.context_parallel_metadata
    assert draft_dcp_metadata is not original_dcp_metadata
    assert torch.equal(
        draft_dcp_metadata.query_lens_cpu,
        torch.tensor([1, 1], dtype=torch.int32),
    )
    assert draft_dcp_metadata.max_query_len == 1
    assert torch.equal(
        draft_dcp_metadata.num_computed_tokens_of_dcp,
        local_seq_lens,
    )
    assert torch.equal(
        draft_dcp_metadata.draft_cp_seq_len,
        torch.tensor([3, 5], dtype=torch.int32),
    )
    assert torch.equal(
        draft_dcp_metadata.draft_base_seq_lens,
        torch.tensor([13, 13], dtype=torch.int32),
    )
    assert draft_dcp_metadata.dcp_mtp_attn_mask is mtp_mask
    assert not torch.any(common_attn_metadata.is_prefilling)
    assert original_dcp_metadata.max_query_len == 8
    assert original_dcp_metadata.draft_cp_seq_len is None
    assert original_dcp_metadata.draft_base_seq_lens is None
    manager._get_dcp_local_seq_lens.assert_called_once()
    assert torch.equal(
        manager._get_dcp_local_seq_lens.call_args.args[0],
        torch.tensor([15, 15], dtype=torch.int32),
    )


def test_update_spec_decode_drafting_metadata_requires_mla_decode() -> None:
    manager = object.__new__(DCPManager)
    attn_metadata = SimpleNamespace(decode=None)

    with (
        patch.object(DCPManager, "_is_mla_kv_cache_spec", return_value=True),
        pytest.raises(AssertionError, match="must be classified as decode"),
    ):
        manager.update_spec_decode_drafting_cp_metadata(
            attn_metadata=attn_metadata,
            kv_cache_spec=object(),
            seq_lens=torch.tensor([3]),
            draft_index=1,
        )
