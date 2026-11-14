# SPDX-License-Identifier: Apache-2.0
"""Run MRV2 attention preparation through initial SFA Host-to-hot handoff."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.adapter import MooncakeSFAPDState
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import CopySfaRequestStates
from vllm_ascend.worker.v2.model_states import default


@pytest.mark.parametrize("computed", [127, 128, 2047, 2048, 2049, 2175, 2176, 2303, 4095])
@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("local_hit", [False, True])
def test_first_real_decode_primes_correct_group_before_graph_or_eager_forward(monkeypatch, computed, mode, local_hit):
    pd = MooncakeSFAPDState(main_group_id=1)
    pd.update({"r": 2}, {} if local_hit else {"r": computed + 1})
    pd.record_received({"r"}, set())
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        model_config=SimpleNamespace(max_model_len=4096),
    )
    sparse = SimpleNamespace(topk_buffer_size=2048)
    state = default.AscendModelState.__new__(default.AscendModelState)
    state.vllm_config = cfg
    state.model_config = cfg.model_config
    state._offload_live_req_ids = {"r"}
    state._offload_request_states = CopySfaRequestStates()
    for name, dtype in [("_offload_pool_slots", np.int32), ("_offload_pool_active", np.bool_)]:
        values = np.zeros(4, dtype=dtype)
        setattr(state, name, SimpleNamespace(np=values, cpu=torch.from_numpy(values)))
    state.pcp_manager = state.device_metadata = state.kvpp_runtime = None
    state.decoder_replay = None
    state._prepare_decoder_replay = MagicMock()
    batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=2,
        num_tokens=1,
        num_tokens_after_padding=2,
        req_ids=["r"],
        num_computed_tokens_np=np.array([computed]),
        query_start_loc_np=np.array([0, 1]),
        is_prefilling_np=np.array([False]),
        num_scheduled_tokens=torch.tensor([1]),
        query_start_loc=torch.tensor([0, 1]),
        seq_lens=torch.tensor([computed + 1]),
        seq_lens_np=np.array([computed + 1]),
        positions=torch.tensor([computed]),
        attn_state="decode",
        dcp_local_seq_lens=None,
        is_dummy=False,
    )
    manager = SimpleNamespace(dense_fill_copy_sfa_rows=MagicMock(), restore_copy_sfa_tails=MagicMock())
    connector = SimpleNamespace(get_copy_sfa_pd_state=lambda: pd)
    monkeypatch.setattr(default, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(default, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(default, "get_prebound_copy_sfa_slots", lambda: dict(pd.bindings))
    monkeypatch.setattr(default, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=sparse))
    monkeypatch.setattr(default, "get_sparse_kv_offload_manager", lambda: manager)
    builds = []
    missing_tail_metadata = False

    def build(**kwargs):
        builds.append(kwargs)
        needs_tail = kwargs["copy_sfa_restore_tails"]
        return {
            "main": SimpleNamespace(
                copy_sfa_copy_src_offsets=object() if needs_tail and not missing_tail_metadata else None
            )
        }

    monkeypatch.setattr(default, "build_attn_metadata", build)
    tables = (torch.tensor([[1, 2]]), torch.tensor([[7, 8]]))
    needs_dense = local_hit and computed <= 2048
    needs_tail = (local_hit and computed > 2048 and computed % 128 != 0) or (
        not local_hit and computed + 1 > 2048 and (computed + 1) % 128 == 0
    )
    if needs_dense or needs_tail:
        operation = manager.dense_fill_copy_sfa_rows if needs_dense else manager.restore_copy_sfa_tails
        operation.side_effect = RuntimeError("initial copy failed")
        with pytest.raises(RuntimeError, match="initial copy failed"):
            state.prepare_attn(batch, mode, tables, torch.empty(0), [], None)
        assert pd.pending == pd.received == {"r"}
        operation.reset_mock()
        operation.side_effect = None
    if needs_tail and not local_hit:
        missing_tail_metadata = True
        with pytest.raises(RuntimeError, match="missing its tail-copy metadata"):
            state.prepare_attn(batch, mode, tables, torch.empty(0), [], None)
        assert pd.pending == pd.received == {"r"}
        missing_tail_metadata = False
    state.prepare_attn(batch, mode, tables, torch.empty(0), [], None)
    assert not pd.pending
    assert state._offload_pool_slots.np[0] == 2
    if local_hit and computed <= 2048:
        args = manager.dense_fill_copy_sfa_rows.call_args
        assert args.args == ({2: (0, computed)},)
        assert args.kwargs["block_table"] is tables[1]
    else:
        manager.dense_fill_copy_sfa_rows.assert_not_called()
    assert manager.restore_copy_sfa_tails.call_count == int(needs_tail)
    before = (manager.dense_fill_copy_sfa_rows.call_count, manager.restore_copy_sfa_tails.call_count)
    state.prepare_attn(batch, mode, tables, torch.empty(0), [], None)
    assert before == (manager.dense_fill_copy_sfa_rows.call_count, manager.restore_copy_sfa_tails.call_count)
    assert builds[-1]["copy_sfa_restore_tails"] is False
