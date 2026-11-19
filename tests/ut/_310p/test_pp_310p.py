# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend._310p.worker.v2 import pp


@pytest.fixture
def make_handler(monkeypatch):
    def create(*, is_last_rank):
        group = SimpleNamespace(
            is_last_rank=is_last_rank,
            last_rank=3,
            ranks=[1, 3],
            world_size=2,
            group_ranks=[[0, 2], [1, 3]],
        )
        monkeypatch.setattr(pp, "get_pp_group", lambda: group)
        monkeypatch.setattr(pp.dist, "new_subgroups_by_enumeration", lambda *args, **kwargs: (object(), []))
        return pp.PPHandler310(max_num_reqs=4)

    return create


def make_batch():
    return SimpleNamespace(
        num_reqs=3,
        idx_mapping_np=np.array([0, 1, 2]),
        num_computed_tokens_np=np.array([0, 4, 6]),
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        prefill_len_np=np.array([4, 4, 4]),
        max_seq_len_np=None,
    )


def test_pipeline_send_uses_int64_wire_format(monkeypatch, make_handler):
    handler = make_handler(is_last_rank=True)
    send = Mock(return_value=Mock())
    monkeypatch.setattr(pp.dist, "isend", send)
    tokens = torch.tensor([[11], [12], [13]], dtype=torch.int32)
    counts = torch.tensor([0, 1, 1], dtype=torch.int32)
    handler.broadcast(tokens, counts, torch.zeros_like(counts), make_batch())

    payload = send.call_args.args[0]
    torch.testing.assert_close(payload, torch.tensor([[11, 12, 13], [0, 1, 1]], dtype=torch.int64))
    assert send.call_args.kwargs["dst"] == 1


def test_pipeline_receive_filters_cancelled_and_prefill_rows(monkeypatch, make_handler):
    handler = make_handler(is_last_rank=False)
    work = Mock()

    def receive(payload, *, src, group):
        assert src == 3
        assert group is handler.broadcast_group
        payload.copy_(torch.tensor([[11, 12, 13], [0, 1, 1]], dtype=torch.int64))
        return work

    monkeypatch.setattr(pp.dist, "irecv", receive)
    batch = make_batch()
    assert handler.get_prev_sampled_outputs() is None
    assert not handler.receive(batch, recv_drafts=False)
    batch.idx_mapping_np[:] = 3
    handler.on_req_idx_freed(1)
    assert handler.get_prev_sampled_outputs() is None
    pending = handler.get_prev_sampled_outputs()

    work.wait.assert_called_once()
    assert pending is not None
    assert pending["idx_mapping"].tolist() == [-1, -1, 2]
    assert pending["sampled_tokens"][2, 0].item() == 13
