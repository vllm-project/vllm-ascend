# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.attention.context_parallel.sfa_cp as pcp_module
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


@pytest.mark.parametrize("replicated", [0, 1])
@pytest.mark.parametrize("actual_rows", [2, 4])
def test_main_kv_completion_preserves_padding_slots_and_replicated_prefix(monkeypatch, replicated, actual_rows):
    impl = pcp_module.AscendSFAPCPImpl.__new__(pcp_module.AscendSFAPCPImpl)
    impl.enable_main_kv_async = True
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True))
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_c8 = False
    impl.is_pcp_decode_sharded = False
    metadata = SimpleNamespace(
        num_input_tokens=4,
        num_decode_tokens=replicated,
        num_prefills=1,
        pcp_has_global_prefill=True,
        uses_pcp_sharded_layout=True,
    )
    monkeypatch.setattr(pcp_module, "get_pcp_group", lambda: SimpleNamespace(world_size=8))
    monkeypatch.setattr(pcp_module, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(pcp_module, "get_dcp_group", lambda: SimpleNamespace(world_size=1))
    inputs = torch.arange(actual_rows * 3, dtype=torch.bfloat16).reshape(actual_rows, 3)
    padded = torch.nn.functional.pad(inputs, (0, 0, 0, 4 - actual_rows))
    slots = torch.arange(32)
    cache = (torch.zeros(32, 3, dtype=torch.bfloat16),)
    events = []

    def launch(send, group):
        assert group.world_size == 8
        assert torch.equal(send, padded[replicated:])
        events.append("launch")
        gathered = torch.full((8 * (4 - replicated), 3), float("nan"), dtype=torch.bfloat16)

        def wait():
            events.append("wait")
            gathered.copy_(torch.cat([send + rank * 10 for rank in range(8)]))

        return gathered, SimpleNamespace(wait=wait)

    monkeypatch.setattr(pcp_module, "all_gather_async", launch)

    def write(self, values, cos, sin, actual_cache, actual_slots, actual_metadata):
        assert events == ["launch", "query", "wait"]
        assert actual_metadata is metadata and actual_cache is cache
        cache[0][actual_slots] = values
        return None, None

    writer = Mock(side_effect=write)
    monkeypatch.setattr(AscendSFAImpl, "exec_kv", lambda self, *args: writer(self, *args))
    finish = impl._start_main_kv_gather(inputs, None, None, cache, slots, metadata)
    writer.assert_not_called()
    events.append("query")
    assert finish() == (None, None)
    expected = torch.zeros_like(cache[0])
    expected[:replicated] = padded[:replicated]
    for rank in range(8):
        expected[rank * 4 + replicated : (rank + 1) * 4] = padded[replicated:] + rank * 10
    assert torch.equal(cache[0], expected)


@pytest.mark.parametrize("gate", ["disabled", "graph", "rope", "c8", "fp32"])
def test_main_kv_unsupported_modes_do_not_submit_collectives(monkeypatch, gate):
    impl = pcp_module.AscendSFAPCPImpl.__new__(pcp_module.AscendSFAPCPImpl)
    impl.enable_main_kv_async = gate != "disabled"
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enforce_eager=gate != "graph"))
    impl.qk_rope_head_dim = 8 if gate == "rope" else 0
    impl.enable_sparse_sfa_c8 = gate == "c8"
    values = torch.zeros(2, 3, dtype=torch.float32 if gate == "fp32" else torch.bfloat16)
    gather = Mock()
    monkeypatch.setattr(pcp_module, "all_gather_async", gather)
    assert impl._start_main_kv_gather(values, None, None, (values,), None, None) is None
    gather.assert_not_called()
