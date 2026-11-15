# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs

import vllm_ascend.attention.context_parallel.sfa_cp as pcp_module
import vllm_ascend.attention.sfa_v1 as sparse_mla
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


@pytest.mark.parametrize(
    "rope,rows,replicated,dcp,sharded_decode",
    [
        pytest.param(False, 0, 0, False, False, id="empty-rank"),
        pytest.param(False, 2, 0, False, False, id="padded-nope"),
        pytest.param(False, 4, 1, True, False, id="dcp-replicated-prefix"),
        pytest.param(True, 4, 0, False, False, id="rope-prefill"),
        pytest.param(True, 4, 1, True, False, id="dcp-rope-mixed"),
        pytest.param(False, 2, 1, True, True, id="dcp-sharded-decode"),
    ],
)
@pytest.mark.parametrize("defer_wait", [False, True])
def test_main_kv_matches_synchronous_inputs_and_slots(
    monkeypatch, rope, rows, replicated, dcp, sharded_decode, defer_wait
):
    pcp = 2
    impl_cls = pcp_module.AscendSFADCPImpl if dcp else AscendSFAImpl
    impl = impl_cls.__new__(impl_cls)
    impl.pcp_enabled = True
    impl.qk_rope_head_dim = 2 if rope else 0
    impl.is_pcp_decode_sharded = sharded_decode
    metadata = pcp_module.AscendSFADCPMetadata.__new__(pcp_module.AscendSFADCPMetadata)
    metadata.num_input_tokens = 4
    metadata.num_decode_tokens = replicated
    metadata.num_prefills = 0  # This rank must join another rank's prefill.
    metadata.pcp_has_global_prefill = not sharded_decode
    metadata.uses_pcp_sharded_layout = True
    slots = torch.arange(pcp * 4)
    metadata.dcp_context = SimpleNamespace(slot_mapping=slots.flip(0))
    inputs = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    cos, sin = (torch.randn(rows, 2), torch.randn(rows, 2)) if rope else (None, None)
    cache = (torch.zeros(pcp * 4, 3),)
    events: list[str] = []
    writes: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]] = []

    def gather(value, dim=0):
        return torch.cat([value + rank for rank in range(pcp)], dim=dim)

    group = SimpleNamespace(world_size=pcp, all_gather=gather)
    monkeypatch.setattr(sparse_mla, "get_pcp_group", lambda: group)
    monkeypatch.setattr("vllm.v1.attention.ops.pcp.get_pcp_group", lambda: group)

    def write(self, values, actual_cos, actual_sin, actual_cache, actual_slots, actual_metadata):
        assert actual_cache is cache and actual_metadata is metadata
        if events:
            count = 3 if rope else 1
            assert events == ["launch"] * count + (["query"] if defer_wait else []) + ["wait"] * count
        writes.append((values.clone(), actual_cos, actual_sin, actual_slots.clone()))
        return None, None

    monkeypatch.setattr(AscendSFAImpl, "_exec_kv_local", write)
    # Keep the unchanged upstream synchronous helper as an independent oracle.
    expected_slots = metadata.dcp_context.slot_mapping if dcp and not sharded_decode else slots
    values = (inputs, cos, sin) if rope else (torch.nn.functional.pad(inputs, (0, 0, 0, 4 - rows)),)
    values, expected_slots = _gather_prefill_cache_inputs(values, expected_slots, 0 if sharded_decode else replicated)
    expected_cos, expected_sin = values[1:] if rope else (None, None)
    expected_result = write(impl, values[0], expected_cos, expected_sin, cache, expected_slots, metadata)

    def launch(send, actual_group):
        assert actual_group is group and send.is_contiguous()
        events.append("launch")
        output = send.new_full((send.shape[0] * pcp, send.shape[1]), float("nan"))

        def wait():
            events.append("wait")
            output.copy_(gather(send))

        return output, SimpleNamespace(wait=wait)

    monkeypatch.setattr(sparse_mla, "all_gather_async", launch)
    result = impl.exec_kv(inputs, cos, sin, cache, slots, metadata, defer_wait=defer_wait)
    if defer_wait:
        assert callable(result) and len(writes) == 1
        events.append("query")
        result = result()
    assert result == expected_result
    assert len(writes) == 2
    for expected, actual in zip(*writes):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(expected, actual)


@pytest.mark.parametrize("mode", ["non_pcp", "unexpanded", "replicated_only", "empty"])
def test_main_kv_non_collective_paths_do_not_submit_work(monkeypatch, mode):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = mode != "non_pcp"
    impl.qk_rope_head_dim = 0
    impl.is_pcp_decode_sharded = False
    metadata = SimpleNamespace(
        num_input_tokens=0 if mode == "empty" else 2,
        num_decode_tokens=2 if mode == "replicated_only" else 0,
        num_prefills=1,
        pcp_has_global_prefill=False,
        uses_pcp_sharded_layout=True,
    )
    values = torch.zeros(0 if mode == "empty" else 2, 3)
    slots = torch.arange(2 if mode == "unexpanded" else 4)
    gather = Mock()
    monkeypatch.setattr(sparse_mla, "all_gather_async", gather)
    write = Mock(return_value=(None, None))
    monkeypatch.setattr(AscendSFAImpl, "_exec_kv_local", write)
    finish = impl.exec_kv(values, None, None, (values,), slots, metadata, defer_wait=True)
    write.assert_not_called()
    assert callable(finish)
    result = finish()
    assert result == (None, None)
    write.assert_called_once()
    gather.assert_not_called()
    expected_rows = 0 if mode in {"unexpanded", "empty"} else 2
    assert torch.equal(write.call_args.args[0], values[:expected_rows])
    assert torch.equal(write.call_args.args[4], slots if mode == "non_pcp" else slots[:expected_rows])


def test_main_kv_retains_contiguous_send_until_completion(monkeypatch):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.pcp_enabled = True
    impl.qk_rope_head_dim = 0
    impl.is_pcp_decode_sharded = False
    metadata = SimpleNamespace(
        num_input_tokens=4,
        num_decode_tokens=0,
        num_prefills=1,
        pcp_has_global_prefill=True,
        uses_pcp_sharded_layout=True,
    )
    values = torch.arange(12).view(3, 4).T
    assert not values.is_contiguous()
    group = SimpleNamespace(world_size=2)
    monkeypatch.setattr(sparse_mla, "get_pcp_group", lambda: group)

    def launch(send, actual_group):
        assert actual_group is group and send.is_contiguous()
        send_ref = weakref.ref(send)
        output = torch.empty(8, 3, dtype=send.dtype)

        def wait():
            retained_send = send_ref()
            assert retained_send is not None
            output.copy_(torch.cat((retained_send, retained_send)))

        return output, SimpleNamespace(wait=wait)

    monkeypatch.setattr(sparse_mla, "all_gather_async", launch)
    write = Mock(return_value=(None, None))
    monkeypatch.setattr(AscendSFAImpl, "_exec_kv_local", write)
    finish = impl.exec_kv(values, None, None, (), torch.arange(8), metadata, defer_wait=True)
    assert callable(finish)
    finish()
    write.assert_called_once()
    torch.testing.assert_close(write.call_args.args[0], torch.cat((values, values)))
