# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops import mega_kda


@pytest.mark.parametrize("mixed", [False, True])
def test_mega_kda_reads_device_metadata_and_merges_token_order(monkeypatch, mixed):
    dim = mega_kda.MEGA_KDA_HEAD_DIM
    mx = None
    if mixed:
        qkv = torch.arange(4 * dim, dtype=torch.int32).to(torch.uint8).reshape(4, dim)
        scales = torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2)
        weight = torch.zeros(3 * dim, dim, dtype=torch.uint8)
        weight_scale = torch.zeros(3 * dim, dim // 32, dtype=torch.uint8)
        mx = mega_kda.MxQkvProjection(scales, weight, weight_scale)
    else:
        qkv = torch.randn(4, 3 * dim, dtype=torch.bfloat16)
    raw_gate = torch.randn(1, 4, 1, dim, dtype=torch.bfloat16)
    gate = torch.randn(4, 1, dim, dtype=torch.bfloat16)
    beta = torch.randn(1, 4, 1, dtype=torch.bfloat16)
    result = torch.empty_like(raw_gate)
    spec_indices = torch.tensor([1, 3])
    prefill_indices = torch.tensor([0, 2])
    accepted = torch.tensor([2], dtype=torch.int32)
    spec_slots = torch.tensor([[7, 8]], dtype=torch.int32)
    prefill_slots = torch.tensor([11], dtype=torch.int32)
    no_initial = torch.tensor([False])
    prefix = torch.tensor([0, 2], dtype=torch.int32)
    norm = MagicMock()
    layer = SimpleNamespace(local_num_heads=1, o_norm=SimpleNamespace(forward_oot=norm))
    operator = MagicMock()
    packed_weights = torch.empty(4, 3 * dim, dtype=torch.bfloat16)
    spec_output = torch.ones(2, dim, dtype=torch.bfloat16)
    prefill_output = torch.full_like(spec_output, 2)
    launch = MagicMock(side_effect=[spec_output, prefill_output])
    monkeypatch.setattr(mega_kda, "_run_projected", launch)
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        num_spec_decodes=1,
        num_prefills=1,
        num_decodes=0,
        spec_token_indx=spec_indices,
        non_spec_token_indx=prefill_indices,
        spec_state_indices_tensor=spec_slots,
        non_spec_state_indices_tensor=prefill_slots,
        spec_decode_metadata=SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(
                query_start_loc=prefix, cache_indices=spec_slots[:, 0], num_accepted_tokens=accepted
            ),
        ),
        non_spec_prefill_metadata=SimpleNamespace(
            causal_conv1d=SimpleNamespace(
                query_start_loc=prefix, cache_indices=prefill_slots, initial_state_mode=no_initial
            ),
        ),
    )

    mega_kda.run_mega_kda(layer, operator, metadata, qkv, raw_gate, gate, beta, result, packed_weights, mx)

    spec_args, prefill_args = [call.args for call in launch.call_args_list]
    torch.testing.assert_close(spec_args[2], qkv[spec_indices])
    torch.testing.assert_close(prefill_args[2], qkv[prefill_indices])
    assert spec_args[6] is prefix
    assert spec_args[7] is spec_slots
    assert spec_args[9] is accepted
    assert prefill_args[10] is no_initial
    if mixed:
        spec_call, prefill_call = launch.call_args_list
        torch.testing.assert_close(spec_call.kwargs["mx_input_scale"], scales[spec_indices])
        torch.testing.assert_close(prefill_call.kwargs["mx_input_scale"], scales[prefill_indices])
        assert spec_call.kwargs["mx_weights"][0].data_ptr() == weight.data_ptr()
        assert spec_call.kwargs["mx_weights"][1].shape == (3 * dim, dim // 64, 2)
    expected = torch.stack([prefill_output[0], spec_output[0], prefill_output[1], spec_output[1]]).reshape_as(result)
    torch.testing.assert_close(norm.call_args.args[0], expected)
    assert norm.call_args.kwargs["out"] is result


def test_mixed_decode_uses_existing_state_and_single_token_scale(monkeypatch):
    dim = mega_kda.MEGA_KDA_HEAD_DIM
    qkv = torch.zeros(2, dim, dtype=torch.uint8)
    scales = torch.arange(8, dtype=torch.uint8).reshape(2, 2, 2)
    weights = torch.empty(3 * dim, dim, dtype=torch.uint8)
    mx = mega_kda.MxQkvProjection(scales, weights, torch.empty(3 * dim, dim // 32, dtype=torch.uint8))
    gate = torch.zeros(1, 2, 1, dim, dtype=torch.bfloat16)
    beta = torch.zeros(1, 2, 1, dtype=torch.bfloat16)
    slots = torch.tensor([7], dtype=torch.int32)
    conv_slots = torch.tensor([3], dtype=torch.int32)
    prefix = torch.tensor([0, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        num_spec_decodes=0,
        num_prefills=0,
        num_decodes=1,
        non_spec_state_indices_tensor=slots,
        non_spec_decode_metadata=SimpleNamespace(
            causal_conv1d=SimpleNamespace(query_start_loc=prefix, cache_indices=conv_slots)
        ),
    )
    layer = SimpleNamespace(local_num_heads=1, o_norm=MagicMock())
    launch = MagicMock(return_value=torch.ones(1, dim, dtype=torch.bfloat16))
    monkeypatch.setattr(mega_kda, "_run_projected", launch)
    result = torch.empty_like(gate[:, :1])
    conv_weight = torch.empty(4, 3 * dim, dtype=torch.bfloat16)

    mega_kda.run_mega_kda(layer, MagicMock(), metadata, qkv, gate, gate[0], beta, result, conv_weight, mx)

    launch.assert_called_once()
    args, kwargs = launch.call_args
    torch.testing.assert_close(args[2], qkv[:1])
    torch.testing.assert_close(kwargs["mx_input_scale"], scales[:1])
    assert args[6] is prefix
    assert args[7] is slots
    assert args[8] is conv_slots
    torch.testing.assert_close(args[9], torch.ones_like(slots))
    torch.testing.assert_close(args[10], torch.ones_like(slots))
    assert layer.o_norm.forward_oot.call_args.kwargs["out"] is result


def test_mega_kda_clears_idle_graph_output(monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr(mega_kda, "_run_projected", launch)
    metadata = SimpleNamespace(num_actual_tokens=4, num_spec_decodes=0, num_prefills=0, num_decodes=0)
    result = torch.full((1, 4, 1, mega_kda.MEGA_KDA_HEAD_DIM), float("nan"))

    mega_kda.run_mega_kda(None, None, metadata, None, None, None, None, result, None)

    assert torch.count_nonzero(result) == 0
    launch.assert_not_called()


def test_mega_kda_requires_projected_wheel(tmp_path):
    with pytest.raises(RuntimeError, match="projected operator wheel"):
        mega_kda.load_mega_kda(str(tmp_path))
