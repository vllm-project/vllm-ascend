# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState
from vllm.models.qwen4_exp.common.qsa_cache import _build_qsa_metadata_torch
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.qwen4_exp.ngram_context import gather_ngram_context
from vllm_ascend.ops.triton.qwen4_exp.qsa_metadata import build_qsa_metadata_triton

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)


def _capture(launch):
    launch()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = launch()
    return graph, output


def test_ple_history_replay_uses_corrected_device_lengths():
    history = torch.arange(96, dtype=torch.int32).reshape(3, 32)[:, ::2]
    storage = torch.full((3, 6), -17, dtype=torch.int32, device="npu")
    context = storage[:, ::2]
    gpu_history = torch.arange(96, dtype=torch.int32, device="npu").reshape(3, 32)[:, ::2]
    mapping = torch.tensor([2, 0], dtype=torch.int32, device="npu")
    lengths = torch.tensor([0, 7, 4], dtype=torch.int32, device="npu")
    graph, actual = _capture(lambda: gather_ngram_context(context, mapping, lengths, gpu_history, 2, 99))
    assert actual.data_ptr() == context.data_ptr()
    for current_mapping, current_lengths in (([2, 0], [0, 7, 4]), ([1, 2], [1, 3, 2])):
        mapping.copy_(torch.tensor(current_mapping, dtype=torch.int32))
        lengths.copy_(torch.tensor(current_lengths, dtype=torch.int32))
        state = SimpleNamespace(
            ngram_context=torch.full((3, 3), -17, dtype=torch.int32),
            ngram_eos_token_id=99,
            ngram_context_offsets=torch.arange(-3, 0),
        )
        batch = SimpleNamespace(num_reqs=2, num_reqs_after_padding=3, idx_mapping=mapping.cpu())
        requests = SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu=lengths.cpu()), all_token_ids=SimpleNamespace(gpu=history)
        )
        expected = Qwen4ExpModelState._prepare_ngram_context(state, batch, requests)
        graph.replay()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        assert torch.all(storage[:, 1::2] == -17)


@pytest.mark.parametrize("ratio,circular", [(1, 0), (4, 0), (1, 8)])
def test_qsa_metadata_replay_preserves_torch_values_and_owners(ratio, circular):
    query_cpu = torch.tensor([0, 5, 8], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_cpu.npu(),
        query_start_loc_cpu=query_cpu,
        seq_lens=torch.tensor([9, 13], dtype=torch.int32, device="npu"),
        num_reqs=2,
        num_actual_tokens=12,
        max_query_len=5,
        max_seq_len=13,
        block_table_tensor=torch.arange(1, 9, dtype=torch.int32, device="npu").reshape(2, 4),
        slot_mapping=torch.tensor(list(range(8)) + [-1] * 4, dtype=torch.int64, device="npu"),
    )
    expected_buffers = [
        torch.full((16,), -17, dtype=dtype, device="npu")
        for dtype in (torch.int32, torch.int64, torch.int32, torch.int64)
    ]
    actual_buffers = [buffer.clone() for buffer in expected_buffers]
    expected_work = torch.full((6, 2), -17, dtype=torch.int32, device="npu") if ratio > 1 else None
    actual_work = None if expected_work is None else expected_work.clone()
    kwargs = dict(storage_block_size=4, compress_ratio=ratio, circular_buffer_size=circular)
    graph, actual = _capture(
        lambda: build_qsa_metadata_triton(common, *actual_buffers, k_work_metadata_buffer=actual_work, **kwargs)
    )
    owners = actual_buffers[:3] + [actual_buffers[3] if ratio > 1 or circular else common.slot_mapping]
    assert all(value.data_ptr() == owner.data_ptr() for value, owner in zip(actual, owners))
    for seq_lens in ([9, 13], [8, 10]):
        common.seq_lens.copy_(torch.tensor(seq_lens, dtype=torch.int32))
        expected = _build_qsa_metadata_torch(common, *expected_buffers, k_work_metadata_buffer=expected_work, **kwargs)
        expected = [value.cpu().clone() for value in expected]
        graph.replay()
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value.cpu(), reference, rtol=0, atol=0)
        for value, reference in zip(actual_buffers, expected_buffers):
            torch.testing.assert_close(value.cpu(), reference.cpu(), rtol=0, atol=0)
        if actual_work is not None:
            torch.testing.assert_close(actual_work.cpu(), expected_work.cpu(), rtol=0, atol=0)
