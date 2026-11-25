# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch_npu

from vllm_ascend.compilation import fia_graph_replay


def make_task(block_table, name):
    return SimpleNamespace(
        operation=torch_npu.npu_fused_infer_attention_score.out,
        kwargs={
            "block_table": block_table,
            "actual_seq_lengths_kv": [1, 1],
            "query": torch.empty((2, 16, 128), dtype=torch.bfloat16),
            "out": [torch.empty((2, 16, 128), dtype=torch.bfloat16)],
            "input_layout": "TND",
            "sparse_mode": 0,
        },
        provider=SimpleNamespace(layer_name=name),
        event=object(),
    )


def test_replay_shares_one_device_length_buffer(monkeypatch):
    native = MagicMock(ARGS_CAPACITY=4096)
    monkeypatch.setattr(fia_graph_replay, "_native", native)
    table = torch.zeros((2, 4), dtype=torch.int32)
    tasks = [make_task(table, name) for name in ("first", "second")]

    replay = fia_graph_replay.FIAReplay.bind(42, tasks)
    assert replay is not None
    native.bind_lengths.assert_called_once_with(
        42,
        [(task.kwargs["query"].data_ptr(), task.kwargs["out"][0].data_ptr()) for task in tasks],
        table.data_ptr(),
        [1, 1],
        replay.lengths.data_ptr(),
        replay.launch_args.data_ptr(),
    )

    metadata = SimpleNamespace(block_tables=table, seq_lens_gpu=torch.tensor([129, 0]), max_query_len=1)
    length_pointer = replay.lengths.data_ptr()
    replay.prepare({"first": metadata})
    metadata.seq_lens_gpu[0] += 1
    replay.prepare({"first": metadata})
    assert replay.lengths.tolist() == [130, 0]
    assert replay.lengths.data_ptr() == length_pointer

    metadata.block_tables = table.clone()
    with pytest.raises(RuntimeError, match="captured single-token batch shape"):
        replay.prepare({"first": metadata})


def test_rebind_leaves_different_block_tables_on_original_update_path(monkeypatch):
    native = MagicMock(ARGS_CAPACITY=4096)
    monkeypatch.setattr(fia_graph_replay, "_native", native)
    table = torch.zeros((2, 4), dtype=torch.int32)
    tasks = [make_task(table, "first"), make_task(table.clone(), "second")]

    assert fia_graph_replay.FIAReplay.bind(42, tasks) is None
    native.bind_lengths.assert_not_called()


def test_rebind_leaves_unnamed_provider_on_original_update_path(monkeypatch):
    native = MagicMock(ARGS_CAPACITY=4096)
    monkeypatch.setattr(fia_graph_replay, "_native", native)
    task = make_task(torch.zeros((2, 4), dtype=torch.int32), None)
    task.provider = SimpleNamespace()

    assert fia_graph_replay.FIAReplay.bind(42, [task]) is None
    native.bind_lengths.assert_not_called()


@pytest.mark.parametrize("missing", ["metadata", "block_tables", "seq_lens_gpu"])
def test_replay_requires_layer_metadata(missing):
    table = torch.zeros((2, 4), dtype=torch.int32)
    lengths = torch.tensor([129, 0])
    replay = fia_graph_replay.FIAReplay("first", table, lengths, torch.empty(1))
    metadata = SimpleNamespace(block_tables=table, seq_lens_gpu=lengths, max_query_len=1)

    if missing == "metadata":
        with pytest.raises(RuntimeError, match="Missing FIA replay metadata for layer first"):
            replay.prepare({})
    else:
        setattr(metadata, missing, None)
        with pytest.raises(RuntimeError, match="captured single-token batch shape"):
            replay.prepare({"first": metadata})
