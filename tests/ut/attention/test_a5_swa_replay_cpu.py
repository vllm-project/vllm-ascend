# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata composition checks; the device index builder is emulated."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_v41.py"
WINDOW = 128


def _load_query_inputs():
    # Load the real metadata code without importing the NPU-only model stack.
    nodes = ast.parse(SOURCE.read_text()).body
    visible = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "build_replay_swa_visible_lens")
    builder = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "AscendDSAV41MetadataBuilder")
    prepare = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_query_inputs")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, visible, prepare], type_ignores=[]))
    scope = {"torch": torch}
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope["_prepare_query_inputs"], scope


def _window_indices(positions, window_size, *, visible_lengths=None, indices_output, lengths_output):
    # Emulate the device boundary only; assertions below use explicit KV ranges.
    indices_output.fill_(-1)
    lengths_output.zero_()
    for row, position in enumerate(positions.tolist()):
        floor = max(0, position - window_size + 1)
        if visible_lengths is not None:
            if row >= visible_lengths.numel():
                continue
            floor = max(floor, position + 1 - max(0, int(visible_lengths[row])))
        values = torch.arange(floor, position + 1, dtype=torch.int32)
        indices_output[row, 0, : values.numel()] = values
        lengths_output[row] = values.numel()
    return indices_output, lengths_output


def _inputs(a5=True):
    positions = torch.tensor(list(range(33792, 33920)) + [0, 1, 639, 0, 0])
    req = SimpleNamespace(
        seq_lens=torch.tensor([33920, 2, 640], dtype=torch.int32),
        cache_kind="swa",
        positions=positions,
        num_reqs=3,
        num_actual_tokens=131,
        num_input_tokens=133,
        storage_block_size=128,
    )
    common = SimpleNamespace(
        query_start_loc=torch.tensor([0, 128, 130, 131], dtype=torch.int32),
        replay_start=None,
        max_seq_len=33920,
        causal=True,
    )
    backend = Mock(side_effect=_window_indices)
    builder = SimpleNamespace(
        _window_size=WINDOW,
        _a5_causal_swa_indices=torch.empty((133, 1, WINDOW), dtype=torch.int32) if a5 else None,
        _a5_causal_swa_lengths=torch.empty((133, 1), dtype=torch.int32) if a5 else None,
        _device_backend=SimpleNamespace(build_window_indices=backend),
        common_v41_batch_metadata={},
    )
    return builder, common, req, backend


@pytest.mark.parametrize("prime_cache", [False, True])
def test_a5_replay_keeps_the_valid_kv_prefix_inside_the_replay_boundary(prime_cache):
    prepare, _ = _load_query_inputs()
    builder, common, req, backend = _inputs()
    if prime_cache:
        prepare(builder, common, req)
    common.replay_start = torch.tensor([33792, 0, 639], dtype=torch.int32)

    assert prepare(builder, common, req) == common.max_seq_len

    expected = [list(range(33792, position + 1)) for position in range(33792, 33920)]
    expected += [[0], [0, 1], [639], [], []]
    for row, wanted in enumerate(expected):
        count = req.ori_topk_length[row, 0].item()
        assert count == len(wanted)
        assert req.ori_sparse_indices[row, 0, :count].tolist() == wanted
        assert req.ori_sparse_indices[row, 0, count:].tolist() == [-1] * (WINDOW - count)
    assert backend.call_count == 1 + int(prime_cache)
    assert backend.call_args.kwargs["visible_lengths"] is not None
    assert (req.ori_mask_mode, req.ori_win_left, req.ori_win_right) == (4, WINDOW - 1, 0)


def test_a5_ordinary_causal_windows_still_reuse_the_batch_cache():
    prepare, _ = _load_query_inputs()
    builder, common, req, backend = _inputs()
    prepare(builder, common, req)
    first_indices, first_lengths = req.ori_sparse_indices, req.ori_topk_length
    prepare(builder, common, req)
    backend.assert_called_once()
    assert req.ori_sparse_indices is first_indices
    assert req.ori_topk_length is first_lengths
    assert first_indices[0, 0].tolist() == list(range(33665, 33793))
    assert first_lengths[0, 0].item() == WINDOW


def test_scalar_band_replay_keeps_its_per_token_lengths():
    prepare, _ = _load_query_inputs()
    builder, common, req, backend = _inputs(a5=False)
    common.replay_start = torch.tensor([33792, 0, 639], dtype=torch.int32)
    prepare(builder, common, req)
    backend.assert_not_called()
    assert req.ori_sparse_indices is None
    assert req.ori_topk_length.flatten().tolist() == list(range(1, 129)) + [1, 2, 1]


def test_noncausal_explicit_indices_keep_their_existing_lengths():
    prepare, _ = _load_query_inputs()
    builder, common, req, backend = _inputs()
    common.causal = False
    indices = torch.tensor([[[2, 4, -1]], [[1, 3, 5]]], dtype=torch.int32)
    prepare(builder, common, req, ori_sparse_indices=indices)
    backend.assert_not_called()
    assert req.ori_sparse_indices is indices
    assert req.ori_topk_length.tolist() == [[2], [3]]
    assert (req.ori_mask_mode, req.ori_win_left, req.ori_win_right) == (0, WINDOW - 1, 0)
