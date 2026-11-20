# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata wiring checks; numerical index coverage lives in the NPU tests."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_v41.py"
WINDOW = 128
PADDED_TOKENS = 8


def _load_metadata_builder(real_visible_lens=False):
    # Load the real metadata method without importing the NPU-only model stack.
    nodes = ast.parse(SOURCE.read_text(encoding="utf-8")).body
    builder = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == "AscendDSAV41MetadataBuilder")
    build = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    body: list[ast.stmt] = [future]
    if real_visible_lens:
        body.append(
            next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "build_replay_swa_visible_lens")
        )
    body.append(build)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    scope = {
        "torch": torch,
        "AscendDSAV41Metadata": SimpleNamespace,
        "_config_value": getattr,
        "get_kv_cache_compression_ratio": lambda spec: 0,
        "get_storage_block_size": lambda spec: spec.block_size,
        "build_replay_swa_visible_lens": Mock(),
    }
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope["build"], scope["build_replay_swa_visible_lens"]


def _inputs(a5=True):
    positions = torch.tensor(list(range(33792, 33920)) + [0, 1, 639, 0, 0])
    seq_lens = torch.tensor([33920, 2, 640], dtype=torch.int32)
    common = SimpleNamespace(
        seq_lens=seq_lens,
        query_start_loc=torch.tensor([0, 128, 130, 131], dtype=torch.int32),
        slot_mapping=torch.arange(positions.numel()),
        block_table_tensor=torch.empty((seq_lens.numel(), 0), dtype=torch.int32),
        num_actual_tokens=131,
        replay_start=None,
        max_seq_len=33920,
        causal=True,
    )
    indices = torch.empty((positions.numel(), 1, WINDOW), dtype=torch.int32) if a5 else None
    lengths = torch.empty((positions.numel(), 1), dtype=torch.int32) if a5 else None
    backend = Mock(return_value=(indices, lengths))
    text_config = SimpleNamespace(sliding_window=WINDOW, num_attention_heads=64, head_dim=512, index_topk=512)
    builder = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(block_size=WINDOW, sliding_window=WINDOW),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=text_config),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        ),
        _cache_kind="swa",
        _device_metadata_in_graph=False,
        _uses_a5_packed_cache=a5,
        _build_query_metadata=False,
        _build_compressor_metadata=False,
        _slot_mapping_2d=torch.empty((positions.numel(), 2), dtype=torch.int32),
        _flat_slot_mapping=torch.empty_like(common.slot_mapping),
        _a5_causal_swa_indices=indices,
        _a5_causal_swa_lengths=lengths,
        _device_backend=SimpleNamespace(build_window_indices=backend),
    )
    batch_shared = {
        "batch": {"positions": positions, "seq_lens": seq_lens, "max_cache_seq_len": common.max_seq_len},
        "rope": (None, None),
    }
    return builder, common, batch_shared, backend


@pytest.mark.parametrize("a5", [False, True])
@pytest.mark.parametrize("query_lens", [(1,), (2, 1)])
def test_replay_visible_lengths_exclude_padded_requests(a5, query_lens):
    build, _ = _load_metadata_builder(real_visible_lens=True)
    builder, common, batch_shared, backend = _inputs(a5=a5)
    num_actual_reqs = len(query_lens)
    common.num_actual_tokens = sum(query_lens)
    common.num_input_tokens = PADDED_TOKENS
    num_padding_tokens = PADDED_TOKENS - common.num_actual_tokens
    common.num_reqs = num_actual_reqs + num_padding_tokens
    padded_query_lens = torch.tensor([*query_lens, *([1] * num_padding_tokens)], dtype=torch.int32)
    common.query_start_loc = torch.cat((torch.zeros(1, dtype=torch.int32), padded_query_lens.cumsum(0).to(torch.int32)))
    cached_prefix_tokens = 1024
    seq_lens = [cached_prefix_tokens + 1 + request_idx * WINDOW for request_idx in range(num_actual_reqs)]
    common.seq_lens = torch.tensor(seq_lens + [0] * num_padding_tokens, dtype=torch.int32)
    common.replay_start = torch.tensor([length - WINDOW for length in seq_lens] + [0] * num_padding_tokens)
    common.slot_mapping = torch.arange(PADDED_TOKENS)
    positions = [position for length, count in zip(seq_lens, query_lens) for position in range(length - count, length)]
    batch_shared["batch"].update(
        positions=torch.tensor(positions + [0] * num_padding_tokens),
        seq_lens=common.seq_lens,
    )

    metadata = build(
        builder,
        0,
        common,
        num_actual_reqs=num_actual_reqs,
        common_v41_batch_metadata=batch_shared,
        full_graph_mode=True,
    )

    expected = [[WINDOW - count + offset + 1] for count in query_lens for offset in range(count)]
    if a5:
        visible_lengths = backend.call_args.kwargs["visible_lengths"]
    else:
        backend.assert_not_called()
        visible_lengths = metadata.ori_topk_length
    assert visible_lengths.dtype == torch.int32
    assert visible_lengths.tolist() == expected


@pytest.mark.parametrize("prime_cache", [False, True])
def test_a5_replay_forwards_visible_lengths_and_bypasses_cached_windows(prime_cache):
    build, visible_builder = _load_metadata_builder()
    builder, common, batch_shared, backend = _inputs()
    visible_lengths = torch.empty((common.num_actual_tokens, 1), dtype=torch.int32)
    visible_builder.return_value = visible_lengths
    if prime_cache:
        build(builder, 0, common, common_v41_batch_metadata=batch_shared)
    common.replay_start = torch.tensor([33792, 0, 639], dtype=torch.int32)

    metadata = build(builder, 0, common, common_v41_batch_metadata=batch_shared)

    visible_builder.assert_called_once()
    assert visible_builder.call_args.args[3] is common.replay_start
    assert backend.call_count == 1 + int(prime_cache)
    assert backend.call_args.kwargs.get("visible_lengths") is visible_lengths
    assert (f"a5-causal-swa:w{WINDOW}" in batch_shared) == prime_cache
    assert metadata.ori_sparse_indices is backend.return_value[0]
    assert metadata.ori_topk_length is backend.return_value[1]
    assert (metadata.ori_mask_mode, metadata.ori_win_left, metadata.ori_win_right) == (4, WINDOW - 1, 0)

    build(builder, 0, common, common_v41_batch_metadata=batch_shared)

    visible_builder.assert_called_once()
    assert backend.call_count == 2 + int(prime_cache)
    assert backend.call_args.kwargs["visible_lengths"] is visible_lengths


def test_a5_ordinary_causal_windows_still_reuse_the_batch_cache():
    build, _ = _load_metadata_builder()
    builder, common, batch_shared, backend = _inputs()
    first = build(builder, 0, common, common_v41_batch_metadata=batch_shared)
    second = build(builder, 0, common, common_v41_batch_metadata=batch_shared)
    backend.assert_called_once()
    assert second.ori_sparse_indices is first.ori_sparse_indices
    assert second.ori_topk_length is first.ori_topk_length
    assert backend.call_args.kwargs.get("visible_lengths") is None


def test_scalar_band_replay_keeps_its_per_token_lengths():
    build, visible_builder = _load_metadata_builder()
    builder, common, batch_shared, backend = _inputs(a5=False)
    visible_lengths = torch.empty((common.num_actual_tokens, 1), dtype=torch.int32)
    visible_builder.return_value = visible_lengths
    common.replay_start = torch.tensor([33792, 0, 639], dtype=torch.int32)
    metadata = build(builder, 0, common, common_v41_batch_metadata=batch_shared)
    backend.assert_not_called()
    assert metadata.ori_sparse_indices is None
    assert metadata.ori_topk_length is visible_lengths


def test_noncausal_explicit_indices_keep_their_existing_lengths():
    build, _ = _load_metadata_builder()
    builder, common, batch_shared, backend = _inputs()
    common.causal = False
    indices = torch.tensor([[[2, 4, -1]], [[1, 3, 5]]], dtype=torch.int32)
    metadata = build(builder, 0, common, common_v41_batch_metadata=batch_shared, ori_sparse_indices=indices)
    backend.assert_not_called()
    assert metadata.ori_sparse_indices is indices
    assert metadata.ori_topk_length.tolist() == [[2], [3]]
    assert (metadata.ori_mask_mode, metadata.ori_win_left, metadata.ori_win_right) == (0, WINDOW - 1, 0)
