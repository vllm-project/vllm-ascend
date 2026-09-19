# SPDX-License-Identifier: Apache-2.0
"""Prefill chunk planning and mixed decode device metadata regressions."""

import ast
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "vllm_ascend/attention/mla_prefill.py"
spec = importlib.util.spec_from_file_location("flash_mla_prefill_test_module", MODULE_PATH)
prefill_ops = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prefill_ops
spec.loader.exec_module(prefill_ops)


def load_functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)


@pytest.mark.parametrize(
    "contexts,budget", [([0, 0, 0], 128), ([0, 127, 129, 515], 512), ([131071, 129761], 16384), ([500] * 7, 256)]
)
def test_prefix_partition_preserves_all_tokens_and_memory_bound(contexts, budget):
    lengths = tuple(5 + i for i in range(len(contexts)))
    offsets = [0, 4, 8]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    plan = prefill_ops.build_prefill_plan(offsets, 2, contexts, budget)
    assert plan.num_decode_tokens == 8
    assert plan.query_lengths == lengths
    covered = [[] for _ in contexts]
    for chunk in plan.chunks:
        assert 0 < chunk.num_tokens <= budget
        assert all(start % 128 == 0 for start in chunk.starts)
        assert all(length > 0 for length in chunk.lengths)
        assert chunk.query_lengths == tuple(lengths[i] for i in chunk.requests)
        for req, start, length in zip(chunk.requests, chunk.starts, chunk.lengths):
            covered[req].extend(range(start, start + length))
    assert covered == [list(range(length)) for length in contexts]
    metadata = prefill_ops.prepare_metadata(plan, torch.device("cpu"))
    assert metadata.current.query_cu[-1].item() == sum(lengths)


def test_lse_layout_preserves_query_and_head_order():
    lse = torch.arange(15).reshape(5, 3, 1).float()
    torch.testing.assert_close(prefill_ops.canonical_lse(lse, 5, 3), lse)
    torch.testing.assert_close(prefill_ops.canonical_lse(lse[..., 0].T, 5, 3), lse)


def test_equal_history_chunks_share_readonly_metadata_within_batch():
    calls = []

    def schedule(call):
        calls.append(call)
        return torch.tensor([len(calls)])

    plan = prefill_ops.build_prefill_plan([0, 5, 14], 0, [131072, 131072], 256)
    metadata = prefill_ops.prepare_metadata(plan, torch.device("cpu"), schedule)
    assert len(metadata.history) == 1024
    assert len(calls) == 2  # One history shape and the causal current chunk.
    first = metadata.history[0]
    assert all(chunk.call is first.call for chunk in metadata.history)
    assert metadata.current is not first.call
    assert metadata.history[-1].starts.tolist() == [130944, 130944]
    assert metadata.history[-1].starts is not first.starts


def test_history_metadata_cache_distinguishes_ragged_lengths_and_batches():
    calls = []

    def schedule(call):
        calls.append(call)
        return torch.tensor([len(calls)])

    plan = prefill_ops.build_prefill_plan([0, 5, 14], 0, [385, 257], 256)
    first = prefill_ops.prepare_metadata(plan, torch.device("cpu"), schedule)
    assert first.history[0].call is first.history[1].call
    assert first.history[2].call.kv_lengths == (128, 1)
    assert first.history[3].call.kv_lengths == (1,)
    assert first.history[2].call is not first.history[0].call
    assert first.history[3].call is not first.history[2].call
    assert len(calls) == 4
    second = prefill_ops.prepare_metadata(plan, torch.device("cpu"), schedule)
    assert len(calls) == 8
    assert second.current is not first.current
    for before, after in zip(first.history, second.history):
        assert before.call is not after.call
        assert before.call.schedule is not after.call.schedule


def test_current_and_history_do_not_share_different_mask_modes():
    plan = prefill_ops.build_prefill_plan([0, 128], 0, [128], 128)
    metadata = prefill_ops.prepare_metadata(plan, torch.device("cpu"))
    history = metadata.history[0].call
    assert history.query_lengths == metadata.current.query_lengths
    assert history.kv_lengths == metadata.current.kv_lengths
    assert history.mask_mode == 0 and metadata.current.mask_mode == 3
    assert history is not metadata.current


def test_reject_decode_lengths_in_prefill_contract():
    with pytest.raises(ValueError, match="metadata mismatch"):
        prefill_ops.build_prefill_plan([0, 4, 8, 13], 2, [9999, 9999, 129], 128)


def test_mixed_decode_uses_live_device_lengths_not_prefill_cpu_counts():
    scope = dict(
        torch=torch,
        dataclass=dataclass,
        replace=replace,
        build_prefill_plan=prefill_ops.build_prefill_plan,
        prepare_metadata=prefill_ops.prepare_metadata,
        native_flash_adapters=lambda *a: (lambda call: None, None),
        _flash_attention_schedule=lambda builder, flash, **kw: torch.stack(
            (flash.used_q.sum(), flash.cache_lens.sum())
        ),
        DeviceMetadataStage=SimpleNamespace(ATTENTION=1),
        DeviceMetadataTask=lambda stage, fn, group: SimpleNamespace(fn=fn, group=group),
    )
    load_functions(ROOT / "vllm_ascend/attention/attention_v1.py", {"AscendFlashAttentionMetadata"}, scope)
    load_functions(ROOT / "vllm_ascend/attention/mla_v1.py", {"_build_full_flash_prefill"}, scope)
    builder = SimpleNamespace(
        flash_num_heads=12,
        _flash_attn_mask=None,
        device=torch.device("cpu"),
        decode_threshold=4,
        chunked_prefill_workspace_size=131072,
        _device_metadata_enabled=True,
        _device_metadata_tasks=(),
    )
    common = SimpleNamespace(
        num_reqs=5,
        num_actual_tokens=170,
        query_start_loc_cpu=torch.tensor([0, 4, 8, 41, 170, 170]),
        num_computed_prefill_tokens_cpu=torch.tensor([-999, -999, 129762, 0, -999]),
    )
    flash = scope["AscendFlashAttentionMetadata"](
        query=torch.empty(0, 12, 576),
        schedule=torch.empty(0),
        cu=torch.tensor([0, 4, 8, 41, 170, 170]),
        used_q=torch.tensor([4, 4, 33, 129, 0]),
        cache_lens=torch.tensor([100, 200, 129795, 129, 0]),
        block_table=torch.zeros((5, 1024), dtype=torch.int32),
        slots=torch.empty(176),
        live_boundaries=torch.empty(177),
        token_live=torch.empty(176),
        positions=torch.empty(176),
        attn_mask=None,
        max_query_len=129,
        max_seq_len=131072,
        is_prefill=True,
    )
    prefill, decode = scope["_build_full_flash_prefill"](builder, common, flash, 2, 8)
    assert prefill.plan.context_lengths == (129762, 0)
    assert prefill.plan.query_lengths == (33, 129)
    assert prefill.plan.workspace_tokens == 16384
    assert max(chunk.num_tokens for chunk in prefill.plan.chunks) <= 16384
    assert decode.query.shape == (8, 12, 576)
    assert decode.block_table.shape == (2, 1024)
    flash.used_q[:2].copy_(torch.tensor([2, 1]))
    flash.cache_lens[:2].copy_(torch.tensor([98, 197]))
    builder._device_metadata_tasks[-1].fn()
    assert decode.schedule.tolist() == [3, 295]
    assert prefill.plan.context_lengths == (129762, 0)


def test_full_prefill_avoids_absorbed_query_allocation_and_schedule():
    def unexpected_schedule(*args, **kwargs):
        raise AssertionError("full prefill should only build expanded chunk schedules")

    scope = dict(torch=torch, dataclass=dataclass, _flash_attention_schedule=unexpected_schedule)
    load_functions(
        ROOT / "vllm_ascend/attention/attention_v1.py",
        {"AscendFlashAttentionMetadata", "_build_flash_attention_metadata"},
        scope,
    )
    builder = SimpleNamespace(
        flash_num_heads=12,
        flash_num_kv_heads=1,
        flash_is_c8=False,
        flash_unabsorbed_prefill=True,
        decode_threshold=4,
        device=torch.device("cpu"),
        _flash_attn_mask=None,
        _flash_buffers={},
        kv_cache_spec=SimpleNamespace(dtype=torch.bfloat16, block_size=128),
        kernel_block_size=128,
        _device_metadata_enabled=False,
    )
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=15959,
        num_input_tokens=15960,
        block_table_tensor=torch.zeros((1, 128), dtype=torch.int32),
        causal=True,
        max_query_len=15959,
        query_start_loc=torch.tensor([0, 15959]),
        seq_lens=torch.tensor([15959]),
        slot_mapping=torch.arange(15960),
        positions=torch.arange(15960),
    )
    metadata = scope["_build_flash_attention_metadata"](
        builder,
        common,
        is_mla=True,
        allow_unabsorbed=False,
        metadata_only=True,
    )
    assert metadata.query.numel() == metadata.schedule.numel() == 0
    assert metadata.slots[-1].item() == -1
    assert metadata.token_live.sum().item() == 15959
