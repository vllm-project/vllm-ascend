# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the graph-external FA/MLA metadata path and shared cache views."""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def helpers(monkeypatch):
    # Execute the real helpers without importing unrelated NPU/worker modules.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/flash_metadata.py"
    selected = {
        "AscendFlashAttentionMetadata",
        "_flash_attention_schedule",
        "_build_flash_attention_metadata",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in selected]
    scope = {
        "torch": torch,
        "dataclass": dataclass,
        "DeviceMetadataStage": SimpleNamespace(ATTENTION=2),
        "DeviceMetadataTask": lambda stage, run, group_id: SimpleNamespace(stage=stage, run=run, group_id=group_id),
        "cdiv": lambda a, b: (a + b - 1) // b,
    }
    exec(compile(tree, str(path), "exec"), scope)

    def local_lengths(lengths, *, dcp_size, dcp_rank, cp_kv_cache_interleave_size):
        block = cp_kv_cache_interleave_size
        base = lengths // (block * dcp_size) * block
        return base + (lengths % (block * dcp_size) - dcp_rank * block).clamp(0, block)

    scope["get_dcp_local_seq_lens"] = local_lengths
    calls = []

    def metadata(cu, lengths, used, kv_heads, kwargs):
        words = (((36 + 72) * lengths.shape[0] * kv_heads + 1) * 16 + 4095) // 4096 * 4096
        if lengths.device.type == "meta":
            return torch.empty(words, dtype=torch.int32, device="meta")
        calls.append((cu.clone(), lengths.clone(), used.clone(), kwargs))
        return torch.full((words,), len(calls), dtype=torch.int32)

    def mla_metadata(lengths, heads, kv_heads, **kwargs):
        return metadata(kwargs["cu_seqlens_q"], lengths, kwargs["seqused_q"], kv_heads, kwargs)

    def gqa_metadata(heads, kv_heads, dim, **kwargs):
        return metadata(kwargs["cu_seqlens_q"], kwargs["seqused_kv"], kwargs["seqused_q"], kv_heads, kwargs)

    monkeypatch.setattr(torch.ops._C_ascend, "flash_mla_with_kvcache_metadata", mla_metadata, raising=False)
    package = ModuleType("cann_ops_transformer")
    module = ModuleType("cann_ops_transformer.ops")
    module.flash_attn_metadata = gqa_metadata
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return SimpleNamespace(**scope), calls


def make_builder():
    return SimpleNamespace(
        device=torch.device("cpu"),
        kv_cache_spec=SimpleNamespace(dtype=torch.float32, block_size=16),
        kernel_block_size=16,
        flash_num_heads=2,
        flash_num_kv_heads=1,
        decode_threshold=4,
        _flash_buffers={},
        _flash_attn_mask=torch.zeros(8, 8, dtype=torch.int8),
        _device_metadata_enabled=True,
        _device_metadata_tasks=(),
    )


def make_common():
    return SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=6,
        num_input_tokens=8,
        max_query_len=2,
        block_table_tensor=torch.tensor([[2, 3], [0, 0], [4, 5]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        seq_lens=torch.tensor([7, 0, 9], dtype=torch.int32),
        # A deliberately invalid host mirror must never affect visibility.
        _seq_lens_cpu=torch.tensor([999, 999, 999]),
        slot_mapping=torch.arange(8),
        positions=torch.arange(8),
        causal=True,
    )


@pytest.mark.parametrize("is_mla", [True, False])
def test_metadata_task_reads_device_values_at_submission_and_reuses_buffers(helpers, is_mla):
    module, calls = helpers
    builder, common = make_builder(), make_common()
    flash = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
    assert calls == []  # AICPU metadata has not run during build/captured forward.
    task = builder._device_metadata_tasks[0]
    assert task.group_id == id(flash.schedule)
    common.seq_lens[0] = 6  # Simulate async rejection correction before submission.
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host readback")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host list")),
    ):
        task.run()
    assert flash.cache_lens.tolist() == [6, 0, 9, 0]
    assert flash.cu.tolist() == [0, 2, 4, 6, 8]
    assert flash.used_q.tolist() == [2, 0, 2, 0]
    assert flash.slots.tolist() == [0, 1, -1, -1, 4, 5, -1, -1]
    assert flash.token_live.tolist() == [True, True, False, False, True, True, False, False]
    pointers = {name: tensor.data_ptr() for name, tensor in vars(flash).items() if isinstance(tensor, torch.Tensor)}

    # Active -> idle -> active must regenerate scheduling in the same storage.
    for lengths in ([0, 0, 0], [8, 2, 10]):
        common.seq_lens.copy_(torch.tensor(lengths))
        again = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
        assert again is flash
        builder._device_metadata_tasks[0].run()
        assert flash.cache_lens.tolist() == [*lengths, 0]
        assert calls[-1][3]["mask_mode"] == 3
        assert pointers == {
            name: tensor.data_ptr() for name, tensor in vars(flash).items() if isinstance(tensor, torch.Tensor)
        }
    assert len(calls) == 3


@pytest.mark.parametrize("is_mla", [True, False])
def test_causal_and_noncausal_graphs_use_distinct_schedules(helpers, is_mla):
    module, calls = helpers
    builder, common = make_builder(), make_common()
    causal = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
    causal_task = builder._device_metadata_tasks[0]
    common.causal = False
    noncausal = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
    noncausal_task = builder._device_metadata_tasks[0]
    causal_task.run()
    noncausal_task.run()
    assert causal.schedule.data_ptr() != noncausal.schedule.data_ptr()
    assert [call[-1]["mask_mode"] for call in calls] == [3, 0]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("is_mla", [True, False])
def test_dcp_visibility_and_replicated_current_pages(helpers, causal, is_mla):
    module, calls = helpers
    builder, common = make_builder(), make_common()
    builder.dcp_size, builder.dcp_rank = 4, 3
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=4))
    common.causal = causal
    common.seq_lens.copy_(torch.tensor([35, 0, 5]))
    flash = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host readback")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host list")),
    ):
        builder._device_metadata_tasks[0].run()
    # Rank3 owns global [12:16], [28:32], ...; the short row has no KV there.
    assert flash.cache_lens.tolist() == [8, 0, 0, 0]
    assert calls[0][-1]["mask_mode"] == 0
    assert flash.query.shape == (8, 8, 576 if is_mla else 64)
    assert len(calls) == (2 if causal else 1)
    if causal:
        assert calls[1][1].tolist() == [2, 0, 2, 0]
        assert calls[1][-1]["mask_mode"] == 3
    if causal and is_mla:
        assert flash.current_slots.tolist() == [0, 1, -1, -1, 128, 129, -1, -1]
        assert flash.current_block_table[:, 0].tolist() == [0, 0, 1, 0]


def test_unabsorbed_prefill_keeps_compressed_history_lengths(helpers):
    module, calls = helpers
    builder, common = make_builder(), make_common()
    builder.flash_unabsorbed_prefill = True
    common.max_query_len = 8
    flash = module._build_flash_attention_metadata(builder, common, is_mla=True)
    builder._device_metadata_tasks[0].run()
    assert flash.unabsorbed and flash.split_kv
    assert flash.current_cache is None
    assert calls[0][1].tolist() == [5, 0, 7, 0]
    assert calls[1][-1]["head_dim_v"] == 128
    assert calls[1][-1]["layout_kv"] == "TND"


@pytest.mark.parametrize("dcp_size", [1, 8])
def test_c8_prefill_separates_quantized_history_and_bf16_current(helpers, dcp_size):
    module, calls = helpers
    builder, common = make_builder(), make_common()
    builder.flash_is_c8 = True
    builder.flash_unabsorbed_prefill = True
    builder.kv_cache_spec.dtype = torch.float8_e4m3fn
    builder.kernel_block_size = 128
    builder.dcp_size, builder.dcp_rank = dcp_size, 0
    builder.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=4),
    )
    common.max_query_len = 8
    common.seq_lens.copy_(torch.tensor([132, 0, 138]))
    flash = module._build_flash_attention_metadata(builder, common, is_mla=True)
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host readback")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host list")),
    ):
        builder._device_metadata_tasks[0].run()
    assert flash.is_c8 and flash.split_kv and not flash.unabsorbed
    assert flash.query.dtype == flash.current_cache.dtype == torch.bfloat16
    assert flash.current_cache.shape[2:] == (128, 576)
    assert calls[0][-1]["is_c8"] is True
    assert calls[0][-1]["head_dim_qk"] == 576
    assert calls[0][-1]["mask_mode"] == 0
    assert "is_c8" not in calls[1][-1]
    assert calls[1][-1]["mask_mode"] == 3
    assert calls[1][1].tolist() == [2, 0, 2, 0]
    expected = [130, 0, 136, 0] if dcp_size == 1 else [18, 0, 20, 0]
    assert calls[0][1].tolist() == expected


@pytest.mark.parametrize("filename,classname", [("mla_cp.py", "AscendMlaDCPImpl")])
def test_dcp_flash_graph_does_not_touch_fia_host_parameters(filename, classname):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    cls.bases = []
    cls.body = [node for node in cls.body if getattr(node, "name", None) == "update_graph_params"]
    tree.body = [cls]
    scope = {"envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True)}
    exec(compile(tree, str(path), "exec"), scope)
    # No legacy FIA graph registry/forward context is initialized on this path.
    scope[classname].update_graph_params(None, None, 16)


def test_metadata_fallback_is_eager_when_executor_is_disabled(helpers):
    module, calls = helpers
    builder = make_builder()
    builder._device_metadata_enabled = False
    module._build_flash_attention_metadata(builder, make_common(), is_mla=True)
    assert len(calls) == 1
    assert builder._device_metadata_tasks == ()


@pytest.mark.parametrize("is_mla", [True, False])
def test_eager_prefill_shapes_do_not_accumulate_graph_buffers(helpers, is_mla):
    module, _ = helpers
    builder, common = make_builder(), make_common()
    decode = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
    for tokens in (16, 24, 32):
        common.max_query_len = tokens
        common.num_input_tokens = tokens
        flash = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
        assert flash.query.shape[0] == tokens
        assert len(builder._flash_buffers) == 1
    common = make_common()
    assert module._build_flash_attention_metadata(builder, common, is_mla=is_mla) is decode


@pytest.mark.parametrize("is_mla", [True, False])
def test_large_noncausal_draft_queries_keep_graph_buffers(helpers, is_mla):
    module, _ = helpers
    builder, common = make_builder(), make_common()
    common.causal = False
    common.max_query_len = common.num_input_tokens = 32
    flash = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
    assert module._build_flash_attention_metadata(builder, common, is_mla=is_mla) is flash
