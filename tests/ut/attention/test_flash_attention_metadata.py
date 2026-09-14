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
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/attention_v1.py"
    selected = {"AscendFlashAttentionMetadata", "_build_flash_attention_metadata", "split_gqa_cache"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) in selected]
    scope = {
        "torch": torch,
        "dataclass": dataclass,
        "DeviceMetadataStage": SimpleNamespace(ATTENTION=2),
        "DeviceMetadataTask": lambda stage, run, group_id: SimpleNamespace(stage=stage, run=run, group_id=group_id),
    }
    exec(compile(tree, str(path), "exec"), scope)
    calls = []

    def metadata(cu, lengths, used, kv_heads, kwargs):
        calls.append((cu.clone(), lengths.clone(), used.clone(), kwargs))
        words = (((36 + 72) * lengths.shape[0] * kv_heads + 1) * 16 + 4095) // 4096 * 4096
        return torch.full((words,), len(calls), dtype=torch.int32)

    def mla_metadata(lengths, heads, kv_heads, **kwargs):
        return metadata(kwargs["cu_seqlens_q"], lengths, kwargs["seqused_q"], kv_heads, kwargs)

    def gqa_metadata(heads, kv_heads, dim, **kwargs):
        return metadata(kwargs["cu_seqlens_q"], kwargs["seqused_kv"], kwargs["seqused_q"], kv_heads, kwargs)

    monkeypatch.setattr(torch.ops._C_ascend, "flash_mla_with_kvcache_metadata", mla_metadata, raising=False)
    package = ModuleType("cann_ops_transformer")
    module = ModuleType("cann_ops_transformer.ops")
    module.flash_attn_metadata = gqa_metadata  # type: ignore[attr-defined]
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
        common.causal = False
        again = module._build_flash_attention_metadata(builder, common, is_mla=is_mla)
        assert again is flash
        builder._device_metadata_tasks[0].run()
        assert flash.cache_lens.tolist() == [*lengths, 0]
        assert calls[-1][3]["mask_mode"] == 0
        assert pointers == {
            name: tensor.data_ptr() for name, tensor in vars(flash).items() if isinstance(tensor, torch.Tensor)
        }
    assert len(calls) == 3


def test_metadata_fallback_is_eager_when_executor_is_disabled(helpers):
    module, calls = helpers
    builder = make_builder()
    builder._device_metadata_enabled = False
    module._build_flash_attention_metadata(builder, make_common(), is_mla=True)
    assert len(calls) == 1
    assert builder._device_metadata_tasks == ()


def test_gqa_alias_keeps_page_stride_and_storage_offset(helpers):
    module, _ = helpers
    storage = torch.zeros(7, 3, 2, 16, 64)
    cache = storage[:, 1]
    key, value = module.split_gqa_cache(cache, 1)
    assert key.stride(0) == value.stride(0) == cache.stride(0)
    assert key.storage_offset() == cache.storage_offset()
    assert value.storage_offset() == cache.storage_offset() + 16 * 64
    key[2, 0, 3, 4] = 17
    value[4, 0, 5, 6] = 19
    assert storage[2, 1, 0, 3, 4] == 17
    assert storage[4, 1, 1, 5, 6] == 19
