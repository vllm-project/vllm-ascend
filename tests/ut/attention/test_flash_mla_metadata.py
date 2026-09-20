# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for external FlashMLA metadata buffer management."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch


@pytest.fixture
def helpers():
    """Load just the production helper, without importing the NPU runtime."""
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/attention_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = {
        "AscendFlashAttentionMetadata",
        "_build_flash_attention_metadata",
    }
    tree.body = [node for node in tree.body if getattr(node, "name", None) in selected]
    calls = []

    def metadata(cache_lens, num_heads, num_kv_heads, **kwargs):
        del num_heads, num_kv_heads
        calls.append((cache_lens.clone(), kwargs))
        rows = cache_lens.shape[0]
        words = ((36 + 72) * rows + 1) * 16
        words = ((words + 4095) // 4096) * 4096
        return torch.full((words,), len(calls), dtype=torch.int32)

    scope = {
        "torch": torch,
        "dataclass": dataclass,
        "replace": replace,
        "Any": Any,
        "DeviceMetadataStage": SimpleNamespace(ATTENTION=2),
        "DeviceMetadataTask": lambda stage, run, group_id: SimpleNamespace(
            stage=stage, run=run, group_id=group_id
        ),
        "flash_mla_with_kvcache_metadata": metadata,
    }
    # The contract lives in flash_mla.py, not attention_v1.py. Load its real
    # definition before compiling the metadata helpers, without NPU imports.
    contract_path = path.with_name("flash_mla.py")
    contract_tree = ast.parse(contract_path.read_text(encoding="utf-8"))
    contract_tree.body = [
        node for node in contract_tree.body if getattr(node, "name", None) == "FlashMLAContract"
    ]
    exec(compile(contract_tree, str(contract_path), "exec"), scope)
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**scope), calls


def _builder():
    return SimpleNamespace(
        device=torch.device("cpu"),
        kv_cache_spec=SimpleNamespace(dtype=torch.float32, block_size=16),
        kernel_block_size=16,
        flash_num_heads=2,
        decode_threshold=4,
        _flash_buffers={},
        _flash_attn_mask=torch.zeros(8, 8, dtype=torch.int8),
        _device_metadata_enabled=True,
        _device_metadata_tasks=(),
    )


def _common():
    return SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=6,
        num_input_tokens=8,
        max_query_len=2,
        block_table_tensor=torch.tensor([[2, 3], [0, 0], [4, 5]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        seq_lens=torch.tensor([7, 0, 9], dtype=torch.int32),
        slot_mapping=torch.arange(8),
        positions=torch.arange(8),
        causal=True,
    )


def test_metadata_task_uses_device_lengths_and_reuses_buffer(helpers):
    module, calls = helpers
    builder, common = _builder(), _common()
    flash = module._build_flash_attention_metadata(builder, common)
    assert calls == []
    task = builder._device_metadata_tasks[0]
    assert task.group_id == id(flash.schedule)
    assert flash.contract.num_heads_q == builder.flash_num_heads
    assert flash.contract.num_heads_kv == 1
    assert flash.contract.head_dim_qk == 576
    assert flash.contract.head_dim_v == 512
    assert flash.contract.layout_kv == "PA_BBND"

    # The task sees a correction made after the builder ran, without a CPU
    # mirror/readback. This is the async-speculative-decode contract.
    common.seq_lens[0] = 6
    with (
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host readback")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar")),
        patch.object(torch.Tensor, "tolist", side_effect=AssertionError("host list")),
    ):
        task.run()

    assert flash.cache_lens.tolist() == [6, 0, 9, 0]
    assert flash.used_q.tolist() == [2, 0, 2, 0]
    assert flash.slots.tolist() == [0, 1, -1, -1, 4, 5, -1, -1]
    assert flash.token_live.tolist() == [True, True, False, False, True, True, False, False]
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in vars(flash).items()
        if isinstance(tensor, torch.Tensor)
    }

    common.seq_lens.copy_(torch.tensor([8, 2, 10]))
    common.causal = False
    assert module._build_flash_attention_metadata(builder, common) is flash
    builder._device_metadata_tasks[0].run()
    assert flash.cache_lens.tolist() == [8, 2, 10, 0]
    assert calls[-1][1]["mask_mode"] == 0
    assert calls[-1][1]["head_dim_qk"] == 576
    assert calls[-1][1]["head_dim_v"] == 512
    assert calls[-1][1]["layout_q"] == "TND"
    assert pointers == {
        name: tensor.data_ptr()
        for name, tensor in vars(flash).items()
        if isinstance(tensor, torch.Tensor)
    }


def test_metadata_falls_back_to_eager_without_executor(helpers):
    module, calls = helpers
    builder = _builder()
    builder._device_metadata_enabled = False
    module._build_flash_attention_metadata(builder, _common())
    assert len(calls) == 1
    assert builder._device_metadata_tasks == ()


def test_contract_uses_pa_bbnd_layout(helpers):
    module, _ = helpers
    contract = module.FlashMLAContract(num_heads_q=8, max_seqlen_q=16, max_seqlen_kv=128)
    metadata_kwargs = contract.metadata_kwargs(
        torch.zeros(2, dtype=torch.int32), torch.ones(2, dtype=torch.int32)
    )
    assert metadata_kwargs["head_dim_qk"] == 576
    assert contract.attention_kwargs(
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        cache_seqlens=torch.ones(2, dtype=torch.int32),
        cu_seqlens_q=torch.zeros(3, dtype=torch.int32),
        seqused_q=torch.ones(2, dtype=torch.int32),
        attn_mask=None,
        metadata=torch.zeros(4096, dtype=torch.int32),
    )["layout_kv"] == "PA_BBND"
