# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark physical KV sizes belong to draft groups, independently of offload."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_module
import vllm_ascend.worker.v2.spec_decode.dspark.speculator as dspark_module
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@pytest.mark.parametrize("physical_sizes", [(128, 128), (128, 384)])
@pytest.mark.parametrize("offload_enabled", [False, True])
def test_input_preparation_uses_each_draft_group(monkeypatch, physical_sizes, offload_enabled):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=8))
    spec.expert_offload_manager = Mock() if offload_enabled else None
    spec.draft_kv_cache_group_ids = [1, 2]
    spec.draft_attn_layer_names = set()
    spec.attn_groups = []
    spec._context_slot_mappings = torch.zeros(2, 4)
    monkeypatch.setattr(AscendDSparkSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    monkeypatch.setattr(DSparkSpeculator, "set_attn", lambda *_args: None)
    monkeypatch.setattr(dspark_module, "set_current_vllm_config", lambda *_args: nullcontext())
    backend = type("Backend", (), {})
    monkeypatch.setattr(dspark_module, "_get_graph_update_backend", lambda *_args: backend)
    monkeypatch.setattr(dspark_module, "AscendMLABackend", backend)
    monkeypatch.setattr(dspark_module, "get_layers_from_vllm_config", lambda *_args: {})
    monkeypatch.setattr(dspark_module.dflash_speculator, "prepare_dflash_inputs", None)
    prepare = Mock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs", prepare)
    tables = SimpleNamespace(
        input_block_tables=[torch.zeros(1, 4, dtype=torch.int32) for _ in range(3)],
        kernel_block_sizes=[16, 128, 128],
    )
    cache = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[], kv_cache_spec=SimpleNamespace(block_size=size))
            for size in (32, *physical_sizes)
        ]
    )

    spec.set_attn(None, cache, tables, None, None)

    # Later config refresh must not change the bound per-group physical sizes.
    spec.vllm_config.cache_config.block_size = 32

    for group, physical_size in enumerate(physical_sizes, start=1):
        inputs: list[torch.Tensor | int | None] = [None] * 28
        inputs[16] = tables.input_block_tables[group]
        inputs[17] = physical_size
        dspark_module.dflash_speculator.prepare_dflash_inputs(*inputs)
        assert prepare.call_args.args[16] is tables.input_block_tables[group]
        assert prepare.call_args.args[17] == tables.kernel_block_sizes[group]
        assert prepare.call_args.kwargs["kv_cache_block_size"] == physical_size
    assert prepare.call_count == 2


def test_group_mapping_rejects_unknown_table_before_preparation(monkeypatch):
    prepare = Mock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs", prepare)
    table = torch.zeros(1, 4, dtype=torch.int32)
    sizes = {id(table): 128}
    tables = SimpleNamespace(input_block_tables=[table], kernel_block_sizes=[32])
    wrapper = dflash_module.prepare_dflash_inputs_factory(sizes, tables)
    inputs: list[torch.Tensor | int | None] = [None] * 28
    inputs[16] = torch.zeros_like(table)
    with pytest.raises(RuntimeError, match="block table"):
        wrapper(*inputs)
    prepare.assert_not_called()


@pytest.mark.parametrize("as_keyword", [False, True])
def test_group_mapping_preserves_table_identity_for_keyword_and_positional_calls(monkeypatch, as_keyword):
    prepare = Mock()
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs", prepare)
    table = torch.zeros(1, 4, dtype=torch.int32)
    tables = SimpleNamespace(input_block_tables=[table], kernel_block_sizes=[128])
    wrapper = dflash_module.prepare_dflash_inputs_factory({id(table): 384}, tables)
    inputs: list[torch.Tensor | int | None] = [None] * 28
    inputs[16] = table
    inputs[17] = 384
    if as_keyword:
        names = list(inspect.signature(dflash_module.prepare_dflash_inputs_triton).parameters)[: len(inputs)]
        wrapper(**dict(zip(names, inputs)))
    else:
        wrapper(*inputs)
    assert prepare.call_args.kwargs["kv_cache_block_size"] == 384
