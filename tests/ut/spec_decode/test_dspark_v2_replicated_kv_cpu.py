# SPDX-License-Identifier: Apache-2.0
"""Check MRv2 draft table ownership and physical lane expansion on CPU."""

import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch


def test_draft_tables_expand_lanes_without_mutating_target():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    tree = ast.parse((root / "worker/v2/spec_decode/dspark/speculator.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"set_attn", "_refresh_replicated_block_tables"}
    ]
    helper_tree = ast.parse((root / "attention/context_parallel/common_cp.py").read_text(encoding="utf-8"))
    helper = next(
        node
        for node in helper_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "expand_dcp_replicated_block_table"
    )

    class ReplicatedSpec:
        block_size = 128
        dcp_replication_size = 2

    class Parent:
        def set_attn(self, *args):
            self._context_slot_mappings = torch.zeros((1, 16), dtype=torch.int64)
            self.draft_kv_cache_group_ids = [1]
            self.attn_groups = [[], [SimpleNamespace(kv_cache_spec=ReplicatedSpec())]]

    namespace = {
        "torch": torch,
        "copy": copy,
        "cast": cast,
        "Any": Any,
        "AttentionLayerBase": object,
        "DSparkSpeculator": Parent,
        "AscendDCPReplicatedDraftAttentionSpec": ReplicatedSpec,
        "set_current_vllm_config": lambda config: nullcontext(),
        "get_layers_from_vllm_config": lambda config, kind, names: {
            name: SimpleNamespace(get_attn_backend=lambda: object) for name in names
        },
    }
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(body=[helper, cls], type_ignores=[]))
    exec(compile(source, "mrv2_replicated_tables", "exec"), namespace)
    speculator = namespace[cls.name]()
    speculator.attn_vllm_config = speculator.vllm_config = object()
    speculator._draft_dcp_context = nullcontext
    speculator.draft_attn_layer_names = {"draft"}
    speculator.replicated_draft_kv = True
    speculator.device = "cpu"
    target_tables = SimpleNamespace(
        input_block_tables=[torch.tensor([[1, 4], [3, 7]]), torch.tensor([[2, 5], [8, 9]])],
        slot_mappings=torch.full((2, 16), 777, dtype=torch.int64),
        kernel_block_sizes=[128, 128],
        cp_size=2,
        cp_rank=1,
        cp_interleave=128,
    )
    config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["target"]), SimpleNamespace(layer_names=["draft"])]
    )
    speculator.set_attn(None, config, target_tables, None, None)
    draft_table = speculator.block_tables.input_block_tables[1]
    address = draft_table.data_ptr()
    batch = SimpleNamespace(num_reqs=2, seq_lens=torch.tensor([129, 500]))
    speculator._refresh_replicated_block_tables(batch)
    torch.testing.assert_close(draft_table, torch.tensor([[4, 5, 10, 11], [16, 17, 18, 19]]))
    assert speculator.block_tables.cp_size == 1 and target_tables.cp_size == 2
    assert torch.all(target_tables.slot_mappings == 777)
    assert torch.all(speculator.block_tables.slot_mappings == -1)
    torch.testing.assert_close(target_tables.input_block_tables[1], torch.tensor([[2, 5], [8, 9]]))

    batch.num_reqs = 1
    speculator._refresh_replicated_block_tables(batch)
    assert draft_table.data_ptr() == address
    assert torch.all(draft_table[1] == 0)
