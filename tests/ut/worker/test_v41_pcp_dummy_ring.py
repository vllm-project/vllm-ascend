# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression checks for the runner's PCP-local ring-page contract.

Execute the production methods with only the upstream runner and cache-spec
interfaces substituted, so these checks do not require an NPU installation.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def runner_type():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    methods = [node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "prepare_dummy_attn"]

    class UpstreamRunner:
        def prepare_dummy_attn(self, input_batch, **kwargs):
            provider = self.pcp_manager or self.block_tables
            return provider.get_dummy_block_tables(input_batch.num_reqs), self.current_slots

    scope = {
        "torch": torch,
        "AscendInputBatch": object,
        "GPUModelRunner": UpstreamRunner,
        "vllm_version_is": lambda version: False,
        "is_circular_kv_cache_spec": lambda spec: spec.is_circular,
        "ring_state_update_skipped": lambda: scope["skip_ring"],
        "skip_ring": False,
    }
    helpers_path = path.with_name("utils.py")
    helper = next(
        node
        for node in ast.parse(helpers_path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_v41_dummy_ring_state"
    )
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(helpers_path), "exec"), scope)
    selected = ast.ClassDef(
        name="NPUModelRunner",
        bases=[ast.Name(id="GPUModelRunner", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["NPUModelRunner"], scope


def make_runner(runner_type, pcp):
    cls, scope = runner_type
    runner = cls()
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=5,
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(is_circular=False), layer_names=["swa"]),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(is_circular=True), layer_names=["ring"]),
        ],
    )
    runner.current_tables = (torch.full((3, 2), 17), torch.zeros((3, 2), dtype=torch.int64))
    runner.current_slots = torch.full((2, 8), -1, dtype=torch.int64)
    runner.block_tables = SimpleNamespace(input_block_tables=(torch.full((3, 2), 99), torch.full((3, 2), 99)))
    ring = torch.full((5, 4), 7.0)
    runner.compilation_config = SimpleNamespace(static_forward_context={"ring": SimpleNamespace(kv_cache=[ring])})
    runner.pcp_manager = SimpleNamespace(get_dummy_block_tables=lambda num_reqs: runner.current_tables) if pcp else None
    runner.block_tables.get_dummy_block_tables = lambda num_reqs: runner.current_tables
    return runner, scope, ring


@pytest.mark.parametrize("pcp", [False, True])
def test_dummy_ring_prepares_returned_table_instead_of_global_storage(runner_type, pcp):
    runner, _, ring = make_runner(runner_type, pcp)
    tables, slots = runner.prepare_dummy_attn(SimpleNamespace(num_reqs=2))

    assert tables is runner.current_tables
    assert slots is runner.current_slots
    assert tables[1][:, 0].tolist() == [1, 2, 0]
    assert torch.all(tables[1][:, 1] == 0)
    assert torch.all(tables[0] == 17)
    assert all(torch.all(table == 99) for table in runner.block_tables.input_block_tables)
    assert torch.count_nonzero(ring[1:3]) == 0
    assert torch.all(ring[[0, 3, 4]] == 7)


def test_idle_pcp_dummy_preserves_live_ring_pages(runner_type):
    runner, scope, ring = make_runner(runner_type, pcp=True)
    scope["skip_ring"] = True
    runner.prepare_dummy_attn(SimpleNamespace(num_reqs=2))

    assert torch.count_nonzero(runner.current_tables[1]) == 0
    assert torch.all(ring == 7)
    assert all(torch.all(table == 99) for table in runner.block_tables.input_block_tables)


def test_insufficient_ring_pages_fails_before_preparing_pcp_views(runner_type):
    runner, _, ring = make_runner(runner_type, pcp=True)
    runner.kv_cache_config.num_blocks = 2
    with pytest.raises(ValueError, match="Insufficient ring pages"):
        runner.prepare_dummy_attn(SimpleNamespace(num_reqs=2))

    assert torch.count_nonzero(runner.current_tables[1]) == 0
    assert torch.all(ring == 7)


def test_empty_pcp_dummy_has_no_ring_writes(runner_type):
    runner, _, ring = make_runner(runner_type, pcp=True)
    runner.prepare_dummy_attn(SimpleNamespace(num_reqs=0))

    assert torch.count_nonzero(runner.current_tables[1]) == 0
    assert torch.all(ring == 7)
