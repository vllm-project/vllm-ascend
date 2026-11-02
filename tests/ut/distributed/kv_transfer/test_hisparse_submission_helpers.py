# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for descriptor order, current values and allocation lifetimes.

These tests do not qualify NPU kernels, graph replay or model performance.
"""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend"


def load_module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VIEWS = load_module("patch/worker/patch_hisparse_block_table_views.py", "hisparse_test_views")
PAGES = load_module("distributed/kv_transfer/hisparse/page_dma_descriptors.py", "hisparse_test_pages")


@pytest.mark.parametrize("rows", [0, 1, 4, 8])
def test_prefix_views_alias_current_values_without_recreating_views(rows):
    tables = [torch.arange(48).reshape(8, 6), torch.full((8, 6), -1)]
    owner = SimpleNamespace(input_block_tables=tables, max_num_reqs=8)
    first = VIEWS.current_table_views(owner, tables, rows)
    tables[0].fill_(71)
    tables[1].fill_(93)
    second = VIEWS.current_table_views(owner, tables, rows)
    assert second is first
    for view, table in zip(second, tables, strict=True):
        assert torch.equal(view, table[:rows])
        assert view.untyped_storage().data_ptr() == table.untyped_storage().data_ptr()
    assert len(owner._ascend_hisparse_prefix_views[1]) == 1


def test_rebound_or_custom_tables_do_not_reuse_old_allocation_views():
    original = [torch.ones((4, 2))]
    owner = SimpleNamespace(input_block_tables=original, max_num_reqs=4)
    old = VIEWS.current_table_views(owner, original, 2)
    replacement = [torch.zeros((4, 2))]
    custom = VIEWS.current_table_views(owner, replacement, 2)
    owner.input_block_tables = replacement
    rebound = VIEWS.current_table_views(owner, replacement, 2)
    assert torch.equal(custom[0], replacement[0][:2])
    assert torch.equal(rebound[0], replacement[0][:2])
    assert owner._ascend_hisparse_prefix_views[0][0] is original[0]
    assert torch.equal(old[0], torch.ones((2, 2)))


def test_page_descriptors_match_scalar_oracle_and_preserve_duplicate_transfer_order():
    sources = [torch.zeros((9, 4, 7), dtype=torch.uint8) for _ in range(3)]
    hosts = [torch.zeros((40, 7), dtype=torch.uint8) for _ in range(3)]
    indices = [2, 0, 1]
    submitted = []

    def acquire(count):
        return SimpleNamespace(**{name: np.empty(count, dtype=np.int64) for name in ("src_np", "dst_np", "sizes_np")})

    def submit(descriptors, count, *, transfer_ids):
        submitted.append((descriptors, count, transfer_ids))

    worker = SimpleNamespace(
        cache_handles=[SimpleNamespace(runtime=SimpleNamespace(resident_source_index=i)) for i in indices],
        resident_caches=sources,
        host_caches=hosts,
        kernel_block_size=4,
        is_host_writer=True,
        _acquire_dma_descriptors=acquire,
        _submit_dma_descriptors=submit,
    )
    transfers = [
        SimpleNamespace(resident_block_ids=[3, 5, 2], host_block_id=6, transfer_id=42),
        SimpleNamespace(resident_block_ids=[1, 4, 7], host_block_id=2, transfer_id=19),
    ]
    transfers.append(transfers[0])
    assert PAGES.enqueue_transfers(worker, transfers)
    descriptors, count, transfer_ids = submitted[0]
    assert count == 9 and transfer_ids == (42, 19, 42)
    expected = []
    for transfer in transfers:
        for index, source, host in zip(indices, sources, hosts, strict=True):
            expected.append(
                (
                    source.data_ptr() + transfer.resident_block_ids[index] * source.stride(0),
                    host.data_ptr() + transfer.host_block_id * 4 * 7,
                    4 * 7,
                )
            )
    np.testing.assert_array_equal(
        np.stack((descriptors.src_np, descriptors.dst_np, descriptors.sizes_np), axis=1),
        expected,
    )
    sources[0] = torch.zeros_like(sources[0])
    with pytest.raises(RuntimeError, match="storage changed"):
        PAGES.enqueue_transfers(worker, transfers)
    assert len(submitted) == 1


def resident_functions():
    path = ROOT / "distributed/kv_transfer/hisparse/resident_updates.py"
    tree = ast.parse(path.read_bytes())
    wanted = {"pack_current_updates", "unique_destinations", "eligible_update_matrix"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_resident_matrix_uses_current_request_order_and_page_values():
    functions = resident_functions()
    updates = {
        "b": SimpleNamespace(pages=[3, 1], block_ids=[[7, 8], [9, 10]]),
        "a": SimpleNamespace(pages=[0], block_ids=[[11], [12]]),
    }
    actual = functions["pack_current_updates"](2, updates, ["a", "b"])
    np.testing.assert_array_equal(actual, [[1, 3, 7, 9], [1, 1, 8, 10], [0, 0, 11, 12]])
    assert functions["eligible_update_matrix"](actual, ["a", "b"], 2, 4)
    updates["b"].block_ids[0][0] = 73
    new = functions["pack_current_updates"](2, updates, ["b", "a"])
    assert new[0].tolist() == [0, 3, 73, 9]
    assert actual[0].tolist() == [1, 3, 7, 9]
    duplicate = np.concatenate((new, new[:1]))
    assert not functions["unique_destinations"](duplicate)
    assert not functions["eligible_update_matrix"](duplicate, ["b", "a"], 2, 4)
    assert not functions["eligible_update_matrix"](new, ["b", "b"], 2, 4)
