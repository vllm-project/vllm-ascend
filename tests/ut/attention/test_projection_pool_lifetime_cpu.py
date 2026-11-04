# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Standard-library CPU reproduction for persistent attention scratch pools.

Run: python tests/ut/attention/test_projection_pool_lifetime_cpu.py

Execute the checkout's actual pool initializer and allocation method. Only the
allocator is mocked; Python reference ownership and garbage collection are real.
"""

from __future__ import annotations

import ast
import gc
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from weakref import WeakValueDictionary

ROOT = Path(__file__).resolve().parents[3]
BUFFER_NAMES = ("all_gather_input", "all_gather_output", "reduce_scatter_input", "reduce_scatter_output")


class Allocation:
    def __init__(self, shape, *, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device


def execute_nodes(nodes, path, namespace):
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def load_pool(relative_path, class_name):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    initializer = next(
        n
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "o_proj_full_pools"
    )
    return execute_nodes([initializer], path, {"WeakValueDictionary": WeakValueDictionary})["o_proj_full_pools"]


def load_allocator():
    path = ROOT / "vllm_ascend/weight_switch/o_proj.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "ParallelOProjection"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_exchange_buffers")
    allocator = Mock(side_effect=Allocation)
    function = execute_nodes([method], path, {"torch": SimpleNamespace(empty=allocator)})[method.name]
    return function, allocator


class TestProjectionPoolLifetime(unittest.TestCase):
    def test_buffers_share_while_live_and_release_after_both_owners_are_gone(self):
        cases = (
            ("vllm_ascend/attention/context_parallel/sfa_cp.py", "AscendSFAPCPImpl", "sfa_pcp_o_proj"),
            ("vllm_ascend/attention/context_parallel/dsa_cp.py", "AscendDSAPCPImpl", "dsa_pcp_o_proj"),
        )
        for relative_path, class_name, pool_key in cases:
            with (
                self.subTest(backend=class_name),
                patch.dict("sys.modules", {"torch_npu": MagicMock(), "acl": MagicMock()}),
            ):
                pool = load_pool(relative_path, class_name)
                allocate, allocator = load_allocator()
                config = SimpleNamespace(
                    model_config=SimpleNamespace(dtype="bf16"),
                    scheduler_config=SimpleNamespace(max_num_seqs=8),
                    num_speculative_tokens=0,
                    compilation_config=SimpleNamespace(max_cudagraph_capture_size=8, cudagraph_capture_sizes=[8]),
                )

                def new_projection(pool=pool, pool_key=pool_key, allocate=allocate, config=config):
                    projection = SimpleNamespace(
                        config=SimpleNamespace(world_size=2, group=SimpleNamespace(unique_name="pcp:0")),
                        o_proj=SimpleNamespace(weight=SimpleNamespace(device="cpu")),
                        input_size=16,
                        output_size=32,
                        o_proj_full_pools=pool,
                        o_proj_weight_switch_pool_key=pool_key,
                    )
                    allocate(projection, config)
                    return projection

                first = new_projection()
                second = new_projection()
                self.assertEqual(allocator.call_count, 4)
                for name in BUFFER_NAMES:
                    self.assertIs(getattr(first, name), getattr(second, name))
                refs = [weakref.ref(getattr(first, name)) for name in BUFFER_NAMES]
                del first
                gc.collect()
                self.assertTrue(all(ref() is not None for ref in refs))
                self.assertEqual(len(pool), 4)
                del second
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs), "Class pool retains all dead projections' scratch")
                self.assertEqual(len(pool), 0)


if __name__ == "__main__":
    unittest.main()
