from typing import Callable
from unittest.mock import MagicMock

from tests.ut.base import TestBase
from vllm_ascend import ops

ops.register_dummy_fusion_op()

import torch

from vllm.compilation.backends import VllmBackend
from vllm.config import VllmConfig
from vllm_ascend.compilation.piecewise_backend import (ConcreteSizeEntry,
                                                       NPUPiecewiseBackend)


class TestConcreteSizeEntry(TestBase):

    def test_init(self):
        entry = ConcreteSizeEntry(runtime_shape=10,
                                  need_to_compile=True,
                                  use_aclgraph=False,
                                  compiled=True,
                                  runnable=lambda: None,
                                  num_finished_warmup=5,
                                  aclgraph=None,
                                  output="some_output",
                                  input_addresses=[12345, 67890])
        self.assertEqual(entry.runtime_shape, 10)
        self.assertTrue(entry.need_to_compile)
        self.assertFalse(entry.use_aclgraph)
        self.assertTrue(entry.compiled)
        self.assertEqual(entry.num_finished_warmup, 5)
        self.assertIsNone(entry.aclgraph)
        self.assertEqual(entry.output, "some_output")
        self.assertEqual(entry.input_addresses, [12345, 67890])


class TestNPUPiecewiseBackend(TestBase):

    def setUp(self):
        self.graph = MagicMock(spec=torch.fx.GraphModule)
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.compilation_config = MagicMock()
        self.vllm_config.compilation_config.compile_sizes = [10, 20]
        self.vllm_config.compilation_config.cudagraph_capture_sizes = [10]
        self.vllm_config.compilation_config.use_cudagraph = True
        self.vllm_config.compilation_config.cudagraph_num_of_warmups = 2
        self.vllm_config.compilation_config.inductor_compile_config = MagicMock(
        )
        self.graph_pool = MagicMock()
        self.piecewise_compile_index = 0
        self.total_piecewise_compiles = 2
        self.sym_shape_indices = [0]
        self.compiled_graph_for_general_shape = MagicMock(spec=Callable)
        self.vllm_backend = MagicMock(spec=VllmBackend)
        self.vllm_backend.compiler_manager = MagicMock()

    def test_init(self):
        backend = NPUPiecewiseBackend(
            self.graph, self.vllm_config, self.graph_pool,
            self.piecewise_compile_index, self.total_piecewise_compiles,
            self.sym_shape_indices, self.compiled_graph_for_general_shape,
            self.vllm_backend)
        self.assertEqual(backend.graph, self.graph)
        self.assertEqual(backend.vllm_config, self.vllm_config)
        self.assertEqual(backend.graph_pool, self.graph_pool)
        self.assertEqual(backend.piecewise_compile_index,
                         self.piecewise_compile_index)
        self.assertEqual(backend.total_piecewise_compiles,
                         self.total_piecewise_compiles)
        self.assertEqual(backend.sym_shape_indices, self.sym_shape_indices)
        self.assertEqual(backend.compiled_graph_for_general_shape,
                         self.compiled_graph_for_general_shape)
        self.assertEqual(backend.vllm_backend, self.vllm_backend)
        self.assertEqual(backend.concrete_size_entries[10].runtime_shape, 10)
        self.assertEqual(backend.concrete_size_entries[20].runtime_shape, 20)

    def test_check_for_ending_compilation(self):
        backend = NPUPiecewiseBackend(
            self.graph, self.vllm_config, self.graph_pool,
            self.piecewise_compile_index, self.total_piecewise_compiles,
            self.sym_shape_indices, self.compiled_graph_for_general_shape,
            self.vllm_backend)
        backend.is_last_graph = True
        backend.to_be_compiled_sizes = set()
        backend.check_for_ending_compilation()
        backend.vllm_backend.compiler_manager.save_to_file.assert_called_once()
