# SPDX-License-Identifier: Apache-2.0
from itertools import permutations
from types import SimpleNamespace
from unittest import TestCase

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.models.minimax_m3.msa_m3 import AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend


class TestGraphUpdateBackend(TestCase):
    def test_minimax_mixed_groups_select_dense_attention(self):
        backends = AscendAttentionBackend, AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend
        for order in permutations(backends):
            groups = [SimpleNamespace(backend=backend) for backend in order]
            for layout in ([groups], [[group] for group in groups]):
                with self.subTest(order=order, num_cache_groups=len(layout)):
                    self.assertIs(_get_graph_update_backend(layout), AscendAttentionBackend)

    def test_sparse_only_keeps_first_executable_backend(self):
        for order in permutations((AscendMiniMaxM3IndexerBackend, AscendMiniMaxM3SparseBackend)):
            with self.subTest(order=order):
                groups = [[SimpleNamespace(backend=backend)] for backend in order]
                self.assertIs(_get_graph_update_backend(groups), order[0])

    def test_no_executable_backend(self):
        with self.assertRaisesRegex(RuntimeError, "No executable attention backend"):
            _get_graph_update_backend([[]])
