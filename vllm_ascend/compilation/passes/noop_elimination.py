from collections.abc import Iterable

import torch
import torch.fx
from torch import SymInt
from torch.fx.experimental.symbolic_shapes import statically_known_true
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.logger import logger


class NoOpEliminationPass(VllmInductorPass):
    """Remove no-op view/reshape nodes after pattern rewrites."""

    def __call__(self, graph: torch.fx.Graph) -> None:
        fx_graph = graph.graph if hasattr(graph, "graph") else graph
        removed = 0
        for node in list(fx_graph.nodes):
            if not self._is_view_like(node):
                continue

            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue

            input_meta = input_node.meta.get("val")
            output_meta = node.meta.get("val")
            if input_meta is None or output_meta is None:
                continue

            input_shape = getattr(input_meta, "shape", None)
            output_shape = getattr(output_meta, "shape", None)
            if input_shape is None or output_shape is None:
                continue

            if self._all_dims_equivalent(input_shape, output_shape):
                node.replace_all_uses_with(input_node)
                fx_graph.erase_node(node)
                removed += 1

        logger.debug("NoOpEliminationPass removed %s no-op views", removed)

    @staticmethod
    def _is_view_like(node: torch.fx.Node) -> bool:
        return (node.op == "call_method" and node.target in {"view", "reshape"}) or (
            node.op == "call_function"
            and node.target
            in {
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
            }
        )

    @staticmethod
    def _dims_equivalent(dim: int | SymInt, i_dim: int | SymInt) -> bool:
        return statically_known_true(dim == i_dim)  # type: ignore[no-any-return]

    def _all_dims_equivalent(self, dims: Iterable[int | SymInt], i_dims: Iterable[int | SymInt]) -> bool:
        dims_ = list(dims)
        i_dims_ = list(i_dims)
        if len(dims_) != len(i_dims_):
            return False
        return all(self._dims_equivalent(s, i_s) for s, i_s in zip(dims_, i_dims_))
