import operator

import torch
from torch import fx


def _make_size_dim_graph(dim: int = 0) -> fx.GraphModule:
    graph = fx.Graph()
    tensor = graph.placeholder("tensor")
    tensor.meta["example_value"] = torch.empty(3, 2)
    size = graph.call_method("size", args=(tensor, dim))
    output = graph.call_function(torch.zeros, args=((size, 2),))
    graph.call_function(
        operator.setitem,
        args=(output, slice(None, size, None), tensor),
    )
    graph.output(output)
    return fx.GraphModule({}, graph)


def test_decompose_dim_size_node_replaces_nested_users() -> None:
    from vllm_ascend._310p.piecewise_size_nodes import (
        decompose_dim_size_nodes,
    )

    graph = _make_size_dim_graph()

    decompose_dim_size_nodes(graph)

    assert list(graph.graph.find_nodes(op="call_method", target="size")) == []
    zeros = next(node for node in graph.graph.nodes if node.op == "call_function" and node.target is torch.zeros)
    setitem = next(node for node in graph.graph.nodes if node.op == "call_function" and node.target is operator.setitem)
    assert zeros.args == ((3, 2),)
    assert setitem.args[1] == slice(None, 3, None)


def test_decompose_dim_size_node_normalizes_negative_dim() -> None:
    from vllm_ascend._310p.piecewise_size_nodes import (
        decompose_dim_size_nodes,
    )

    graph = _make_size_dim_graph(-2)

    decompose_dim_size_nodes(graph)

    assert list(graph.graph.find_nodes(op="call_method", target="size")) == []


def test_install_piecewise_size_node_compat_is_idempotent(monkeypatch) -> None:
    from vllm.compilation import backends

    from vllm_ascend._310p.piecewise_size_nodes import (
        install_piecewise_size_node_compat,
    )

    calls = []

    def original(graph: fx.GraphModule) -> None:
        calls.append(graph)

    monkeypatch.setattr(backends, "_decompose_size_nodes", original)

    install_piecewise_size_node_compat()
    installed = backends._decompose_size_nodes
    install_piecewise_size_node_compat()

    assert backends._decompose_size_nodes is installed
    graph = _make_size_dim_graph()
    installed(graph)
    assert calls == [graph]
    assert list(graph.graph.find_nodes(op="call_method", target="size")) == []


def test_install_full_decode_size_node_compat_uses_same_scalar_guard(
    monkeypatch,
) -> None:
    from vllm.compilation import backends

    from vllm_ascend._310p.piecewise_size_nodes import (
        install_full_decode_size_node_compat,
    )

    calls = []

    def original(graph: fx.GraphModule) -> None:
        calls.append(graph)

    monkeypatch.setattr(backends, "_decompose_size_nodes", original)

    install_full_decode_size_node_compat()
    installed = backends._decompose_size_nodes
    graph = _make_size_dim_graph()
    installed(graph)

    assert calls == [graph]
    assert list(graph.graph.find_nodes(op="call_method", target="size")) == []


def test_hybrid_size_node_capability_does_not_stack_process_global_patch(
    monkeypatch,
) -> None:
    from vllm.compilation import backends

    from vllm_ascend._310p.piecewise_size_nodes import (
        install_full_and_piecewise_size_node_compat,
        install_full_decode_size_node_compat,
        install_piecewise_size_node_compat,
    )

    calls = []

    def original(graph: fx.GraphModule) -> None:
        calls.append(graph)

    monkeypatch.setattr(backends, "_decompose_size_nodes", original)

    install_full_and_piecewise_size_node_compat()
    installed = backends._decompose_size_nodes
    install_full_and_piecewise_size_node_compat()
    install_piecewise_size_node_compat()
    install_full_decode_size_node_compat()

    assert backends._decompose_size_nodes is installed
    graph = _make_size_dim_graph()
    installed(graph)
    assert calls == [graph]
    assert list(graph.graph.find_nodes(op="call_method", target="size")) == []
