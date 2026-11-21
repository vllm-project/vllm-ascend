from unittest.mock import patch

import pytest
import torch
from torch._functorch.aot_autograd import aot_module_simplified

from vllm_ascend.compilation.compiler_interface import AscendCompiler


@pytest.mark.parametrize("artifact_exists", [False, True])
def test_cache_reload_preserves_aot_input_mutation_wrapper(tmp_path, artifact_exists):
    """A saved inner FX kernel cannot replace the callable returned by AOT."""

    class Model(torch.nn.Module):
        def forward(self, left, right):
            left.add_(right)
            return (left * 3,)

    graph = torch.fx.symbolic_trace(Model())
    sample = torch.arange(4, dtype=torch.float32)
    inner = []

    def fw_compiler(gm, inputs):
        inner.append(gm)
        return gm

    rhs = torch.ones_like(sample)
    wrapped = aot_module_simplified(graph, [sample, rhs], fw_compiler=fw_compiler, keep_inference_input_mutations=False)
    reference_input = sample.clone()
    expected = wrapped(reference_input, rhs)
    assert len(inner) == 1
    # AOT returns the mutated input separately and its wrapper copies it back.
    assert len(next(n for n in inner[0].graph.nodes if n.op == "output").args[0]) == 2
    artifact = tmp_path / "artifact"
    if artifact_exists:
        artifact.write_text("serialized inner graph")
    compiler = AscendCompiler()
    compiler.vllm_config = object()
    cfg = type("Config", (), {"enable_npugraph_ex": True})()
    config = type("AscendConfig", (), {"ascend_compilation_config": cfg})()
    with (
        patch("vllm_ascend.compilation.compiler_interface.get_ascend_config", return_value=config),
        patch("vllm_ascend.compilation.compiler_interface.npugraph_ex_compile", return_value=(wrapped, None)),
        patch("npugraph_ex.npu_fx_compiler._CompiledFxGraph.load_artifacts", return_value=inner[0]),
    ):
        restored = compiler.load(("artifact", str(artifact)), graph, [sample, rhs], 0, (1, 4))
        actual = restored(sample, rhs)
    assert len(actual) == 1
    torch.testing.assert_close(actual[0], expected[0])

    torch.testing.assert_close(sample, reference_input)
