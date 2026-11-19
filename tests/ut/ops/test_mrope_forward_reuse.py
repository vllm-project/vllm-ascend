import ast
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/patch_qwen3_5.py"
PP = SimpleNamespace(is_first_rank=True, is_last_rank=True)
ns = {"torch": torch, "islice": islice, "IntermediateTensors": dict, "get_pp_group": lambda: PP}
tree = ast.parse(SOURCE.read_text())
for name in ("_uses_multimodal_rope", "_get_shared_mrope_cos_sin", "qwen3_5_model_forward"):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)


class Rope(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.register_buffer("cos_sin_cache", torch.arange(64 * 8).reshape(64, 8).float() + offset)
        self.mrope_section = [1, 1, 2]


class Layer(torch.nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.layer_type = "full_attention"
        self.self_attn = torch.nn.Module()
        self.self_attn.rotary_emb = rope
        self.self_attn.config = SimpleNamespace(model_type="qwen3_5")

    def forward(self, positions, hidden_states, residual, mrope_cos_sin=None):
        assert mrope_cos_sin is not None
        return hidden_states + mrope_cos_sin.sum(dim=(0, 2)).unsqueeze(-1), residual


class Norm(torch.nn.Module):
    def forward(self, x, residual):
        return x, None


class Model(torch.nn.Module):
    forward = ns["qwen3_5_model_forward"]

    def __init__(self, separate=False):
        super().__init__()
        rope = Rope()
        self.layers = torch.nn.ModuleList([Layer(Rope(i) if separate else rope) for i in range(10)])
        self.start_layer = 0
        self.end_layer = 10
        self.norm = Norm()
        self.use_sequence_parallel = False

    def _maybe_add_hidden_state(self, states, index, hidden, residual):
        return states


def expected(model, pos, x):
    if pos.ndim == 1:
        pos = pos.unsqueeze(0).expand(3, -1)
    for layer in model.layers:
        x = x + layer.self_attn.rotary_emb.cos_sin_cache[pos].sum(dim=(0, 2)).unsqueeze(-1)
    return x


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("text", [False, True])
def test_forward_reuse_and_refresh(separate, text):
    model = Model(separate)
    pos = torch.tensor([1, 2, 3]) if text else torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = torch.zeros(3, 2)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        a = model(None, pos, inputs_embeds=x)
    assert torch.equal(a, expected(model, pos, x))
    indices = sum(e.count for e in prof.key_averages() if e.key == "aten::index")
    assert indices == (10 if separate else 1)
    pos.add_(1)
    assert torch.equal(model(None, pos, inputs_embeds=x), expected(model, pos, x))
    assert not torch.equal(a, model(None, pos, inputs_embeds=x))


@pytest.mark.parametrize("text", [False, True])
def test_fullgraph_compile_refresh(text):
    model = Model()
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(model, backend=backend, fullgraph=True)
    pos = torch.tensor([1, 2, 3]) if text else torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = torch.zeros(3, 2)
    for _ in range(3):
        assert torch.equal(compiled(None, pos, inputs_embeds=x), expected(model, pos, x))
        pos.add_(1)
    assert len(graphs) == 1
    assert (
        sum(
            n.op == "call_function" and n.target.__name__ == "getitem" and isinstance(n.args[1], torch.fx.Node)
            for n in graphs[0].graph.nodes
        )
        == 1
    )
