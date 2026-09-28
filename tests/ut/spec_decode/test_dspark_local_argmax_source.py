# SPDX-License-Identifier: Apache-2.0
"""Exercise the changed MRV1 methods without importing the NPU model stack.

These source-isolated checks do not replace an end-to-end model test.
"""

import ast
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_PROPOSER = _ROOT / "vllm_ascend/spec_decode/llm_base_proposer.py"
_MODEL = _ROOT / "vllm_ascend/models/deepseek_v4/dspark.py"


def _method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


class _DeepSeekDraft:
    pass


def _selector(lmhead_tp=False):
    return _method(
        _PROPOSER,
        "AscendSpecDecodeBaseProposer",
        "_can_use_dspark_local_argmax",
        {"DSparkDeepseekV4ForCausalLM": _DeepSeekDraft, "lmhead_tp_enable": lambda: lmhead_tp},
    )


@pytest.mark.parametrize("unsupported", [None, "disabled", "probabilistic", "other_model", "mapping", "lmhead_tp"])
def test_local_argmax_selection(unsupported):
    model = _DeepSeekDraft() if unsupported != "other_model" else SimpleNamespace()
    if unsupported == "mapping":
        model.draft_id_to_target_id = torch.arange(3)
    proposer = SimpleNamespace(
        model=model,
        use_local_argmax_reduction=unsupported != "disabled",
        _enable_probabilistic_draft_probs=unsupported == "probabilistic",
    )
    assert _selector(unsupported == "lmhead_tp")(proposer) is (unsupported is None)


@pytest.mark.parametrize("soft_cap,scale", [(None, 1.0), (2.0, 1.0), (None, 0.5), (2.0, 0.5)])
def test_local_projection_keeps_normalization_and_logits_transforms(soft_cap, scale):
    hidden = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, 0.5]])
    norm = Mock(side_effect=lambda values: values / 2)
    apply_head = Mock(side_effect=lambda head, values, bias: values @ head.T)
    model = SimpleNamespace(
        lm_head=weight,
        model=SimpleNamespace(norm=norm),
        logits_processor=SimpleNamespace(_apply_head=apply_head, soft_cap=soft_cap, scale=scale),
    )
    compute = _method(_MODEL, "DSparkDeepseekV4ForCausalLM", "compute_local_draft_logits", {"torch": torch})
    actual = compute(model, hidden)
    expected = (hidden / 2) @ weight.T
    if soft_cap is not None:
        expected = torch.tanh(expected / soft_cap) * soft_cap
    expected *= scale
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    norm.assert_called_once()
    apply_head.assert_called_once()


def _sampling_branch():
    """Compile the actual DSpark branch, including the dynamic-length update."""
    tree = ast.parse(_PROPOSER.read_text())
    run = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run_merged_draft"
    )
    branch = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.method == 'dspark'"
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "dspark_has_vocab_mapping" for target in child.targets
            )
            for child in node.body
        )
    )
    helper_path = _ROOT / "vllm_ascend/spec_decode/dspark_local_argmax.py"
    spec = importlib.util.spec_from_file_location("dspark_argmax_source_test", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    namespace = {
        "torch": torch,
        "lmhead_tp_enable": lambda: False,
        "get_tp_group": lambda: SimpleNamespace(world_size=1, all_gather=Mock(side_effect=AssertionError)),
        "sample_local_draft_tokens": helper.sample_local_draft_tokens,
        "logger": Mock(),
    }
    wrapper = ast.parse(
        "def sample(self, sample_hidden_states, last_hidden_states, num_indices, sampling_metadata):\n"
        "    draft_probs_step0 = None\n"
        "    return draft_token_ids, draft_probs_step0\n"
    )
    wrapper.body[0].body[1:1] = branch.body
    exec(compile(ast.fix_missing_locations(wrapper), str(_PROPOSER), "exec"), namespace)
    return namespace["sample"]


@pytest.mark.parametrize("steps", [5, 7])
@pytest.mark.parametrize("mode", ["local", "disabled", "probabilistic", "mapping", "other_model"])
def test_sampling_preserves_buffers_fallbacks_and_dynamic_update(steps, mode):
    batch, vocab = 2, 11
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn(batch * steps, vocab, generator=generator)
    seeds = torch.tensor([3, 7])
    model = _DeepSeekDraft() if mode != "other_model" else SimpleNamespace()
    model.config = SimpleNamespace(vocab_size=vocab)
    model.lm_head = SimpleNamespace(shard_indices=SimpleNamespace(org_vocab_start_index=0))
    model.compute_local_draft_logits = Mock(side_effect=lambda _: logits.clone())
    model.compute_draft_logits = Mock(side_effect=lambda _: logits.clone())
    model.markov_embed = lambda tokens: tokens

    def bias(tokens):
        values = torch.zeros(batch, vocab)
        values.scatter_(1, ((tokens + 1) % vocab).unsqueeze(-1), 4)
        return values

    model.markov_bias = bias
    if mode == "mapping":
        model.draft_id_to_target_id = torch.arange(vocab)
        model.map_draft_to_target = lambda tokens: (tokens + 2) % vocab
    proposer = SimpleNamespace(
        model=model,
        num_speculative_tokens=steps,
        use_local_argmax_reduction=mode != "disabled",
        _enable_probabilistic_draft_probs=mode == "probabilistic",
        _dspark_seed_buffer=seeds,
        _dspark_draft_buffer=torch.full((batch + 1, steps + 1), -1, dtype=torch.int64),
        dynamic_spec=SimpleNamespace(update=Mock()),
        _sample_draft_from_logits=lambda values, metadata: (values.argmax(-1), values.softmax(-1)),
    )
    proposer._can_use_dspark_local_argmax = MethodType(_selector(), proposer)
    expected = torch.empty(batch, steps + 1, dtype=torch.int64)
    expected[:, 0] = seeds
    for step in range(steps):
        tokens = (logits.view(batch, steps, vocab)[:, step] + bias(expected[:, step])).argmax(-1)
        expected[:, step + 1] = model.map_draft_to_target(tokens) if mode == "mapping" else tokens
    hidden = torch.zeros(batch * steps, 4)
    actual, probs = _sampling_branch()(proposer, hidden, hidden, batch * steps, None)
    torch.testing.assert_close(actual, expected)
    assert actual.data_ptr() == proposer._dspark_draft_buffer.data_ptr()
    assert torch.all(proposer._dspark_draft_buffer[batch] == -1)
    assert model.compute_local_draft_logits.call_count == (mode == "local")
    assert model.compute_draft_logits.call_count == (mode != "local")
    proposer.dynamic_spec.update.assert_called_once()
    assert proposer.dynamic_spec.update.call_args.kwargs["draft_token_ids"] is actual
    assert proposer.dynamic_spec.update.call_args.kwargs["num_reqs"] == batch
    if mode == "probabilistic":
        assert probs.shape == (batch * steps, vocab)
    else:
        assert probs is None
