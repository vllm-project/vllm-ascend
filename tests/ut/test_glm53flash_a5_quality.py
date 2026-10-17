# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real A5 numerical collector without importing an NPU runtime."""

import ast
import contextlib
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_DIRECTORY = Path(__file__).resolve().parents[1] / "e2e/pull_request/four_card"
_SOURCE = _DIRECTORY / "test_glm5_3_flash_a5_quality.py"
_SMOKE_SOURCE = _DIRECTORY / "test_glm5_3_flash_a5.py"
_SPEC = importlib.util.spec_from_file_location("glm53flash_quality", _DIRECTORY / "glm53flash_quality.py")
assert _SPEC is not None and _SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)


def _collector_namespace():
    # Execute selected real definitions so tests cover production control flow
    # while the module's pytest/torch/vllm imports remain outside CPU-only UTs.
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    functions = {"_prompt", "_probe_token_ids", "_check_completions", "_rank_counts", "collect_numerical"}
    constants = {"ACCURACY_TOKENS", "ACCURACY_CONTEXT_TOKEN", "ACCURACY_LOGPROBS", "NUMERICAL_SCENARIOS"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
        )
    ]
    namespace = {
        "DP_SIZE": 4,
        "contextlib": contextlib,
        "summarize_outputs": _HELPER.summarize_outputs,
        "_accuracy_parameters": lambda _: None,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SOURCE), "exec"), namespace)
    return namespace


def _parameter_seed_function():
    tree = ast.parse(_SMOKE_SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_parameter_seed"]
    assert len(nodes) == 1
    namespace = {"hashlib": hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SMOKE_SOURCE), "exec"), namespace)
    return namespace["_parameter_seed"]


def _protocol_namespace(assets):
    symbols = {
        name: object()
        for name in (
            "load_weights",
            "initialize_parameter",
            "_parameter_seed",
            "_write_model",
            "_write_a5_model",
            "_prompt",
            "_parameters",
            "_accuracy_parameters",
        )
    }
    sources = {
        symbol: f"def {name}():\n    marker = 'original'\n    return marker\n" for name, symbol in symbols.items()
    }
    namespace = {
        **symbols,
        "__name__": "tests.e2e.pull_request.four_card.test_glm5_3_flash_a5_quality",
        "BASELINE_FILE": assets / "quality_a5_baseline.json",
        "hashlib": hashlib,
        "json": json,
        "inspect": SimpleNamespace(getsource=sources.__getitem__),
        "FlashA5DummyLoader": SimpleNamespace(
            load_weights=symbols["load_weights"], initialize_parameter=symbols["initialize_parameter"]
        ),
    }
    for path in (_SMOKE_SOURCE, _SOURCE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in {"quality_protocol", "_probe_token_ids", "_engine_settings"}
            )
            or (
                isinstance(node, ast.Assign)
                and all(
                    isinstance(target, ast.Name) and target.id.isupper() and target.id != "BASELINE_FILE"
                    for target in node.targets
                )
            )
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, symbols, sources


class FakeDP:
    def __init__(self, namespace, *, eager=False, fault=None):
        self.namespace = namespace
        self.eager = eager
        self.fault = fault
        self.calls = []
        self.stopped = False

    def collective_rpc(self, method):
        if method in ("check_raw_logprobs", "start_replay_probe"):
            return [[True]] * 4
        if method == "stop_replay_probe":
            self.stopped = True
        counts = [0 if self.eager else len(self.calls) * 15] * 4
        if self.fault == "idle_last_rank":
            counts[-1] = 0
        return [[count] for count in counts]

    def generate_raw(self, prompts, parameters, **kwargs):
        self.calls.append(prompts)
        if self.fault == "engine_error":
            raise RuntimeError("engine error")
        keys = self.namespace["_probe_token_ids"](1024)
        results = []
        for prompt in prompts:
            results.append(
                SimpleNamespace(
                    finished=True,
                    prompt_token_ids=list(prompt),
                    outputs=[
                        SimpleNamespace(
                            token_ids=[42] * 16,
                            logprobs=[
                                {key: SimpleNamespace(logprob=-5.0 - index * 0.001) for index, key in enumerate(keys)}
                                for _ in range(16)
                            ],
                        )
                    ],
                )
            )
        if self.fault == "missing_rank":
            results.pop()
        elif self.fault == "wrong_prompt":
            results[-1].prompt_token_ids[0] += 1
        elif self.fault == "unfinished":
            results[-1].finished = False
        elif self.fault == "masked_logprobs":
            results[-1].outputs[0].logprobs[0][42].logprob = 0.0
        return results


class TestA5NumericalCollector(unittest.TestCase):
    def test_each_scenario_retains_all_four_ranks_with_identical_shapes(self):
        namespace = _collector_namespace()
        runner = FakeDP(namespace)
        observed = namespace["collect_numerical"](runner, 1024, False)
        self.assertEqual(len(observed["numerical"]), 16)
        self.assertEqual(observed["replay_counts"], [60] * 4)
        self.assertEqual(observed["scenario_replays"], [[15] * 4, [30] * 4, [45] * 4, [60] * 4])
        self.assertEqual([len(prompts[0]) for prompts in runner.calls], [127, 128, 129, 513])
        for prompts in runner.calls:
            self.assertEqual(len(prompts), 4)
            self.assertTrue(all(prompt == prompts[0] for prompt in prompts))
        self.assertTrue(runner.stopped)

    def test_eager_expects_no_replay(self):
        namespace = _collector_namespace()
        runner = FakeDP(namespace, eager=True)
        observed = namespace["collect_numerical"](runner, 1024, True)
        self.assertEqual(observed["replay_counts"], [0] * 4)
        with self.assertRaises(AssertionError):
            namespace["collect_numerical"](FakeDP(namespace), 1024, True)

    def test_graph_requires_each_rank_to_replay(self):
        namespace = _collector_namespace()
        runner = FakeDP(namespace, fault="idle_last_rank")
        with self.assertRaisesRegex(AssertionError, "every DP rank"):
            namespace["collect_numerical"](runner, 1024, False)
        self.assertTrue(runner.stopped)

    def test_invalid_last_rank_cannot_hide_behind_other_ranks(self):
        namespace = _collector_namespace()
        for fault in ("missing_rank", "wrong_prompt", "unfinished", "masked_logprobs"):
            with self.subTest(fault=fault):
                runner = FakeDP(namespace, fault=fault)
                with self.assertRaises((AssertionError, ValueError)):
                    namespace["collect_numerical"](runner, 1024, False)
                self.assertTrue(runner.stopped)

    def test_engine_failure_restores_replay_instrumentation(self):
        namespace = _collector_namespace()
        runner = FakeDP(namespace, fault="engine_error")
        with self.assertRaisesRegex(RuntimeError, "engine error"):
            namespace["collect_numerical"](runner, 1024, False)
        self.assertTrue(runner.stopped)

    def test_dead_engine_error_survives_empty_or_failed_cleanup_rpc(self):
        namespace = _collector_namespace()

        class DeadDP(FakeDP):
            def __init__(self, cleanup_error):
                super().__init__(namespace)
                self.cleanup_error = cleanup_error
                self.connections = [object()] * 4

            def generate_raw(self, *args, **kwargs):
                # Match DPVllmRunner's failure path: it stops the child workers
                # and clears their connections before propagating the error.
                self.connections.clear()
                raise RuntimeError("original EngineDeadError")

            def collective_rpc(self, method):
                if method == "stop_replay_probe":
                    self.stopped = True
                    if self.cleanup_error:
                        raise BrokenPipeError("cleanup pipe is gone")
                    return [True for _ in self.connections]
                return super().collective_rpc(method)

        for cleanup_error in (False, True):
            with self.subTest(cleanup_error=cleanup_error):
                runner = DeadDP(cleanup_error)
                with self.assertRaisesRegex(RuntimeError, "original EngineDeadError"):
                    namespace["collect_numerical"](runner, 1024, False)
                self.assertEqual(runner.connections, [])
                self.assertTrue(runner.stopped)

    def test_rank_count_shape_and_values(self):
        validate = _collector_namespace()["_rank_counts"]
        for counts in ([[1]] * 3, [[1, 2]] * 4, [[True]] * 4, [[-1]] * 4, [1] * 4):
            with self.subTest(counts=counts), self.assertRaises(AssertionError):
                validate(counts)


class TestA5ParameterSeeds(unittest.TestCase):
    def test_same_name_and_rank_are_deterministic(self):
        parameter_seed = _parameter_seed_function()
        for name in ("model.layers.3.mlp.experts.w13_weight", "model.layers.3.self_attn.q_b_proj.weight"):
            for rank in range(4):
                with self.subTest(name=name, rank=rank):
                    self.assertEqual(parameter_seed(name, rank), parameter_seed(name, rank))
                    self.assertIs(type(parameter_seed(name, rank)), int)

    def test_expert_shards_have_distinct_rank_seeds(self):
        parameter_seed = _parameter_seed_function()
        for name in ("model.layers.3.mlp.experts.w13_weight", "model.layers.3.mlp.experts.w2_weight"):
            with self.subTest(name=name):
                self.assertEqual(len({parameter_seed(name, rank) for rank in range(4)}), 4)

    def test_replicated_parameters_keep_the_same_seed_across_ranks(self):
        parameter_seed = _parameter_seed_function()
        for name in (
            "model.layers.3.self_attn.q_b_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.3.mlp.shared_experts.gate_up_proj.weight",
        ):
            with self.subTest(name=name):
                self.assertEqual(len({parameter_seed(name, rank) for rank in range(4)}), 1)


class TestA5ProtocolSources(unittest.TestCase):
    def test_model_hash_tracks_complete_a5_loader_initializer_and_seed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            for name in (
                "config.json",
                "multimodal.json",
                "quant.json",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ):
                (assets / name).write_text(json.dumps({"vocab_size": 1024}), encoding="utf-8")
            namespace, symbols, sources = _protocol_namespace(assets)
            protocol = namespace["quality_protocol"]
            original = protocol()
            self.assertEqual(protocol(), original)
            for name in ("load_weights", "initialize_parameter", "_parameter_seed"):
                with self.subTest(source=name):
                    symbol = symbols[name]
                    source = sources[symbol]
                    # Only the final body line changes: hashing just a name or
                    # signature must not satisfy this invalidation contract.
                    sources[symbol] = source.replace("return marker", "return marker + 'changed'")
                    changed = protocol()
                    self.assertNotEqual(changed["model_sha256"], original["model_sha256"])
                    self.assertEqual(changed["sampling_sha256"], original["sampling_sha256"])
                    sources[symbol] = source
                    self.assertEqual(protocol(), original)


if __name__ == "__main__":
    unittest.main()
