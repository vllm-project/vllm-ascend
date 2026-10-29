# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ordering checks for the unchanged production graph-entry methods.

Only the relevant methods are compiled from the production AST. Native graph
execution is mocked: these tests establish call ordering and error handling,
not device capture, numerical parity, or performance.
"""

import ast
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py"
HELPER_MODULE = "vllm_ascend.models.qwen4_exp.qsa_target_metadata"


def production_method(class_name, method_name, namespace):
    source = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    tree = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace[method_name]


class TestQsaTargetMetadataGraphOrdering(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.producer_error = False
        self.guard_error = False
        self.native_error = False
        helper = ModuleType(HELPER_MODULE)

        def materialize(state):
            self.assertIs(state, self.state)
            self.events.append("metadata")
            if self.producer_error:
                raise ValueError("producer failed")

        def validate(state, desc):
            self.assertIs(state, self.state)
            self.assertIs(desc, self.desc)
            self.events.append("validate")
            if self.guard_error:
                raise ValueError("replay guard failed")

        def success(state):
            self.assertIs(state, self.state)
            self.events.append("success")

        helper.execute_target_metadata_at_model_entry = materialize
        helper.validate_target_metadata_replay = validate
        helper.mark_target_metadata_replay_success = success
        self.modules = patch.dict(sys.modules, {HELPER_MODULE: helper})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.state = SimpleNamespace(_qsa_target_metadata_enabled=True, attn_metadata=object())
        self.desc = SimpleNamespace(num_tokens=8)
        self.result = object()
        self.device_metadata = SimpleNamespace(
            begin_forward=lambda: self.events.append("begin"),
            finish=lambda: self.events.append("finish"),
            finish_replay=lambda: self.events.append("finish_replay"),
        )

    def wrapper(self, target=True):
        extra_context = SimpleNamespace()
        namespace = {
            "get_forward_context": lambda: SimpleNamespace(cudagraph_runtime_mode="FULL"),
            "_EXTRA_CTX": extra_context,
            "torch": SimpleNamespace(npu=SimpleNamespace(is_current_stream_capturing=lambda: True)),
            "CUDAGraphMode": SimpleNamespace(PIECEWISE="PIECEWISE"),
        }
        forward = production_method("ModelWithContext", "forward", namespace)

        def model(*args, **kwargs):
            self.events.append("model")
            self.assertEqual((args, kwargs), ((17,), {"marker": 23}))
            return self.result

        wrapper = SimpleNamespace(
            original_model=model,
            is_draft_model=False,
            is_draft_model_prefill=False,
            device_metadata=self.device_metadata,
            qsa_metadata_state=self.state if target else None,
        )
        return wrapper, forward, extra_context

    def graph_manager(self, updatable=False):
        self.state.device_metadata = self.device_metadata
        namespace = {
            "BatchExecutionDescriptor": object,
            "torch": SimpleNamespace(Tensor=object),
            "set_current_vllm_config": lambda config: nullcontext(),
            "_get_graph_update_backend": lambda groups: "backend",
            "use_updatable_graph": lambda backend: updatable,
        }
        run = production_method("ModelAclGraphManager", "run_fullgraph", namespace)

        def native(*args):
            self.events.append("native")
            if self.native_error:
                raise ValueError("native replay failed")
            return self.result

        manager = SimpleNamespace(
            model_runner=SimpleNamespace(model_state=self.state, attn_groups=[]),
            update_stream=object(),
            vllm_config=object(),
            _graph_relay=native,
            _updatable_graph_replay=native,
        )
        return manager, run

    def test_target_entry_materializes_before_model_and_retires_after_consumers(self):
        wrapper, forward, context = self.wrapper()
        self.assertIs(forward(wrapper, 17, marker=23), self.result)
        self.assertEqual(self.events, ["begin", "metadata", "model", "finish"])
        self.assertTrue(context.capturing)

    def test_producer_failure_prevents_model_and_still_retires(self):
        wrapper, forward, _ = self.wrapper()
        self.producer_error = True
        with self.assertRaisesRegex(ValueError, "producer failed"):
            forward(wrapper, 17, marker=23)
        self.assertEqual(self.events, ["begin", "metadata", "finish"])

    def test_non_target_wrapper_keeps_original_model_path(self):
        wrapper, forward, _ = self.wrapper(target=False)
        self.assertIs(forward(wrapper, 17, marker=23), self.result)
        self.assertEqual(self.events, ["begin", "model", "finish"])

    def test_mtp_wrapper_retains_draft_context_without_target_metadata(self):
        wrapper, forward, context = self.wrapper(target=False)
        wrapper.is_draft_model = True
        wrapper.is_draft_model_prefill = True
        self.assertIs(forward(wrapper, 17, marker=23), self.result)
        self.assertEqual(self.events, ["begin", "model", "finish"])
        self.assertTrue(context.is_draft_model)
        self.assertTrue(context.is_draft_model_prefill)

    def test_replay_validates_before_either_native_path_and_marks_after_success(self):
        for updatable in (False, True):
            with self.subTest(updatable=updatable):
                self.events.clear()
                manager, run = self.graph_manager(updatable)
                self.assertIs(run(manager, self.desc), self.result)
                self.assertEqual(self.events, ["validate", "native", "success", "finish_replay"])

    def test_guard_failure_prevents_native_submission_and_success(self):
        manager, run = self.graph_manager()
        self.guard_error = True
        with self.assertRaisesRegex(ValueError, "replay guard failed"):
            run(manager, self.desc)
        self.assertEqual(self.events, ["validate"])

    def test_native_failure_is_not_counted_as_replay_success(self):
        manager, run = self.graph_manager()
        self.native_error = True
        with self.assertRaisesRegex(ValueError, "native replay failed"):
            run(manager, self.desc)
        self.assertEqual(self.events, ["validate", "native", "finish_replay"])


if __name__ == "__main__":
    unittest.main()
