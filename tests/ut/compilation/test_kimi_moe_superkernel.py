# SPDX-License-Identifier: Apache-2.0
"""CPU tests of PR64 scopes, graph ownership, config reuse and draft isolation."""

import ast
import copy
import sys
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(path, names, ns, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    selected = [n for n in nodes if getattr(n, "name", None) in names]
    selected.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), path, "exec"), ns)
    return ns


class TestKimiMoESuperkernel(unittest.TestCase):
    def setUp(self):
        self.context = NS(additional_kwargs={})
        self.npu = NS(super_kernel_scope_begin=Mock(), super_kernel_scope_end=Mock())
        self.ns = load(
            "vllm_ascend/compilation/kimi_moe_superkernel.py",
            {
                "collect_moe_superkernel_scopes",
                "moe_superkernel_scope",
                "moe_superkernel_options",
                "optimize_moe_superkernel",
                "use_kimi_moe_superkernel",
            },
            dict(
                contextmanager=contextmanager,
                torch=NS(npu=self.npu),
                logger=Mock(),
                get_forward_context=lambda: self.context,
                get_capture_resource=lambda key, factory: factory(),
                _CAPTURE_SCOPES_KEY="scopes",
                _is_kimi_moe_model=lambda vc: vc.kimi,
                get_current_hardware_profile=lambda: NS(supports=lambda cap: True),
                HardwareCapability=NS(CANN_MEGAMOE_MXFP=1),
            ),
        )

    def test_existing_switch_and_runtime_family_select_scopes(self):
        cfg = NS(ascend_compilation_config=NS(enable_super_kernel=True), multistream_overlap_shared_expert=True)
        vc = NS(kimi=True, use_v2_model_runner=False)
        select = self.ns["use_kimi_moe_superkernel"]
        for runner_v2 in (False, True):
            vc.use_v2_model_runner = runner_v2
            self.assertTrue(select(vc, cfg))
        for obj, key, value in (
            (cfg.ascend_compilation_config, "enable_super_kernel", False),
            (cfg, "multistream_overlap_shared_expert", False),
            (vc, "kimi", False),
        ):
            old = getattr(obj, key)
            setattr(obj, key, value)
            self.assertFalse(select(vc, cfg))
            setattr(obj, key, old)

    def test_v2_capture_owns_scopes_after_forward_context_exits(self):
        # Upstream V2 enters torch.cuda.graph before creating the forward
        # context, then destroys the context before leaving the graph.
        active_graph = ContextVar("test_kimi_graph", default=None)
        resources = load(
            "vllm_ascend/compilation/updatable_graph.py",
            {"get_capture_resource"},
            {"_ACTIVE_GRAPH": active_graph},
        )["get_capture_resource"]
        graph_resource = load(
            "vllm_ascend/compilation/updatable_graph.py", {"get_capture_resource"}, {}, "UpdatableGraph"
        )["get_capture_resource"]
        self.ns["get_capture_resource"] = resources
        cfg = NS(ascend_compilation_config=NS(enable_super_kernel=True), multistream_overlap_shared_expert=True)
        vc = NS(kimi=True, use_v2_model_runner=True)
        order = []

        @contextmanager
        def capture(graph, *args, **kwargs):
            order.append("capture_begin")
            token = active_graph.set(graph)
            try:
                yield
            finally:
                active_graph.reset(token)
                order.append("capture_end")

        @contextmanager
        def whole_model(name, enabled):
            self.assertFalse(enabled, "Kimi must not inherit whole-model fusion")
            yield

        wrapper_ns = load(
            "vllm_ascend/worker/v2/utils.py",
            {"torch_npu_graph_wrapper"},
            dict(
                contextmanager=contextmanager,
                torch=NS(npu=NS(graph=capture)),
                get_ascend_config=lambda: cfg,
                get_current_vllm_config_or_none=lambda: vc,
                use_kimi_moe_superkernel=self.ns["use_kimi_moe_superkernel"],
                optimize_moe_superkernel=self.ns["optimize_moe_superkernel"],
                super_kernel_scope=whole_model,
                weak_ref_workspaces=Mock(),
                get_graph_params=lambda: "target",
                get_draft_graph_params=lambda: "draft",
            ),
        )

        class Graph:
            get_capture_resource = graph_resource

            def __init__(self):
                self.capture_resources = {}
                self.super_kernel_optimize = Mock(side_effect=lambda **kw: order.append("optimize"))

        scope = self.ns["moe_superkernel_scope"]
        # Warmup has no owning graph and must not leak names into capture.
        with scope("warmup", True):
            pass
        for name, draft in (("layer.0", False), ("layer.1", False), (None, False), ("draft", True)):
            graph = Graph()
            order.clear()
            with wrapper_ns["torch_npu_graph_wrapper"](graph):
                self.context = NS(additional_kwargs={})
                with scope(name, name is not None and not draft):
                    order.append("forward")
                self.context = None  # Context no longer exists at optimize.
            self.assertEqual(order[:3], ["capture_begin", "forward", "capture_end"])
            if name is not None and not draft:
                self.assertEqual(graph.capture_resources["scopes"], {name})
                graph.super_kernel_optimize.assert_called_once_with(
                    optimize_options=self.ns["moe_superkernel_options"]()
                )
                self.assertEqual(order[-1], "optimize")
            else:
                graph.super_kernel_optimize.assert_not_called()
        self.assertEqual(wrapper_ns["weak_ref_workspaces"].call_count, 8)

        graph = Graph()
        with self.assertRaisesRegex(RuntimeError, "capture failed"), wrapper_ns["torch_npu_graph_wrapper"](graph):
            self.context = NS(additional_kwargs={})
            with scope("failed", True):
                raise RuntimeError("capture failed")
        graph.super_kernel_optimize.assert_not_called()
        self.assertIsNone(active_graph.get())
        self.assertEqual(self.context.additional_kwargs, {})
        self.assertEqual(wrapper_ns["weak_ref_workspaces"].call_count, 10)

    def test_v2_other_models_keep_whole_model_policy_and_cleanup(self):
        for enabled in (False, True):
            graph = Mock()
            whole_model = Mock(return_value=nullcontext())
            ns = load(
                "vllm_ascend/worker/v2/utils.py",
                {"torch_npu_graph_wrapper"},
                dict(
                    contextmanager=contextmanager,
                    torch=NS(npu=NS(graph=lambda *a, **kw: nullcontext())),
                    get_ascend_config=lambda enabled=enabled: NS(
                        ascend_compilation_config=NS(enable_super_kernel=enabled)
                    ),
                    get_current_vllm_config_or_none=lambda: NS(kimi=False),
                    use_kimi_moe_superkernel=lambda *a: False,
                    optimize_moe_superkernel=Mock(),
                    super_kernel_scope=whole_model,
                    weak_ref_workspaces=Mock(),
                    get_graph_params=lambda: "target",
                    get_draft_graph_params=lambda: "draft",
                    logger=Mock(),
                ),
            )
            with ns["torch_npu_graph_wrapper"](cuda_graph=graph):
                pass
            whole_model.assert_called_once_with("full_model", enabled)
            ns["optimize_moe_superkernel"].assert_not_called()
            self.assertEqual(ns["weak_ref_workspaces"].call_count, 2)
            if enabled:
                graph.super_kernel_optimize.assert_called_once_with(optimize_options={"dcci_after_kernel_end": [".*"]})
            else:
                graph.super_kernel_optimize.assert_not_called()

    def test_nested_scope_and_failure_cleanup(self):
        collect, scope = self.ns["collect_moe_superkernel_scopes"], self.ns["moe_superkernel_scope"]
        with collect(self.context, True) as outer, scope("outer", True):
            with (
                collect(self.context, True) as inner,
                self.assertRaisesRegex(RuntimeError, "kernel"),
                scope("inner", True),
                scope(None, True),
            ):
                raise RuntimeError("kernel")
            with scope("restored", True):
                pass
        self.assertEqual(outer, {"outer", "restored"})
        self.assertEqual(inner, {"inner"})
        self.assertEqual(self.context.additional_kwargs, {})
        self.assertEqual(self.npu.super_kernel_scope_begin.call_count, self.npu.super_kernel_scope_end.call_count)

    def test_runtime_scope_covers_projections_but_excludes_final_add(self):
        active = []
        observed = {}

        @contextmanager
        def scope(name, enabled):
            if enabled:
                active.append(name)
            try:
                yield
            finally:
                if enabled:
                    active.pop()

        def record(name, result):
            observed[name] = active[-1] if active else None
            return result

        x = torch.ones(2, 4)
        runner = NS(
            layer_name="layer",
            _kimi_gate=object(),
            _can_run_kimi_decode=lambda _: True,
            _maybe_pad_hidden_states=lambda shared, hidden: (hidden, None, None),
            _fused_output_is_reduced=True,
            _maybe_reduce_routed_output_before_transform=lambda output, reduced: (output, reduced),
            _maybe_reduce_shared_expert_output=lambda shared, reduced: shared,
            _maybe_apply_routed_scale_to_output=lambda shared, routed: (shared, routed),
            _maybe_reduce_final_output=lambda output, *args: output,
            _maybe_add_zero_expert_output=lambda output: output,
        )
        runner.apply_routed_input_transform = lambda hidden: record("front", (hidden, hidden))
        runner._forward_impl = lambda *args: record("experts", (x, x * 2))
        runner.apply_routed_output_transform = lambda output: record("up", output)
        ns = dict(
            torch=torch,
            _unpack=lambda output: output,
            get_layer_from_name=lambda _: runner,
            _resolve_layer_name=lambda name: name,
            moe_superkernel_scope=scope,
            get_ascend_config=lambda: NS(ascend_compilation_config=NS(enable_super_kernel=True)),
            _EXTRA_CTX=NS(is_draft_model=False),
        )
        load("vllm_ascend/ops/fused_moe/fused_moe.py", {"_ascend_kimi_moe"}, ns)
        load("vllm_ascend/ops/fused_moe/fused_moe.py", {"_forward_standard"}, ns, "AscendMoERunner")
        runner._forward_standard = lambda *args: ns["_forward_standard"](runner, *args)
        tensor_add = torch.Tensor.__add__
        with patch.object(torch.Tensor, "__add__", lambda a, b: record("add", tensor_add(a, b))):
            result = ns["_ascend_kimi_moe"](x, None, "layer")
        torch.testing.assert_close(result, x * 3)
        self.assertEqual(observed, {"front": "layer_1.moe", "experts": "layer_1.moe", "up": "layer_1.moe", "add": None})
        self.assertFalse(runner._kimi_superkernel_active)
        self.assertEqual(active, [])
        runner._forward_standard = Mock(side_effect=RuntimeError("kernel"))
        with self.assertRaisesRegex(RuntimeError, "kernel"):
            ns["_ascend_kimi_moe"](x, None, "layer")
        self.assertFalse(runner._kimi_superkernel_active)
        self.assertEqual(active, [])

    def test_runtime_scope_uses_resolved_runner_name_with_one_lookup_per_layer(self):
        x = torch.ones(2, 4)
        layer_names = [f"model.layers.{index}.mlp.experts" for index in range(2)]
        for encoding in ("context", "opaque", "string"):
            with self.subTest(encoding=encoding):
                self.npu.super_kernel_scope_begin.reset_mock()
                runners = [
                    NS(layer_name=name, _can_run_kimi_decode=lambda _: True, _forward_standard=Mock(return_value=x))
                    for name in layer_names
                ]
                lookup = Mock(side_effect=runners)
                ns = load(
                    "vllm_ascend/ops/fused_moe/fused_moe.py",
                    {"_ascend_kimi_moe"},
                    dict(
                        get_layer_from_name=lookup,
                        _resolve_layer_name=lambda value: value.value if isinstance(value, NS) else value,
                        moe_superkernel_scope=self.ns["moe_superkernel_scope"],
                        get_ascend_config=lambda: NS(ascend_compilation_config=NS(enable_super_kernel=True)),
                        _EXTRA_CTX=NS(is_draft_model=False),
                    ),
                )
                with self.ns["collect_moe_superkernel_scopes"](self.context, True) as scopes:
                    for name in layer_names:
                        encoded = "from_forward_context" if encoding == "context" else NS(value=name)
                        if encoding == "string":
                            encoded = name
                        self.assertIs(ns["_ascend_kimi_moe"](x, None, encoded), x)
                expected_names = [f"{name}_1.moe" for name in layer_names]
                self.assertEqual(scopes, set(expected_names))
                self.assertEqual(
                    [call.args[0] for call in self.npu.super_kernel_scope_begin.call_args_list], expected_names
                )
                self.assertEqual(lookup.call_count, len(layer_names))
                self.assertEqual(
                    [call.args[0] for call in lookup.call_args_list],
                    ["from_forward_context"] * len(layer_names) if encoding == "context" else layer_names,
                )
                for runner in runners:
                    runner._forward_standard.assert_called_once_with(x, x, None, None)
                self.assertEqual(self.context.additional_kwargs, {})

    def test_backend_options_and_hash_keep_generic_and_draft_separate(self):
        cfg = NS(enable_static_kernel=True, enable_npugraph_ex=True, enable_super_kernel=True)
        vc = NS(
            model_config=NS(runner_type="generate"),
            speculative_config=None,
            scheduler_config=NS(max_num_seqs=16),
            compilation_config=NS(cudagraph_capture_sizes=[1, 16, 32]),
        )
        named = [True]
        ns = load(
            "vllm_ascend/compilation/compiler_interface.py",
            {"_configure_backend", "_compute_decode_cudagraph_batch_sizes"},
            dict(
                logger=Mock(),
                get_ascend_config=lambda: cfg,
                use_kimi_moe_superkernel=lambda *args: named[0],
                moe_superkernel_options=self.ns["moe_superkernel_options"],
            ),
        )
        load("vllm_ascend/compilation/compiler_interface.py", {"compute_hash"}, ns, "AscendCompiler")
        ns["get_ascend_config"] = lambda: NS(ascend_compilation_config=cfg)
        hashes = []
        with patch.dict(sys.modules, {"torch_npu": NS(__version__="test")}):
            for use_named, draft in ((True, False), (True, True), (False, False)):
                named[0] = use_named
                vc.model_config.runner_type = "draft" if draft else "generate"
                process = Mock()
                ns["_configure_backend"](object(), cfg, vc, process)
                options = process.call_args.args[1]["options"]
                self.assertTrue(options["static_kernel_compile"])
                if draft:
                    self.assertNotIn("super_kernel_optimize", options)
                else:
                    self.assertTrue(options["super_kernel_optimize"])
                    expected = self.ns["moe_superkernel_options"]() if use_named else {"dcci_after_kernel_end": [".*"]}
                    self.assertEqual(options["super_kernel_optimize_options"], expected)
                hashes.append(ns["compute_hash"](NS(), vc))
        self.assertEqual(len(set(hashes)), 3)
        named[0] = True
        with self.assertRaisesRegex(ValueError, "legacy"):
            ns["_configure_backend"](object(), cfg, vc)

    def test_acl_capture_optimizes_once_and_skips_draft_or_unmarked(self):
        for enabled, marked, draft in (
            (True, True, False),
            (True, False, False),
            (False, False, False),
            (True, True, True),
        ):
            with self.subTest(enabled=enabled, marked=marked, draft=draft):
                order = []
                self.context.batch_descriptor = "batch"
                self.context.cudagraph_runtime_mode = "full"
                graph = Mock()
                graph.super_kernel_optimize.side_effect = lambda order=order, **kw: order.append("optimize")
                graph.replay.side_effect = lambda order=order: order.append("replay")

                @contextmanager
                def capture(*args, order=order, **kwargs):
                    order.append("begin")
                    yield
                    order.append("end")

                def runnable(order=order, marked=marked):
                    with self.ns["moe_superkernel_scope"]("moe", marked):
                        order.append("forward")
                    return "result"

                self.npu.NPUGraph = lambda graph=graph: graph
                self.npu.graph = capture
                ns = dict(
                    self.ns,
                    torch=NS(npu=self.npu, Tensor=type("Tensor", (), {})),
                    CUDAGraphMode=NS(NONE="none", FULL="full"),
                    UpdatableGraph=lambda graph=graph: graph,
                    use_updatable_graph=lambda backend: False,
                    ACLGraphEntry=lambda **kw: NS(**kw, aclgraph=None, output=None),
                    validate_cudagraph_capturing_enabled=lambda: None,
                    ExitStack=ExitStack,
                    super_kernel_scope=lambda *args: nullcontext(),
                    weak_ref_workspaces=lambda _: None,
                    weak_ref_tensors=lambda x: x,
                    _graph_params=None,
                    _draft_graph_params=None,
                    _draft_graph_prefill_params=None,
                    _EXTRA_CTX=NS(is_draft_model=draft),
                    compilation_counter=NS(num_cudagraph_captured=0),
                    _is_old_hdk_capture_error=lambda _: False,
                    _is_stream_resource_capture_error=lambda _: False,
                )
                load("vllm_ascend/compilation/acl_graph.py", {"__call__"}, ns, "ACLGraphWrapper")
                wrapper = NS(
                    runtime_mode="full",
                    graph_pool=None,
                    concrete_aclgraph_entries={},
                    aclgraph_options=NS(debug_log_enable=False, gc_disable=False, weak_ref_output=False),
                    enable_moe_superkernel=enabled,
                    enable_super_kernel=False,
                    is_debugging_mode=False,
                    use_eagle=True,
                    enable_enpu=True,
                    attn_backend=None,
                    runnable=runnable,
                )
                with patch.dict(sys.modules, {"vllm.model_executor.offloader.base": NS(get_offloader=lambda: Mock())}):
                    self.assertEqual(ns["__call__"](wrapper), "result")
                    self.assertEqual(ns["__call__"](wrapper), "result")
                expected = ["begin", "forward", "end"]
                if enabled and marked and not draft:
                    expected.append("optimize")
                    graph.super_kernel_optimize.assert_called_once_with(
                        optimize_options=self.ns["moe_superkernel_options"]()
                    )
                else:
                    graph.super_kernel_optimize.assert_not_called()
                self.assertEqual(order, expected + ["replay"])
                self.assertEqual(self.context.additional_kwargs, {})

    def test_draft_config_marks_returned_copy_without_changing_upstream_context(self):
        ns = {"copy": copy}
        tree = ast.parse((ROOT / "vllm_ascend/spec_decode/llm_base_proposer.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSpecDecodeBaseProposer")
        cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
        cls.body = [n for n in cls.body if getattr(n, "name", None) == "_create_draft_vllm_config"]
        fails = [False]

        class Base:
            def _create_draft_vllm_config(self):
                assert self.vllm_config is target
                assert self.vllm_config.model_config.runner_type == "generate"
                if fails[0]:
                    raise RuntimeError("upstream")
                return self.vllm_config

        ns["Base"] = Base
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), "draft", "exec"), ns)
        proposer = ns["AscendSpecDecodeBaseProposer"]()
        target = NS(model_config=NS(runner_type="generate"), additional_config={"unchanged": True})
        proposer.vllm_config = target
        proposer.speculative_config = NS(draft_model_config=NS(runner_type="draft"))
        draft = proposer._create_draft_vllm_config()
        self.assertIsNot(draft, target)
        self.assertIsNot(draft.model_config, target.model_config)
        self.assertEqual(draft.model_config.runner_type, "draft")
        self.assertIs(draft.additional_config, target.additional_config)
        self.assertIs(proposer.vllm_config, target)
        fails[0] = True
        with self.assertRaisesRegex(RuntimeError, "upstream"):
            proposer._create_draft_vllm_config()
        self.assertIs(proposer.vllm_config, target)
        self.assertEqual(target.model_config.runner_type, "generate")


if __name__ == "__main__":
    unittest.main()
