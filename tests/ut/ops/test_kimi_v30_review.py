# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for storage, serial execution and caller input contracts."""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_class(path, name, methods, ns, base=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.bases = [] if base is None else [ast.Name(id=base, ctx=ast.Load())]
    node.decorator_list = []
    node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns[name]


class TestKimiMigrationReview(unittest.TestCase):
    def test_decode_uses_live_megamoe_weight_views(self):
        cls = load_class(
            "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py",
            "AscendW4A8MXFPDynamicFusedMoEMethod",
            {"_get_mlp_weights"},
            {},
        )
        method = cls()
        method.use_expert_weight_list = False
        names = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]
        layer = NS()
        originals = []
        for name in names:
            # Simulate the released dense weights left by 0.30's MegaMoE loader.
            setattr(layer, name, torch.empty(0))
            weights = [torch.arange(24).reshape(6, 4), torch.ones(6, 4)]
            setattr(layer, "cann_mega_moe_" + name + "_list", weights)
            originals.append(weights)
        for views, weights in zip(method._get_mlp_weights(layer), originals):
            self.assertEqual(len(views), len(weights))
            for view, weight in zip(views, weights):
                torch.testing.assert_close(view, weight.transpose(0, 1))
                self.assertEqual(view.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr())

    def test_real_weight_loader_keeps_prefill_decode_prefill_storage_live(self):
        context = NS(use_mega_moe=True)
        released = []

        def dispose(parameter):
            released.append(parameter)
            parameter.data = torch.empty(0, dtype=parameter.dtype)

        ns = dict(
            torch=torch,
            torch_npu=NS(npu_format_cast=lambda w, *a, **kw: w, float4_e2m1fn_x2=296),
            ACL_FORMAT_FRACTAL_NZ=29,
            use_cann_megamoe=lambda cfg: True,
            get_current_vllm_config=lambda: None,
            dispose_tensor=dispose,
            _EXTRA_CTX=context,
            MoEWeights=NS,
        )
        cls = load_class(
            "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py",
            "AscendW4A8MXFPDynamicFusedMoEMethod",
            {"process_weights_after_loading", "_get_mlp_weights", "get_fused_mc2_weights"},
            ns,
        )
        method = cls()
        method.use_expert_weight_list = False
        layer = torch.nn.Module()
        for name, shape in [
            ("w13_weight", (2, 128, 32)),
            ("w2_weight", (2, 64, 32)),
            ("w13_weight_scale", (2, 128, 2)),
            ("w2_weight_scale", (2, 64, 2)),
        ]:
            layer.register_parameter(name, torch.nn.Parameter(torch.ones(shape, dtype=torch.uint8), False))
        method.process_weights_after_loading(layer)
        self.assertEqual(len(released), 4)
        self.assertTrue(all(parameter.numel() == 0 for parameter in released))
        first = method.get_fused_mc2_weights(layer)
        context.use_mega_moe = False
        decoded = method._get_mlp_weights(layer)
        self.assertEqual(decoded[0][0].shape, (32, 128))
        self.assertEqual(decoded[1][0].shape, (1, 128, 2))
        self.assertEqual(decoded[1][0].dtype, torch.float8_e8m0fnu)
        self.assertEqual(decoded[0][0].untyped_storage().data_ptr(), first.w1[0].untyped_storage().data_ptr())
        context.use_mega_moe = True
        second = method.get_fused_mc2_weights(layer)
        self.assertIs(first.w1, second.w1)
        torch.testing.assert_close(second.w1[0], torch.ones(128, 32, dtype=torch.uint8))

    def test_preprojected_front_does_not_project_again_or_record_serial_events(self):
        projection = Mock(side_effect=lambda x: (x[:, :2] * 2, x))

        class Base:
            def apply_routed_input_transform(self, x):
                return projection(x)

        event_access = Mock(side_effect=AssertionError("serial router recorded an event"))
        ns = dict(
            torch=NS(npu=NS(current_stream=event_access)),
            Base=Base,
            nullcontext=nullcontext,
            npu_stream_switch=lambda *a, **kw: nullcontext(),
            PreparedSharedExpertInput=lambda x: NS(hidden_states=x, ready_event=None),
        )
        cls = load_class(
            "vllm_ascend/ops/fused_moe/fused_moe.py", "AscendMoERunner", {"apply_routed_input_transform"}, ns, "Base"
        )
        runner = cls()
        runner.ascend_shared_experts = None
        runner._kimi_gate = object()
        runner._kimi_router_stream = None
        runner.routed_input_transform = object()
        runner._can_overlap_sp_shared_with = lambda t: False
        runner._can_run_kimi_decode = lambda x: False
        runner._compute_router_logits = Mock(return_value=torch.ones(2, 8))
        original = torch.ones(2, 4)
        projected = torch.full((2, 2), 3.0)
        actual, shared = runner.apply_routed_input_transform(original, routed_input=projected)
        self.assertIs(actual, projected)
        self.assertIs(shared, original)
        projection.assert_not_called()
        runner._compute_router_logits.assert_called_once_with(original, original)
        event_access.assert_not_called()

    def test_serial_shared_forward_never_acquires_auxiliary_stream(self):
        stream = Mock(side_effect=AssertionError("serial execution acquired an auxiliary stream"))
        modes = NS(TENSOR_PARALLEL=0, SHARED_EXPERT_DATA_PARALLEL_ONLY=1, SEQUENCE_PARALLEL_ONLY=2)
        paths = NS(A8_INT_FUSED=0, A8_MXFP_FUSED=1, LINEAR_WRAPPER=2)
        ns = dict(
            nullcontext=nullcontext,
            shared_experts_calculation_stream=stream,
            npu_stream_switch=lambda *a, **kw: nullcontext(),
            SharedExpertParallelMode=modes,
            SharedExpertMLPPath=paths,
            PreparedSharedExpertExecution=NS,
            SubmittedSharedExpertGateUp=NS,
        )
        cls = load_class(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            "AscendSharedExperts",
            {
                "forward",
                "enqueue_gate_up_after_router",
                "_wait_for_milestone",
                "_compute_stream_context",
                "wait_for_output",
            },
            ns,
        )
        shared = cls()
        shared.multistream_overlap = False
        shared.parallel_mode = lambda: modes.TENSOR_PARALLEL
        shared._select_mlp_path = lambda: paths.LINEAR_WRAPPER
        x = torch.tensor([[1.0, 2.0]])
        shared._prepare_execution_input = lambda *a: (x, None)
        shared._prepare_gate_up_input = lambda x: x
        shared.part1 = lambda x: x * 2
        shared._finish_shared_mlp = lambda submitted, milestones: submitted.gate_up + 1
        output = shared.forward(NS(hidden_states=x), NS(router_output_ready=None))
        torch.testing.assert_close(output, x * 2 + 1)
        shared.wait_for_output()
        stream.assert_not_called()

    def test_model_uses_normalized_shared_situ_beta(self):
        cfg = NS(
            hidden_size=4,
            moe_intermediate_size=4,
            num_experts=2,
            num_experts_per_token=1,
            routed_expert_hidden_size=None,
            latent_moe_use_norm=False,
            routed_scaling_factor=1.0,
            num_shared_experts=1,
            activation_situ_beta=None,
            activation_situ_linear_beta=25.0,
            hidden_act="situ",
            moe_renormalize=True,
            use_grouped_topk=False,
            num_expert_group=None,
            topk_group=None,
            moe_router_activation_func="softmax",
        )
        shared = torch.nn.Module()
        shared.act_fn = NS(beta=1.0, linear_beta=25.0)
        runner = torch.nn.Module()
        runner.moe_config = NS(num_logical_experts=2, num_experts=2, num_local_experts=2)
        sparse = Mock(side_effect=lambda beta, alpha: NS(beta=float(beta), linear_beta=alpha))
        ns = dict(
            torch=torch,
            nn=torch.nn,
            Module=torch.nn.Module,
            GateLinear=lambda **kw: torch.nn.Module(),
            KimiMLP=lambda **kw: shared,
            FusedMoEFactory=lambda **kw: runner,
            AscendSharedSparseSituAndMul=sparse,
            get_current_hardware_profile=lambda: NS(supports=lambda cap: True),
            HardwareCapability=NS(CANN_MEGAMOE_MXFP=1),
            get_ascend_config=lambda: NS(multistream_overlap_shared_expert=False),
        )
        cls = load_class("vllm_ascend/models/kimi_k3.py", "AscendKimiMoE", {"__init__"}, ns, "Module")
        cls(config=cfg, vllm_config=NS(parallel_config=NS(enable_eplb=False)))
        sparse.assert_called_once_with(1.0, 25.0)

    def test_public_moe_op_keeps_preprojected_input(self):
        op = Mock(return_value=torch.zeros(2, 4))
        ns = dict(torch=NS(ops=NS(vllm=NS(ascend_kimi_moe=op))))
        cls = load_class("vllm_ascend/ops/fused_moe/fused_moe.py", "AscendMoERunner", {"forward"}, ns)
        runner = cls()
        runner._kimi_gate = object()
        runner._encode_layer_name = lambda: "layer"
        projected = torch.ones(2, 2)
        original = torch.ones(2, 4)
        scores = torch.ones(2, 8)
        runner.forward(projected, scores, shared_experts_input=original)
        self.assertTrue(
            any(arg is original for arg in op.call_args.args), "the opaque operation dropped the original shared input"
        )
        self.assertTrue(any(arg is scores for arg in op.call_args.args))

    def test_moe_compilation_preserves_preprojected_shape_and_router_inputs(self):
        path = ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py"
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if getattr(n, "name", None) in {"_ascend_kimi_moe", "_ascend_kimi_moe_fake"}]
        run = Mock(
            side_effect=lambda hidden, logits, ids, shared: shared
            + hidden.sum(-1, keepdim=True)
            + logits.sum(-1, keepdim=True)
        )
        runner = NS(_forward_standard=run)
        resolve = Mock(return_value=runner)
        namespace = dict(
            torch=torch, _layer_name_type=str, get_layer_from_name=resolve, _resolve_layer_name=lambda name: name
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
        library = torch.library.Library("kimi_moe_review", "FRAGMENT")
        op = namespace["_ascend_kimi_moe"]
        library.define("forward" + torch.library.infer_schema(op, mutates_args=[]))
        library.impl("forward", op, "CPU")
        library._register_fake("forward", namespace["_ascend_kimi_moe_fake"])

        def forward(projected, shared, scores):
            return torch.ops.kimi_moe_review.forward(projected, None, "layer", shared, scores)

        torch._dynamo.reset()
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(forward, backend=backend, fullgraph=True)
        projected = torch.ones(2, 2)
        shared = torch.ones(2, 4)
        scores = torch.ones(2, 8)
        for multiplier in (1, 2):
            actual = compiled(projected, shared, scores * multiplier)
            torch.testing.assert_close(actual, shared + 2 + 8 * multiplier)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(run.call_count, 2)
        self.assertIsNone(runner._kimi_router_logits)


if __name__ == "__main__":
    unittest.main()
