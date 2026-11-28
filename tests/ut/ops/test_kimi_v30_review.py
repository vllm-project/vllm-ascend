# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for storage, serial execution and caller input contracts."""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from test_kimi_v30_moe import load

ROOT = Path(__file__).resolve().parents[3]


def load_class(path, name, methods, ns, base=None):
    ns.setdefault("moe_superkernel_scope", lambda *args: nullcontext())
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

    def test_eplb_megamoe_scales_share_live_migration_storage(self):
        context = NS(use_mega_moe=True)
        namespace = dict(
            torch=torch,
            torch_npu=NS(npu_format_cast=lambda weight, *args, **kwargs: weight, float4_e2m1fn_x2=296),
            ACL_FORMAT_FRACTAL_NZ=29,
            MoEActivation=NS(SITU="situ"),
            _EXTRA_CTX=context,
            MoEWeights=NS,
        )
        cls = load_class(
            "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py",
            "AscendW4A8MXFPDynamicFusedMoEMethod",
            {"process_weights_after_loading", "_get_mlp_weights", "get_fused_mc2_weights", "get_eplb_weight_views"},
            namespace,
        )
        for activation in ("silu", "situ"):
            with self.subTest(activation=activation):
                method = cls()
                method.use_expert_weight_list = True
                layer = torch.nn.Module()
                layer.activation = activation
                local_experts = (224 + 8) // 8
                for name, shape in (
                    ("w13_weight", (local_experts, 128, 32)),
                    ("w2_weight", (local_experts, 64, 32)),
                    ("w13_weight_scale", (local_experts, 128, 2)),
                    ("w2_weight_scale", (local_experts, 64, 2)),
                ):
                    layer.register_parameter(name, torch.nn.Parameter(torch.ones(shape, dtype=torch.uint8), False))
                with patch.object(torch, "npu", NS(empty_cache=Mock()), create=True):
                    method.process_weights_after_loading(layer)
                context.use_mega_moe = True
                before = method.get_fused_mc2_weights(layer)
                migration_weights = method.get_eplb_weight_views(layer)
                self.assertIs(before.w1, migration_weights[0])
                self.assertIs(before.w2, migration_weights[1])
                for scales, source in (
                    (before.w1_scale, migration_weights[2]),
                    (before.w2_scale, migration_weights[3]),
                ):
                    self.assertEqual(len(scales), local_experts)
                    for scale, original in zip(scales, source):
                        self.assertEqual(scale.dtype, torch.float8_e8m0fnu)
                        self.assertEqual(original.dtype, torch.uint8)
                        self.assertEqual(scale.untyped_storage().data_ptr(), original.untyped_storage().data_ptr())
                for weights in migration_weights:
                    weights[-1].view(torch.uint8).fill_(127)
                context.use_mega_moe = False
                decoded = method.get_fused_mc2_weights(layer)
                self.assertEqual(decoded.w1_scale[0].dtype, torch.uint8)
                self.assertEqual(decoded.w1_scale[0].shape, (1, 128, 2))
                context.use_mega_moe = True
                after = method.get_fused_mc2_weights(layer)
                for weights in (before, decoded, after):
                    for tensors, source in zip(
                        (weights.w1, weights.w2, weights.w1_scale, weights.w2_scale), migration_weights
                    ):
                        self.assertEqual(len(tensors), local_experts)
                        self.assertEqual(
                            tensors[-1].untyped_storage().data_ptr(), source[-1].untyped_storage().data_ptr()
                        )
                        self.assertTrue(torch.all(tensors[-1].view(torch.uint8) == 127))

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
            initialize_kimi_situ_ops=Mock(),
            get_current_hardware_profile=lambda: NS(supports=lambda cap: True),
            HardwareCapability=NS(CANN_MEGAMOE_MXFP=1),
            get_ascend_config=lambda: NS(multistream_overlap_shared_expert=False),
        )
        load("vllm_ascend/models/kimi_k3.py", {"_get_kimi_router_stream"}, ns)
        cls = load_class("vllm_ascend/models/kimi_k3.py", "AscendKimiMoE", {"__init__"}, ns, "Module")
        cls(config=cfg, vllm_config=NS(parallel_config=NS(enable_eplb=False)))
        sparse.assert_called_once_with(1.0, 25.0)
        ns["initialize_kimi_situ_ops"].assert_called_once_with()

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
            torch=torch,
            _layer_name_type=str,
            get_layer_from_name=resolve,
            _resolve_layer_name=lambda name: name,
            get_ascend_config=lambda: NS(ascend_compilation_config=NS(enable_super_kernel=False)),
            moe_superkernel_scope=lambda *args: nullcontext(),
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

    def test_mxfp_shared_keeps_sparse_activation_in_all_execution_modes(self):
        class Sparse:
            beta = 4.0
            linear_beta = 25.0

        fused = Mock(side_effect=AssertionError("Kimi sparse SiTU was bypassed by fused quantization"))
        ns = dict(
            AscendSharedSparseSituAndMul=Sparse,
            torch=NS(ops=NS(_C_ascend=NS(situ_mx_quant=fused))),
            SITU_MX_DST_TYPE_E4M3FN=36,
        )
        cls = load_class("vllm_ascend/ops/fused_moe/shared_experts.py", "AscendSharedExperts", {"_run_a8_mxfp_mlp"}, ns)
        shared = cls()
        shared.situ_activation = Sparse()
        activated = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        shared.apply_activation = Mock(return_value=activated)
        shared._wait_for_routed_stage = Mock()
        shared.part2 = Mock(return_value=activated + 1)
        submitted = NS(gate_up=torch.ones(1, 4, dtype=torch.bfloat16), hidden_states=torch.ones(1, 2))
        for mode in ("prefill", "serial", "ordinary_decode"):
            with self.subTest(mode=mode):
                milestones = NS(shared_activation_overlap_start=object(), routed_combine_start=object())
                output = shared._run_a8_mxfp_mlp(submitted, milestones)
                torch.testing.assert_close(output, activated + 1)
                self.assertIs(shared.part2.call_args.args[1], activated)
        self.assertEqual(shared.apply_activation.call_count, 3)
        fused.assert_not_called()

    def test_both_runners_admit_decode_and_reject_unsupported_schemes(self):
        from unittest.mock import patch

        class Routed:
            pass

        class Linear:
            pass

        class Latent:
            pass

        class Router:
            pass

        modules = {
            "vllm_ascend.ops.fused_moe.router.fused_topk_router": NS(AscendFusedTopKRouter=Router),
            "vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4": NS(
                AscendW4A8MXFPDynamicFusedMoEMethod=Routed, AscendW4A8MXFPDynamicLinearMethod=Linear
            ),
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8": NS(AscendW8A8MXFP8DynamicLinearMethod=Latent),
        }
        context = NS(attn_metadata={"layer": NS(num_prefills=0)})
        ns = dict(
            torch=torch,
            get_forward_context=lambda: context,
            get_current_hardware_profile=lambda: NS(supports=lambda cap: True),
            HardwareCapability=NS(CANN_MEGAMOE_MXFP=1),
            _EXTRA_CTX=NS(moe_comm_type="mc2", is_draft_model=False),
            MoECommType=NS(MC2="mc2"),
            QuantType=NS(W4A8MXFP="fp4"),
        )
        cls = load_class("vllm_ascend/ops/fused_moe/fused_moe.py", "AscendMoERunner", {"_can_run_kimi_decode"}, ns)
        runner = cls()
        runner.quant_type = "fp4"
        runner.moe_config = NS(is_sequence_parallel=True)
        runner._can_overlap_sp_shared_with = lambda transform: True
        runner.routed_input_transform = NS(quant_method=NS(quant_method=Latent()))
        runner.routed_output_transform = object()
        router = Router()
        router.custom_routing_function = None
        router.tid2eid = None
        router.bias_vl = None
        scheme = Routed()
        scheme.use_expert_weight_list = False
        runner.routed_experts = NS(
            quant_method=NS(quant_method=scheme), router=router, dynamic_eplb=False, _use_v2_model_runner=False
        )
        layer = NS(
            gate_up_proj=NS(weight=torch.ones(2, 2, dtype=torch.bfloat16)),
            down_proj=NS(weight=torch.ones(2, 2, dtype=torch.bfloat16)),
        )
        runner.ascend_shared_experts = NS(layer=layer, situ_activation=object())
        x = torch.ones(2, 2, dtype=torch.bfloat16)
        with patch.dict("sys.modules", modules):
            for runner_v2 in (False, True):
                runner.routed_experts._use_v2_model_runner = runner_v2
                self.assertTrue(runner._can_run_kimi_decode(x))
                # Hybrid KDA/MLA metadata and MTP verification are Decode
                # when every attention group reports zero prefill requests.
                context.attn_metadata = {"kda": NS(num_prefills=0), "mla": NS(num_prefills=0)}
                self.assertTrue(runner._can_run_kimi_decode(x))
                for metadata in (
                    None,
                    {},
                    {"layer": NS()},
                    {"layer": NS(num_prefills=1)},
                    {"kda": NS(num_prefills=0), "mla": NS(num_prefills=1)},
                ):
                    context.attn_metadata = metadata
                    self.assertFalse(runner._can_run_kimi_decode(x), (runner_v2, metadata))
                context.attn_metadata = {"layer": NS(num_prefills=0)}
            for obj, attribute, value in [
                (router, "bias_vl", object()),
                (ns["_EXTRA_CTX"], "is_draft_model", True),
                (runner.moe_config, "is_sequence_parallel", False),
                (runner.routed_experts, "dynamic_eplb", True),
                (ns["_EXTRA_CTX"], "moe_comm_type", "alltoall"),
                (scheme, "use_expert_weight_list", True),
                (router, "custom_routing_function", lambda: None),
                (router, "tid2eid", object()),
                (layer, "expert_gate", object()),
                (runner.routed_experts, "_ascend_moe_lora_context", object()),
            ]:
                old = getattr(obj, attribute, None)
                setattr(obj, attribute, value)
                self.assertFalse(runner._can_run_kimi_decode(x), attribute)
                setattr(obj, attribute, old)
            runner.routed_input_transform.quant_method.quant_method = object()
            self.assertFalse(runner._can_run_kimi_decode(x))


if __name__ == "__main__":
    unittest.main()
