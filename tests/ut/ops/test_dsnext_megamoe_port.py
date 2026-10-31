"""Exercise the port's host decisions without importing vLLM or an NPU runtime.

The AST loader executes the actual production functions with explicit dependency
doubles. Run this file directly with Python, or collect it with the unit suite.
These checks do not execute CANN operators or validate numerical precision.
"""

import ast
import inspect
import math
import sys
import unittest
from copy import copy
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    definitions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            definitions.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions.extend(
                child for child in node.body if isinstance(child, ast.FunctionDef) and child.name in names
            )
    if {node.name for node in definitions} != set(names):
        raise AssertionError("Requested production functions were not found")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    isolated = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace


class Quant(Enum):
    NONE = auto()
    W4A8MXFP = auto()
    W8A8MXFP = auto()
    W4A4MXFP = auto()


class Comm(Enum):
    FUSED_MC2 = auto()
    MC2 = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()


class TensorDouble:
    def __init__(self, shape, dtype="uint8", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.contiguous = contiguous

    @property
    def ndim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous

    def stride(self):
        return ()

    def view(self, dtype):
        return TensorDouble(self.shape, dtype, self.contiguous)


class TestMegaMoEPort(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            parallel_config=SimpleNamespace(world_size_across_dp=8, tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[], max_cudagraph_capture_size=0),
            kv_transfer_config=None,
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(top_k_experts=4)),
        )
        self.ascend_config = SimpleNamespace(mega_moe_max_tokens=65536, enable_fused_mc2=1)
        self.host = load_functions("vllm_ascend/ascend_config.py", ["compute_mega_moe_buffer_tokens_per_rank"], {})
        self.host.update(
            get_ascend_config=lambda: self.ascend_config,
            get_mc2_tokens_capacity=lambda: 128,
            is_mega_moe_supported=lambda: True,
            QuantType=Quant,
            MoECommType=Comm,
            A5_SUPPORT_MEGA_MOE_QUANT_TYPES={Quant.W4A8MXFP, Quant.W8A8MXFP, Quant.W4A4MXFP},
            _MC2_TOKENS_PER_RANK_LIMIT=512,
        )
        load_functions(
            "vllm_ascend/ascend_forward_context.py",
            [
                "has_different_draft_moe_topology",
                "get_a5_mega_moe_buffer_tokens_per_rank",
                "_select_capacity_and_world_size_moe_comm_method",
            ],
            self.host,
        )
        self.shared = load_functions(
            "vllm_ascend/ops/fused_moe/mega_moe.py",
            [
                "supports_mega_moe_shared_experts",
                "_as_tensor_list",
                "_view_mxfp_scales_as_e8m0",
                "_validate_shared_mxfp_layout",
                "build_mega_moe_shared_kwargs",
            ],
            {
                "signature": inspect.signature,
                "Parameter": inspect.Parameter,
                "logger": Mock(),
                "torch": SimpleNamespace(uint8="uint8"),
                "QuantType": Quant,
                "_TORCH_FLOAT8_E8M0FNU_DTYPE": "e8m0",
                "_MEGA_MOE_SUPPORTED_SHARED_QUANTS": {Quant.W4A8MXFP, Quant.W8A8MXFP},
                "_FP4_PACK_FACTOR": 2,
                "_MXFP_SCALE_BLOCK_SIZE": 64,
                "_MXFP_SCALE_MULTIPLIER": 2,
                "_FLOAT8_E4M3FN_WEIGHT_TYPE": 24,
            },
        )

    def capacity(self):
        return self.host["get_a5_mega_moe_buffer_tokens_per_rank"](self.config, 128)

    def initialize_mc2_capacity(self, *, a5=True, prefill=True, max_reqs=512, query_len=1):
        self.ascend_config.enable_prefill_mc2 = prefill
        self.host.update(
            _mc2_tokens_capacity=None,
            use_cann_megamoe=lambda _: True,
            get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: a5),
            HardwareCapability=SimpleNamespace(CANN_MEGAMOE_MXFP=object()),
            _is_decode_only_node=lambda _: False,
            _MEGA_MOE_TOKENS_PER_RANK_LIMIT=16384,
            _DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT=512,
        )
        load_functions(
            "vllm_ascend/ascend_forward_context.py",
            ["set_mc2_tokens_capacity", "get_mc2_tokens_capacity"],
            self.host,
        )
        self.host["set_mc2_tokens_capacity"](self.config, max_reqs, query_len)
        return self.host["get_mc2_tokens_capacity"]()

    def test_a5_mc2_initialization_keeps_megamoe_capacity_independent(self):
        self.config.scheduler_config.max_num_batched_tokens = 16384
        capacity = self.initialize_mc2_capacity()
        self.assertEqual(capacity, 512)
        self.assertEqual(self.host["get_a5_mega_moe_buffer_tokens_per_rank"](self.config), 8192)

    def test_a5_mc2_initialization_preserves_graph_and_decode_budgets(self):
        self.config.compilation_config.cudagraph_capture_sizes = [1, 96]
        self.config.compilation_config.max_cudagraph_capture_size = 96
        self.assertEqual(self.initialize_mc2_capacity(prefill=False), 96)
        self.config.compilation_config.cudagraph_capture_sizes = []
        self.assertEqual(self.initialize_mc2_capacity(prefill=False, max_reqs=33, query_len=3), 99)

    def test_a5_mc2_initialization_aligns_tp_and_bounds_each_rank(self):
        self.config.parallel_config.tensor_parallel_size = 2
        self.config.scheduler_config.max_num_batched_tokens = 513
        self.assertEqual(self.initialize_mc2_capacity(), 514)
        self.config.scheduler_config.max_num_batched_tokens = 16384
        self.assertEqual(self.initialize_mc2_capacity(), 1024)

    def test_a3_megamoe_initialization_preserves_large_capacity(self):
        self.config.scheduler_config.max_num_batched_tokens = 16384
        self.assertEqual(self.initialize_mc2_capacity(a5=False, prefill=False), 16384)

    def test_mixed_pd_uses_prefill_capacity(self):
        self.assertEqual(self.capacity(), 4096)
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_both")
        self.assertEqual(self.capacity(), 4096)

    def test_producer_uses_prefill_capacity(self):
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_producer")
        self.assertEqual(self.capacity(), 4096)

    def test_consumer_uses_decode_capacity(self):
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
        self.assertEqual(self.capacity(), 128)

    def test_global_ceiling_bounds_each_ep_rank(self):
        self.ascend_config.mega_moe_max_tokens = 1024
        self.assertEqual(self.capacity(), 128)

    def test_invalid_capacities_fail_before_collectives(self):
        compute = self.host["compute_mega_moe_buffer_tokens_per_rank"]
        for ceiling, execution, ep in [(0, 128, 8), (4, 128, 8), (1024, 0, 8), (1024, 128, 0)]:
            with self.subTest(ceiling=ceiling, execution=execution, ep=ep), self.assertRaises(ValueError):
                compute(ceiling, execution, ep)

    def test_buffer_overflow_falls_back_to_existing_communication(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.assertEqual(select(4096, self.config, 128), Comm.FUSED_MC2)
        self.assertEqual(select(4097, self.config, 128), Comm.ALLTOALL)
        self.ascend_config.mega_moe_max_tokens = 512
        self.assertEqual(select(128, self.config, 128), Comm.MC2)

    def test_initialized_mc2_capacity_prevents_oversized_fallback(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.config.scheduler_config.max_num_batched_tokens = 16384
        self.config.model_config.hf_text_config = SimpleNamespace(num_experts_per_tok=6)
        capacity = self.initialize_mc2_capacity()
        # MegaMoE can accept 8192 tokens, but ordinary dispatch cannot accept
        # the 9000-token request or the startup's 16384-token dummy forward.
        self.assertEqual(select(8192, self.config, capacity), Comm.FUSED_MC2)
        for tokens in (8193, 9000, 16384):
            with self.subTest(tokens=tokens):
                self.assertEqual(select(tokens, self.config, capacity), Comm.ALLTOALL)
        self.config.parallel_config.world_size_across_dp = 4
        self.ascend_config.mega_moe_max_tokens = 32768
        self.assertEqual(select(9000, self.config, capacity), Comm.ALLGATHER)

    def test_small_megamoe_buffer_falls_back_only_within_dispatch_capacity(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.ascend_config.mega_moe_max_tokens = 2048
        capacity = self.initialize_mc2_capacity()
        for tokens, expected in (
            (256, Comm.FUSED_MC2),
            (257, Comm.MC2),
            (512, Comm.MC2),
            (513, Comm.ALLTOALL),
        ):
            with self.subTest(tokens=tokens):
                self.assertEqual(select(tokens, self.config, capacity), expected)
        # A smaller initialized capacity remains binding.
        self.assertEqual(select(401, self.config, 400), Comm.ALLTOALL)

    def test_mc2_fallback_capacity_accounts_for_tp_partition(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.ascend_config.enable_fused_mc2 = 0
        self.config.parallel_config.tensor_parallel_size = 2
        capacity = self.initialize_mc2_capacity()
        self.assertEqual(select(1024, self.config, capacity), Comm.MC2)
        self.assertEqual(select(1025, self.config, capacity), Comm.ALLTOALL)
        self.assertEqual(select(601, self.config, 600), Comm.ALLTOALL)

    def test_draft_and_consumer_fallbacks_obey_dispatch_limit(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        capacity = self.initialize_mc2_capacity()
        self.assertEqual(select(513, self.config, capacity, True, Quant.NONE), Comm.ALLTOALL)
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
        self.assertEqual(select(512, self.config, capacity), Comm.MC2)
        self.assertEqual(select(513, self.config, capacity), Comm.ALLTOALL)

    def test_unquantized_draft_preserves_original_fallback(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.assertEqual(select(128, self.config, 128, True, Quant.NONE), Comm.MC2)
        self.assertEqual(select(4096, self.config, 128, True, Quant.NONE), Comm.ALLTOALL)

    def test_newer_main_quantizations_are_preserved(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        for quant in (Quant.W4A8MXFP, Quant.W8A8MXFP, Quant.W4A4MXFP):
            with self.subTest(quant=quant):
                self.assertEqual(select(128, self.config, 128, False, quant), Comm.FUSED_MC2)

    def test_pd_consumer_preserves_all_ordinary_communication_paths(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.config.kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
        capacity = Mock(side_effect=AssertionError("PD consumer must not resolve MegaMoE capacity"))
        self.host["get_a5_mega_moe_buffer_tokens_per_rank"] = capacity
        for tokens, world_size, expected in (
            (128, 8, Comm.MC2),
            (4096, 8, Comm.ALLTOALL),
            (4096, 4, Comm.ALLGATHER),
            (None, 8, Comm.MC2),
        ):
            with self.subTest(tokens=tokens, world_size=world_size):
                self.config.parallel_config.world_size_across_dp = world_size
                self.assertEqual(select(tokens, self.config, 128), expected)
        capacity.assert_not_called()

    def test_pd_producer_and_mixed_roles_preserve_megamoe_selection(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        for role in ("kv_producer", "kv_both"):
            with self.subTest(role=role):
                self.config.kv_transfer_config = SimpleNamespace(kv_role=role)
                self.assertEqual(select(128, self.config, 128), Comm.FUSED_MC2)
                self.assertEqual(select(4096, self.config, 128), Comm.FUSED_MC2)
        self.config.kv_transfer_config = None
        self.assertEqual(select(4096, self.config, 128), Comm.FUSED_MC2)

    def test_quantized_draft_uses_original_communication_without_allocating_megamoe(self):
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        self.config.model_config.hf_text_config = SimpleNamespace(n_routed_experts=384, num_experts_per_tok=6)
        self.config.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(n_routed_experts=128, num_experts_per_tok=3)
            )
        )
        capacity = Mock(side_effect=AssertionError("Draft must not resolve MegaMoE capacity"))
        self.host["get_a5_mega_moe_buffer_tokens_per_rank"] = capacity
        for quant in Quant:
            for tokens, world_size, expected in (
                (128, 8, Comm.MC2),
                (4096, 8, Comm.ALLTOALL),
                (4096, 4, Comm.ALLGATHER),
                (None, 8, Comm.MC2),
            ):
                with self.subTest(quant=quant, tokens=tokens, world_size=world_size):
                    self.config.parallel_config.world_size_across_dp = world_size
                    self.assertEqual(select(tokens, self.config, 128, True, quant), expected)
        capacity.assert_not_called()

    def test_matching_draft_topology_preserves_megamoe_for_supported_quants(self):
        self.config.model_config.hf_text_config = SimpleNamespace(n_routed_experts=384, num_experts_per_tok=6)
        self.config.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(n_routed_experts=384, num_experts_per_tok=6)
            )
        )
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        for quant in (Quant.W4A8MXFP, Quant.W8A8MXFP, Quant.W4A4MXFP):
            with self.subTest(quant=quant):
                self.assertEqual(select(128, self.config, 128, True, quant), Comm.FUSED_MC2)

    def test_draft_topology_checks_each_dimension_and_preserves_missing_configs(self):
        check = self.host["has_different_draft_moe_topology"]
        self.config.model_config.hf_text_config = SimpleNamespace(n_routed_experts=384, num_experts_per_tok=6)
        for experts, topk, expected in ((384, 6, False), (128, 6, True), (384, 3, True), (128, 3, True)):
            with self.subTest(experts=experts, topk=topk):
                self.config.speculative_config = SimpleNamespace(
                    draft_model_config=SimpleNamespace(
                        hf_text_config=SimpleNamespace(n_routed_experts=experts, num_experts_per_tok=topk)
                    )
                )
                self.assertEqual(check(self.config, True), expected)
                self.assertFalse(check(self.config, False))
        self.config.speculative_config.draft_model_config.hf_text_config = SimpleNamespace(n_routed_experts=128)
        self.assertFalse(check(self.config, True))
        self.config.speculative_config = None
        self.assertFalse(check(self.config, True))

    def test_draft_weight_loading_preserves_non_megamoe_layout(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                runner_type="generate", hf_text_config=SimpleNamespace(n_routed_experts=384, num_experts_per_tok=6)
            ),
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(n_routed_experts=128, num_experts_per_tok=3)
                )
            ),
            parallel_config=SimpleNamespace(enable_expert_parallel=True, tensor_parallel_size=1),
        )
        probe = load_functions(
            "vllm_ascend/ascend_forward_context.py",
            ["has_different_draft_moe_topology", "use_cann_megamoe"],
            {
                "is_mega_moe_supported": lambda: True,
                "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
                "HardwareCapability": SimpleNamespace(CANN_MEGAMOE=object()),
                "get_ascend_config": lambda: self.ascend_config,
                "is_moe_model": lambda _: True,
                "get_ep_group": lambda: SimpleNamespace(world_size=8),
            },
        )["use_cann_megamoe"]
        self.assertTrue(probe(config))
        make_config = load_functions(
            "vllm_ascend/ascend_forward_context.py", ["make_draft_moe_vllm_config"], {"copy": copy}
        )["make_draft_moe_vllm_config"]
        draft = make_config(config)
        self.assertFalse(probe(draft))
        select = self.host["_select_capacity_and_world_size_moe_comm_method"]
        draft.parallel_config.world_size_across_dp = 8
        self.assertEqual(select(128, draft, 128, False, Quant.W4A8MXFP), Comm.MC2)
        self.assertTrue(probe(config))
        config.model_config.runner_type = "draft"
        self.assertFalse(probe(config))
        config.speculative_config.draft_model_config.hf_text_config = SimpleNamespace(
            n_routed_experts=384, num_experts_per_tok=6
        )
        self.assertTrue(probe(config))

    def test_v2_dspark_config_marks_draft_without_mutating_target(self):
        make_config = load_functions(
            "vllm_ascend/ascend_forward_context.py", ["make_draft_moe_vllm_config"], {"copy": copy}
        )["make_draft_moe_vllm_config"]
        self.config.model_config.runner_type = "generate"
        self.config.model_config.hf_text_config = SimpleNamespace(n_routed_experts=384, num_experts_per_tok=6)
        self.config.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(n_routed_experts=128, num_experts_per_tok=3)
            )
        )
        draft = make_config(self.config)
        self.assertIsNot(draft, self.config)
        self.assertIsNot(draft.model_config, self.config.model_config)
        self.assertEqual(self.config.model_config.runner_type, "generate")
        self.assertEqual(draft.model_config.runner_type, "generate")
        self.assertEqual(draft.model_config._ascend_draft_moe_topology, (128, 3))
        self.assertIs(draft.model_config.hf_text_config, self.config.model_config.hf_text_config)
        self.assertTrue(self.host["has_different_draft_moe_topology"](draft))
        self.assertFalse(self.host["has_different_draft_moe_topology"](self.config))
        del draft.model_config.runner_type
        self.assertTrue(self.host["has_different_draft_moe_topology"](draft))
        draft.model_config._ascend_draft_moe_topology = (384, 6)
        self.assertFalse(self.host["has_different_draft_moe_topology"](draft))

    def test_v2_platform_uses_draft_identity_for_communication(self):
        tree = ast.parse((ROOT / "vllm_ascend/platform.py").read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "set_additional_forward_context"
        )
        # Inject dependencies at the platform boundary; run the actual hook
        # without importing torch/vLLM or requiring device hardware.
        method.decorator_list = []
        method.body = [node for node in method.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        select = Mock(return_value=Comm.MC2)
        namespace = {
            "math": math,
            "CUDAGraphMode": SimpleNamespace(NONE=object()),
            "get_dynamic_mx_quant_scale_alg": lambda _: 0,
            "get_mrv2_in_profile_run": lambda: True,
            "get_tensor_model_parallel_world_size": lambda: 1,
            "get_dp_group": lambda: SimpleNamespace(world_size=1),
            "is_moe_model": lambda _: True,
            "select_moe_comm_method": select,
            "get_moe_comm_method": lambda _: object(),
            "get_mc2_mask": lambda: None,
            "use_cann_megamoe": lambda config: not isinstance(
                getattr(config.model_config, "_ascend_draft_moe_topology", None), tuple
            ),
            "_is_decode_only_node": lambda _: False,
            "QuantType": Quant,
        }
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), "platform.py", "exec"
            ),
            namespace,
        )
        self.config.use_v2_model_runner = True
        self.config.parallel_config.tensor_parallel_size = 1
        self.config.parallel_config.prefill_context_parallel_size = 1
        self.config.speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(draft_moe_quant_type=Quant.W4A8MXFP)
        )
        for is_draft in (False, True, False):
            self.config.model_config._ascend_draft_moe_topology = (128, 3) if is_draft else None
            result = namespace["set_additional_forward_context"](None, None, self.config, None, num_tokens=80)
            self.assertEqual(result["is_draft_model"], is_draft)
            self.assertEqual(result["use_mega_moe"], not is_draft)
            self.assertEqual(select.call_args.kwargs["is_draft_model"], is_draft)

    def test_shared_operator_requires_explicit_keyword_abi(self):
        def supported(
            *,
            shared_l1_weights,
            shared_l2_weights,
            shared_l1_weights_sf,
            shared_l2_weights_sf,
            shared_weight1_type,
            shared_weight2_type,
        ):
            pass

        def legacy(*args, **kwargs):
            pass

        probe = self.shared["supports_mega_moe_shared_experts"]
        self.assertTrue(probe(supported))
        self.assertFalse(probe(legacy))
        self.assertFalse(probe(object()))

    def payload(self, quant=Quant.W4A8MXFP):
        pack = 2 if quant == Quant.W4A8MXFP else 1
        return SimpleNamespace(
            hidden_states=TensorDouble((4, 128)),
            quant=SimpleNamespace(quant_type=Quant.W4A8MXFP),
            shared_weights=SimpleNamespace(
                w1=[TensorDouble((128, 128 // pack))],
                w2=[TensorDouble((128, 64 // pack))],
                w1_scale=[TensorDouble((128, 2, 2))],
                w2_scale=[TensorDouble((128, 1, 2))],
                quant_type=quant,
            ),
        )

    def test_fp8_shared_weights_have_independent_type_from_fp4_routed_weights(self):
        kwargs = self.shared["build_mega_moe_shared_kwargs"](self.payload(Quant.W8A8MXFP), 64, 296)
        self.assertEqual(kwargs["shared_weight1_type"], 24)
        self.assertEqual(kwargs["shared_weight2_type"], 24)
        self.assertEqual(kwargs["shared_l1_weights_sf"][0].dtype, "e8m0")

    def test_fp4_shared_weights_use_routed_fp4_type(self):
        kwargs = self.shared["build_mega_moe_shared_kwargs"](self.payload(), 64, 296)
        self.assertEqual(kwargs["shared_weight1_type"], 296)

    def test_incompatible_layout_fails_before_operator_call(self):
        payload = self.payload()
        payload.shared_weights.w1[0].shape = (1, 256, 64)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296)

    def test_stacked_shared_tensors_are_rejected_for_per_expert_routed_layout(self):
        for name in ("w1", "w2", "w1_scale", "w2_scale"):
            with self.subTest(name=name):
                payload = self.payload()
                tensor = getattr(payload.shared_weights, name)[0]
                tensor.shape = (1, *tensor.shape)
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296)

    def test_noncontiguous_weights_are_rejected(self):
        payload = self.payload()
        payload.shared_weights.w1[0].contiguous = False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296)

    def test_invalid_scale_dtype_is_rejected(self):
        payload = self.payload()
        payload.shared_weights.w2_scale[0].dtype = "float32"
        with self.assertRaisesRegex(RuntimeError, "FLOAT8_E8M0"):
            self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296)

    def test_missing_shared_weights_preserve_routed_only_call(self):
        payload = self.payload()
        payload.shared_weights = None
        self.assertEqual(self.shared["build_mega_moe_shared_kwargs"](payload, 64, 296), {})


class TestFusedSharedRunner(unittest.TestCase):
    def test_communicator_enables_shared_fusion_only_for_supported_megamoe(self):
        context = SimpleNamespace(use_mega_moe=True)
        hardware = SimpleNamespace(supports=Mock(return_value=True))
        probe = Mock(return_value=True)
        method = load_functions(
            "vllm_ascend/ops/fused_moe/moe_comm_method.py",
            ["supports_fused_shared_experts"],
            {
                "_EXTRA_CTX": context,
                "get_current_hardware_profile": lambda: hardware,
                "HardwareCapability": SimpleNamespace(CANN_MEGAMOE_MXFP=object()),
                "supports_mega_moe_shared_experts": probe,
            },
        )["supports_fused_shared_experts"]
        communicator = SimpleNamespace(mega_moe=Mock(), _supports_fused_shared_experts=None)
        context.use_mega_moe = False
        self.assertFalse(method(communicator))
        probe.assert_not_called()
        context.use_mega_moe = True
        hardware.supports.return_value = False
        self.assertFalse(method(communicator))
        probe.assert_not_called()
        hardware.supports.return_value = True
        for supported in (True, False):
            with self.subTest(operator_supported=supported):
                communicator._supports_fused_shared_experts = None
                probe.reset_mock()
                probe.return_value = supported
                self.assertEqual(method(communicator), supported)
                self.assertEqual(method(communicator), supported)
                probe.assert_called_once_with(communicator.mega_moe)

    def setUp(self):
        self.input = TensorDouble((4, 128))
        self.output = TensorDouble((4, 128))
        self.weights = object()
        self.ctx = SimpleNamespace(
            moe_comm_type=Comm.FUSED_MC2,
            moe_comm_method=SimpleNamespace(supports_fused_shared_experts=lambda: True),
        )
        self.runner = SimpleNamespace(
            routed_experts=SimpleNamespace(
                quant_type=Quant.W4A8MXFP,
                dynamic_eplb=False,
                mix_placement=False,
                global_redundant_expert_num=0,
                forward_impl=Mock(return_value=(self.output, SimpleNamespace(shared_experts_fused=True))),
            ),
            ascend_shared_experts=SimpleNamespace(get_mega_moe_weights=lambda: self.weights),
            is_internal_router=False,
        )
        self.method = load_functions(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            ["_forward_with_fused_shared_experts"],
            {
                "_EXTRA_CTX": self.ctx,
                "MoECommType": Comm,
                "QuantType": Quant,
                "torch": SimpleNamespace(zeros_like=lambda value: TensorDouble(value.shape, "zeros")),
            },
        )["_forward_with_fused_shared_experts"]

    def forward(self, shared_input=None):
        return self.method(self.runner, self.input, object(), shared_input or self.input, None)

    def test_sum_is_returned_in_shared_slot_without_double_counting(self):
        shared, routed = self.forward()
        self.assertIs(shared, self.output)
        self.assertEqual(routed.shape, shared.shape)
        self.assertEqual(routed.dtype, "zeros")
        self.assertIs(self.runner.routed_experts.forward_impl.call_args.kwargs["shared_experts"], self.weights)

    def enable_cann_internal_router(self, gate):
        self.runner.is_internal_router = True
        self.runner.gate = gate
        device_gate = Mock(return_value=TensorDouble((4, 8), "float32"))
        functions = load_functions(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            ["_prepare_router_input", "_apply_router_linear", "_compute_router_logits"],
            {
                "torch": SimpleNamespace(float32="float32"),
                "DeviceOperator": SimpleNamespace(compute_gate_logits=device_gate),
            },
        )
        for name in ("_prepare_router_input", "_apply_router_linear", "_compute_router_logits"):
            setattr(self.runner, name, MethodType(functions[name], self.runner))
        return device_gate

    def test_shared_fusion_preserves_cann_gate_with_prepared_fp32_input(self):
        gate = SimpleNamespace(weight=object(), weight_fp32=object())
        device_gate = self.enable_cann_internal_router(gate)
        prepared_fp32 = TensorDouble((4, 128), "float32")
        self.method(self.runner, self.input, prepared_fp32, self.input, None)
        device_gate.assert_called_once_with(self.input, gate.weight, gate.weight_fp32, prepared_input=prepared_fp32)
        self.assertIs(
            self.runner.routed_experts.forward_impl.call_args.kwargs["router_logits"], device_gate.return_value
        )

    def test_shared_fusion_preserves_cann_gate_without_precast_weight(self):
        gate = SimpleNamespace(weight=object())
        device_gate = self.enable_cann_internal_router(gate)
        # TensorDouble has no float() method: this path must let the device
        # adaptor handle the input cast rather than allocate it in the runner.
        self.forward()
        device_gate.assert_called_once_with(self.input, gate.weight, None, prepared_input=self.input)

    def test_shared_fusion_preserves_registered_gate_without_weights(self):
        logits = TensorDouble((4, 8), "float32")
        gate = Mock(return_value=(logits, None))
        # Mock creates arbitrary attributes, so explicitly remove the optional
        # weights to model a registered gate that owns its forward operation.
        del gate.weight
        del gate.weight_fp32
        device_gate = self.enable_cann_internal_router(gate)
        self.forward()
        gate.assert_called_once_with(self.input)
        device_gate.assert_not_called()
        self.assertIs(self.runner.routed_experts.forward_impl.call_args.kwargs["router_logits"], logits)

    def test_transformed_shared_input_uses_standalone_fallback(self):
        self.assertIsNone(self.forward(TensorDouble((4, 128))))
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_legacy_operator_uses_standalone_fallback(self):
        self.ctx.moe_comm_method.supports_fused_shared_experts = lambda: False
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_incompatible_shared_weights_use_standalone_fallback(self):
        self.runner.ascend_shared_experts.get_mega_moe_weights = lambda: None
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_other_routed_quantizations_keep_existing_path(self):
        self.runner.routed_experts.quant_type = Quant.W8A8MXFP
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_dynamic_eplb_keeps_existing_path(self):
        self.runner.routed_experts.dynamic_eplb = True
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_redundant_experts_keep_existing_path(self):
        self.runner.routed_experts.global_redundant_expert_num = 1
        self.assertIsNone(self.forward())
        self.runner.routed_experts.forward_impl.assert_not_called()

    def test_missing_fusion_marker_fails_instead_of_losing_shared_output(self):
        self.runner.routed_experts.forward_impl.return_value = self.output, SimpleNamespace(shared_experts_fused=False)
        with self.assertRaisesRegex(RuntimeError, "without fusing"):
            self.forward()


class TestSharedWeightLifecycle(unittest.TestCase):
    def setUp(self):
        self.shared = SimpleNamespace(
            quant_type=Quant.W4A8MXFP,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=0.0,
            lora_context=False,
            weights_replicated=False,
            _mega_moe_weights=None,
            moe_config=SimpleNamespace(
                hidden_dim=128,
                intermediate_size_per_partition=64,
                tp_group=SimpleNamespace(world_size=1),
                in_dtype="bf16",
                swiglu_limit=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                is_sequence_parallel=False,
            ),
            layer=SimpleNamespace(
                act_fn=type("SiluAndMul", (), {})(),
                gate_up_proj=SimpleNamespace(weight=TensorDouble((64, 256)), weight_scale=TensorDouble((2, 256, 2))),
                down_proj=SimpleNamespace(weight=TensorDouble((32, 128)), weight_scale=TensorDouble((1, 128, 2))),
            ),
        )
        self.functions = load_functions(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            ["__init__", "get_mega_moe_weights"],
            {
                "QuantType": Quant,
                "has_lora": bool,
                "wraps": wraps,
                "torch": SimpleNamespace(uint8="uint8", float8_e4m3fn="fp8"),
                "SituAndMul": type("SituAndMul", (), {}),
                "MoEWeights": SimpleNamespace,
                "get_ascend_config": lambda: SimpleNamespace(
                    multistream_overlap_shared_expert=False, enable_shared_expert_dp=False
                ),
            },
        )

    def test_weight_reload_invalidates_contiguous_shared_cache(self):
        calls = []
        quant_method = SimpleNamespace(process_weights_after_loading=lambda layer: calls.append(layer))
        self.functions["__init__"](self.shared, self.shared.layer, self.shared.moe_config, Quant.W4A8MXFP, quant_method)
        self.shared._mega_moe_weights = object()
        quant_method.process_weights_after_loading(self.shared.layer)
        self.assertIsNone(self.shared._mega_moe_weights)
        self.assertEqual(calls, [self.shared.layer])

    def test_active_lora_cannot_reuse_cached_fused_shared_weights(self):
        self.shared._mega_moe_weights = object()
        self.shared.lora_context = True
        self.assertIsNone(self.functions["get_mega_moe_weights"](self.shared))

    def test_missing_or_none_shared_parameter_uses_standalone_fallback(self):
        for projection in ("gate_up_proj", "down_proj"):
            for parameter in ("weight", "weight_scale"):
                module = getattr(self.shared.layer, projection)
                original = getattr(module, parameter)
                for missing in (False, True):
                    with self.subTest(projection=projection, parameter=parameter, missing=missing):
                        if missing:
                            delattr(module, parameter)
                        else:
                            setattr(module, parameter, None)
                        try:
                            self.assertIsNone(self.functions["get_mega_moe_weights"](self.shared))
                        finally:
                            setattr(module, parameter, original)

    def test_incompatible_shared_layout_falls_back_before_contiguous_copy(self):
        # TensorDouble deliberately has no transpose/contiguous allocation API.
        self.assertIsNone(self.functions["get_mega_moe_weights"](self.shared))

    def test_prepared_shared_weights_match_per_expert_operator_layout(self):
        class LayoutTensor:
            def __init__(self, shape, dtype="uint8"):
                self.shape = shape
                self.dtype = dtype

            @property
            def ndim(self):
                return len(self.shape)

            def transpose(self, first, second):
                shape = list(self.shape)
                shape[first], shape[second] = shape[second], shape[first]
                return LayoutTensor(tuple(shape), self.dtype)

            def contiguous(self):
                return self

            def is_contiguous(self):
                return True

            def view(self, dtype):
                return LayoutTensor(self.shape, dtype)

        validator = TestMegaMoEPort()
        validator.setUp()
        for dtype, pack, weight_type in (("uint8", 2, 296), ("fp8", 1, 24)):
            with self.subTest(dtype=dtype):
                self.shared._mega_moe_weights = None
                self.shared.layer.gate_up_proj = SimpleNamespace(
                    weight=LayoutTensor((128 // pack, 128), dtype),
                    weight_scale=LayoutTensor((2, 128, 2)),
                )
                self.shared.layer.down_proj = SimpleNamespace(
                    weight=LayoutTensor((64 // pack, 128), dtype),
                    weight_scale=LayoutTensor((1, 128, 2)),
                )
                weights = self.functions["get_mega_moe_weights"](self.shared)
                self.assertIsNotNone(weights)
                self.assertEqual(weights.w1[0].shape, (128, 128 // pack))
                self.assertEqual(weights.w2[0].shape, (128, 64 // pack))
                self.assertEqual(weights.w1_scale[0].shape, (128, 2, 2))
                self.assertEqual(weights.w2_scale[0].shape, (128, 1, 2))
                self.assertEqual(self.shared.layer.gate_up_proj.weight.shape, (128 // pack, 128))
                self.assertIs(self.functions["get_mega_moe_weights"](self.shared), weights)
                payload = validator.payload()
                payload.shared_weights = weights
                kwargs = validator.shared["build_mega_moe_shared_kwargs"](payload, 64, 296)
                self.assertEqual(kwargs["shared_weight1_type"], weight_type)
                self.assertEqual(kwargs["shared_weight2_type"], weight_type)
                self.assertEqual(kwargs["shared_l1_weights_sf"][0].dtype, "e8m0")
                self.assertEqual(kwargs["shared_l2_weights_sf"][0].dtype, "e8m0")


class TestMegaMoEModelDimensions(unittest.TestCase):
    def setUp(self):
        utils = load_functions("vllm_ascend/utils.py", ["is_deepseek_v41"], {})
        self.utils = SimpleNamespace(is_deepseek_v41=utils["is_deepseek_v41"])
        self.validate = load_functions(
            "vllm_ascend/ascend_config.py", ["_is_a5_megamoe_supported_by_config"], {"logger": Mock()}
        )["_is_a5_megamoe_supported_by_config"]
        self.text = SimpleNamespace(
            model_type="deepseek_v41_text", hidden_size=5120, moe_intermediate_size=2304, num_experts_per_tok=6
        )
        self.config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41", text_config=self.text),
                hf_text_config=self.text,
                architectures=["DeepseekV41ForCausalLM"],
                get_num_experts=lambda: 384,
            ),
            parallel_config=SimpleNamespace(world_size_across_dp=8, pipeline_parallel_size=1),
        )

    def supports(self):
        # Execute the real config predicate and model detector without loading
        # the NPU runtime imported by the rest of utils.
        with patch.dict(sys.modules, {"vllm_ascend.utils": self.utils}):
            return self.validate(self.config)

    def test_v41_accepts_4608_from_multimodal_and_text_configs(self):
        self.assertTrue(self.supports())
        self.config.model_config.hf_config = self.text
        self.assertTrue(self.supports())
        self.config.model_config.hf_config = {"text_config": {"model_type": "deepseek_v41_text"}}
        self.assertTrue(self.supports())

    def test_other_models_reject_4608_but_preserve_common_dimensions(self):
        for model_type in ("deepseek_v4", "kimi_k3", "minimax_m3", "qwen3_moe"):
            with self.subTest(model_type=model_type):
                self.text.model_type = model_type
                self.config.model_config.hf_config = self.text
                self.text.moe_intermediate_size = 2304
                self.assertFalse(self.supports())
                self.text.moe_intermediate_size = 2048
                self.assertTrue(self.supports())

    def test_kimi_and_minimax_keep_6144_exception(self):
        self.text.model_type = "other"
        self.config.model_config.hf_config = self.text
        self.text.moe_intermediate_size = 3072
        for architecture in ("KimiK3ForCausalLM", "MiniMaxM3SparseForCausalLM"):
            with self.subTest(architecture=architecture):
                self.config.model_config.architectures = [architecture]
                self.assertTrue(self.supports())
        self.config.model_config.architectures = ["OtherForCausalLM"]
        self.assertFalse(self.supports())

    def test_v41_preserves_hidden_size_ep_and_topk_guards(self):
        for field, value in (("hidden_size", 896), ("moe_intermediate_size", 2560), ("num_experts_per_tok", 33)):
            with self.subTest(field=field), patch.object(self.text, field, value):
                self.assertFalse(self.supports())
        for ep_size in (1, 5):
            with (
                self.subTest(ep_size=ep_size),
                patch.object(self.config.parallel_config, "world_size_across_dp", ep_size),
            ):
                self.assertFalse(self.supports())


class TestMegaMoEConfigLifetime(unittest.TestCase):
    def setUp(self):
        path = "vllm_ascend/ops/fused_moe/moe_comm_method.py"
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FusedMC2CommImpl")
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"__init__", "_init_mega_moe_symm_buffer", "_apply_cann_mega_moe"}
        ]

        class Dispatcher:
            global_bs = 0
            max_num_tokens_per_rank = 128
            ep_world_size = 8
            ep_rank_id = 0

        class Base:
            def __init__(self, config):
                self.moe_config = config
                self.token_dispatcher = Dispatcher()

        self.config_getter = Mock(return_value=object())
        self.capacity = Mock(return_value=4)
        self.allocate = Mock(return_value=SimpleNamespace())
        self.op = Mock(return_value=(object(), object()))
        self.ascend_config = SimpleNamespace(enable_fused_mc2=1, mega_moe_max_tokens=65536)
        self.ns = {
            "MoECommMethod": Base,
            "TokenDispatcherWithMC2": Dispatcher,
            "get_ascend_config": lambda: self.ascend_config,
            "is_mega_moe_supported": lambda: True,
            "get_current_vllm_config": self.config_getter,
            "get_a5_mega_moe_buffer_tokens_per_rank": self.capacity,
            "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
            "HardwareCapability": SimpleNamespace(CANN_MEGAMOE_MXFP=object()),
            "get_mc2_group": lambda: SimpleNamespace(device_group=object()),
            "logger": Mock(),
            "cast": lambda _, value: value,
            "torch": SimpleNamespace(zeros=Mock(), int32="int32", float32="float32", nn=SimpleNamespace(Module=object)),
            "moe_utils": SimpleNamespace(
                load_cann_mega_moe_ops=lambda: (self.allocate, self.op),
                select_mega_moe_activation_kwargs=lambda *args, **kwargs: {},
                _get_cann_mega_moe_quant_settings=lambda _: (4, 24, 296),
            ),
        }
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        isolated = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
        exec(compile(isolated, path, "exec"), self.ns)
        config = SimpleNamespace(
            activation="silu",
            swiglu_limit=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            activation_situ_beta=None,
            activation_situ_linear_beta=None,
            num_local_experts=1,
            num_experts=8,
            experts_per_token=2,
            hidden_dim=128,
            intermediate_size_per_partition=64,
        )
        self.comm = self.ns["FusedMC2CommImpl"](config)
        # Model loading has finished; V2 CustomOp/profile execution has no
        # current config. Both the capacity helper and getter are forbidden.
        self.config_getter.side_effect = AssertionError("Current vLLM config is not set")
        self.capacity.side_effect = AssertionError("capacity must be resolved during initialization")

    def apply(self, tokens):
        payload = SimpleNamespace(
            hidden_states=TensorDouble((tokens, 128)),
            topk_ids=Mock(),
            topk_weights=Mock(),
            quant=SimpleNamespace(quant_type=Quant.W4A8MXFP),
            layer=SimpleNamespace(mega_moe_activation_kwargs={}),
            shared_weights=None,
        )
        weights = SimpleNamespace(w1=[], w2=[], w1_scale=[], w2_scale=[], w1_scale_bias=None, w2_scale_bias=None)
        return self.comm._apply_cann_mega_moe(payload, weights, is_decode_only_node=False)

    def test_first_forward_allocates_buffer_without_current_config(self):
        result = self.apply(4)
        self.assertEqual(result, self.op.return_value)
        self.assertEqual(self.allocate.call_args.args[2], 4)
        self.assertEqual(self.config_getter.call_count, 1)
        self.assertEqual(self.capacity.call_count, 1)

    def test_smaller_mc2_capacity_does_not_shrink_a5_megamoe_buffer(self):
        self.comm.token_dispatcher.max_num_tokens_per_rank = 2
        # The actual input exceeds ordinary dispatch's capacity, but fits
        # the independent MegaMoE capacity cached during initialization.
        result = self.apply(4)
        self.assertEqual(result, self.op.return_value)
        self.assertEqual(self.allocate.call_args.args[2], 4)
        self.op.assert_called_once()

    def test_later_forward_reuses_buffer_without_current_config(self):
        self.apply(4)
        self.apply(2)
        self.allocate.assert_called_once()
        self.assertEqual(self.op.call_count, 2)
        self.assertEqual(self.config_getter.call_count, 1)

    def test_overflow_rejected_before_allocation_or_operator(self):
        with self.assertRaisesRegex(ValueError, "received 5 tokens"):
            self.apply(5)
        self.allocate.assert_not_called()
        self.op.assert_not_called()


if __name__ == "__main__":
    unittest.main()
