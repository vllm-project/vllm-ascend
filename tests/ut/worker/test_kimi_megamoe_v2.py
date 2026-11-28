"""Exercise production V2 phase, dispatch and platform code without NPU imports.

Run with ``python3 tests/ut/worker/test_kimi_megamoe_v2.py``. AST loading
isolates the real functions from package initialization; CPU tensor doubles
replace storage and collectives, and never substitute the policy functions.
"""

import ast
import importlib.util
import math
import sys
import unittest
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[3]


class GraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class Policy(Enum):
    CAPACITY_AND_WORLD_SIZE = 1
    ALLGATHER = 2


class CpuTensor:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return CpuTensor(self.values[index]) if isinstance(index, slice) else self.values[index]

    def __setitem__(self, index, value):
        if isinstance(value, CpuTensor):
            value = value.values
        if isinstance(index, slice) and not isinstance(value, list):
            value = [value] * len(self.values[index])
        self.values[index] = value

    def tolist(self):
        return self.values

    def max(self):
        return NS(item=lambda: max(self.values))


@dataclass(frozen=True)
class BatchDesc:
    num_tokens: int
    num_reqs: int
    cg_mode: GraphMode


@dataclass(frozen=True)
class DPSync:
    num_tokens_across_dp: CpuTensor
    eager: bool


def load_functions(relative_path, names, namespace, *, class_name=None):
    tree = ast.parse((ROOT / relative_path).read_text())
    candidates = tree.body
    if class_name is not None:
        candidates = next(node.body for node in tree.body if getattr(node, "name", None) == class_name)
    nodes = [node for node in candidates if getattr(node, "name", None) in names]
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            node.body = [
                statement for statement in node.body if not isinstance(statement, (ast.Import, ast.ImportFrom))
            ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, relative_path, "exec"), namespace)


def load_state_module():
    name = "_kimi_megamoe_v2_state_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_ascend/worker/v2/moe_utils.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestKimiMegaMoEV2(unittest.TestCase):
    def setUp(self):
        self.state = load_state_module()
        self.capacity = 64
        self.ascend = NS(enable_fused_mc2=1, mix_placement=False, eplb_config=NS())
        self.torch = NS(
            int32="int32",
            zeros=lambda shape, **kwargs: CpuTensor([[0] * shape[1] for _ in range(shape[0])]),
            tensor=lambda values, **kwargs: CpuTensor(values),
            full_like=lambda tensor, fill: CpuTensor([fill] * len(tensor)),
        )
        self.collective = Mock()
        self.ns = dict(
            Enum=Enum,
            torch=self.torch,
            dist=NS(all_reduce=self.collective),
            get_dp_group=lambda: NS(cpu_group="dp-cpu", world_size=2),
            get_tensor_model_parallel_world_size=lambda: 1,
            get_ep_group=lambda: NS(world_size=2),
            get_mc2_tokens_capacity=lambda: self.capacity,
            get_ascend_config=lambda: self.ascend,
            get_current_hardware_profile=lambda: NS(
                supports=lambda cap: True, moe_comm_policy=Policy.CAPACITY_AND_WORLD_SIZE
            ),
            HardwareCapability=NS(CANN_MEGAMOE_MXFP="mxfp", CANN_MEGAMOE="mega"),
            is_mega_moe_supported=lambda: True,
            is_moe_model=lambda config: True,
            MoECommPolicy=Policy,
            CUDAGraphMode=GraphMode,
            logger=Mock(),
            replace=replace,
            is_pcp_decode_sharding_enabled=lambda config: False,
            math=math,
            get_dynamic_mx_quant_scale_alg=lambda config: None,
            get_mrv2_in_profile_run=lambda: False,
            _is_decode_only_node=lambda config: False,
            get_mc2_mask=lambda: CpuTensor([False] * 64),
            get_moe_comm_method=lambda comm: comm,
        )
        self.ns.update(
            {
                name: getattr(self.state, name)
                for name in (
                    "V2MoEPhaseMetadata",
                    "get_v2_moe_batch_state",
                    "agree_v2_moe_batch",
                    "record_v2_moe_batch_phase",
                    "v2_moe_batch_context",
                )
            }
        )
        load_functions("vllm_ascend/quantization/quant_type.py", {"QuantType"}, self.ns)
        self.ns["_KIMI_MEGA_MOE_GROUP_SIZE"] = 32
        self.ns["_MC2_TOKENS_PER_RANK_LIMIT"] = 512
        self.ns["A5_SUPPORT_MEGA_MOE_QUANT_TYPES"] = {self.ns["QuantType"].W4A8MXFP}
        load_functions(
            "vllm_ascend/ascend_forward_context.py",
            {
                "MoECommType",
                "_is_kimi_moe_model",
                "_is_kv_consumer_only",
                "register_kimi_moe_quant_type",
                "log_kimi_megamoe_config",
                "uses_kimi_prefill_megamoe",
                "_kimi_megamoe_weights_supported",
                "use_cann_megamoe",
                "select_moe_comm_method",
                "_select_capacity_and_world_size_moe_comm_method",
            },
            self.ns,
        )
        self.ns["_MOE_COMM_SELECTORS"] = {
            Policy.CAPACITY_AND_WORLD_SIZE: self.ns["_select_capacity_and_world_size_moe_comm_method"]
        }
        load_functions(
            "vllm_ascend/patch/worker/patch_v2/patch_model_runner.py",
            {"_sync_v2_moe_phase", "dispatch_cg_and_sync_dp"},
            self.ns,
        )
        load_functions("vllm_ascend/platform.py", {"set_additional_forward_context"}, self.ns, class_name="NPUPlatform")
        self.ns["_dispatch_cg_and_sync_dp"] = Mock(side_effect=self.parent_dispatch)
        self.config = NS(
            use_v2_model_runner=True,
            model_config=NS(
                architectures=["KimiK3ForCausalLM"],
                hf_text_config=NS(hidden_act="situ", num_experts_per_tok=1),
                _ascend_kimi_moe_quant_types=frozenset((self.ns["QuantType"].W4A8MXFP,)),
            ),
            parallel_config=NS(
                enable_expert_parallel=True,
                world_size_across_dp=2,
                tensor_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            scheduler_config=NS(max_num_batched_tokens=64),
            quant_config=NS(quant_description={"group_size": 32}),
            kv_transfer_config=None,
            lora_config=None,
        )

    def parent_dispatch(self, manager, num_reqs, num_tokens, uniform, dp_size, dp_rank, **kwargs):
        mode = GraphMode.NONE if kwargs["need_eager"] else GraphMode.FULL
        sync = DPSync(CpuTensor([num_tokens] * dp_size), mode == GraphMode.NONE) if dp_size > 1 else None
        return BatchDesc(num_tokens, num_reqs, mode), sync

    def record(self, flags, num_tokens):
        self.state.record_v2_moe_batch_phase(NS(num_tokens=num_tokens, is_prefilling_np=NS(all=lambda: all(flags))))

    def forward_context(self, num_tokens, sync=None, mode=GraphMode.NONE, config=None):
        dp_metadata = None if sync is None else NS(num_tokens_across_dp_cpu=sync.num_tokens_across_dp)
        return self.ns["set_additional_forward_context"](
            None, None, config or self.config, dp_metadata, num_tokens=num_tokens, cudagraph_runtime_mode=mode
        )

    def run_batch(self, phases, tokens, expected, *, tp_size=1):
        self.config.parallel_config.tensor_parallel_size = tp_size
        self.ns["get_tensor_model_parallel_world_size"] = lambda: tp_size
        metadata = [
            [count, phase != "I", phase in ("D", "M", "I"), False, True, self.capacity]
            for phase, count in zip(phases, tokens)
        ]
        contexts = []
        for rank, (phase, count) in enumerate(zip(phases, tokens)):
            with self.state.v2_moe_batch_context(self.config, count, dummy_run=phase == "I", is_profile=False):
                if phase != "I":
                    self.record([True, False] if phase == "M" else [phase == "P"], count)

                def all_reduce(tensor, group, rank=rank):
                    self.assertEqual(group, "dp-cpu")
                    self.assertEqual(tensor.values[rank], metadata[rank])
                    tensor.values = metadata

                self.collective.side_effect = all_reduce
                desc, sync = self.ns["dispatch_cg_and_sync_dp"](None, 1, count, 1, len(phases), rank)
                context = self.forward_context(desc.num_tokens, sync, desc.cg_mode)
                self.assertEqual(context["moe_comm_type"].name, expected)
                if expected == "FUSED_MC2":
                    padded = math.ceil(max(tokens) / tp_size) * tp_size
                    self.assertEqual(desc.cg_mode, GraphMode.NONE)
                    self.assertEqual(desc.num_tokens, padded)
                    self.assertEqual(sync.num_tokens_across_dp.tolist(), [padded] * len(phases))
                    actual = 0 if phase == "I" else count
                    self.assertEqual(context["mc2_mask"].tolist(), [True] * actual + [False] * (padded - actual))
                contexts.append(context)
            self.assertIsNone(self.state.get_v2_moe_batch_state())
        self.assertEqual(self.collective.call_count, len(phases))
        return contexts

    def test_uneven_prefill_pads_all_dp_ranks(self):
        self.run_batch(["P", "P"], [1, 9], "FUSED_MC2")

    def test_prefill_with_idle_peer(self):
        self.run_batch(["P", "I"], [13, 1], "FUSED_MC2")

    def test_tp_alignment_preserves_valid_rows(self):
        self.run_batch(["P", "P"], [3, 9], "FUSED_MC2", tp_size=4)

    def test_remote_decode_and_local_mixed_batches_fall_back(self):
        for phases in (["P", "D"], ["M", "P"], ["D", "I"], ["I", "I"]):
            with self.subTest(phases=phases):
                self.collective.reset_mock()
                self.run_batch(phases, [8, 1], "MC2")

    def test_unknown_phase_and_graph_contexts_cannot_enable_megamoe(self):
        for mode in GraphMode:
            self.assertEqual(self.forward_context(8, mode=mode)["moe_comm_type"].name, "MC2")
        self.collective.assert_not_called()
        self.assertEqual(self.ns["select_moe_comm_method"](8, self.config, v2_prefill_ready=True).name, "MC2")

    def test_existing_pcp_dispatch_and_extra_kwargs_are_preserved(self):
        self.ns["is_pcp_decode_sharding_enabled"] = lambda config: True
        manager = NS(vllm_config=self.config)
        self.ns["dispatch_cg_and_sync_dp"](manager, 8, 6, 2, 1, 0, additional_option=True)
        call = self.ns["_dispatch_cg_and_sync_dp"].call_args
        self.assertEqual(call.args[1], 3)
        self.assertTrue(call.kwargs["additional_option"])
        self.collective.assert_not_called()

    def test_v1_and_other_models_keep_the_existing_policy(self):
        self.config.use_v2_model_runner = False
        select = self.ns["select_moe_comm_method"]
        self.assertEqual(select(8, self.config, batch_contains_decode=False).name, "FUSED_MC2")
        self.assertEqual(select(8, self.config, batch_contains_decode=True).name, "MC2")
        self.config.use_v2_model_runner = True
        self.config.model_config.architectures = ["OtherMoE"]
        self.assertEqual(
            select(8, self.config, batch_contains_decode=True, aclgraph_runtime_mode=GraphMode.FULL).name,
            "FUSED_MC2",
        )

    def test_capture_is_rejected_even_with_prefill_agreement(self):
        with self.state.v2_moe_batch_context(self.config, 8, dummy_run=False, is_profile=False):
            self.record([True], 8)
            self.ns["dispatch_cg_and_sync_dp"](None, 1, 8, None, 1, 0)
            for mode in (GraphMode.FULL, GraphMode.PIECEWISE):
                self.assertEqual(self.forward_context(8, mode=mode)["moe_comm_type"].name, "MC2")

    def test_single_dp_rank_prefill_uses_no_cpu_collective(self):
        with self.state.v2_moe_batch_context(self.config, 7, dummy_run=False, is_profile=False):
            self.record([True], 7)
            desc, sync = self.ns["dispatch_cg_and_sync_dp"](None, 1, 7, None, 1, 0)
            self.assertIsNone(sync)
            self.assertEqual(self.forward_context(desc.num_tokens)["moe_comm_type"].name, "FUSED_MC2")
        self.collective.assert_not_called()

    def test_v2_eplb_redundant_experts_allow_prefill_but_not_decode(self):
        self.config.parallel_config.enable_eplb = True
        self.config.parallel_config.eplb_config = NS(num_redundant_experts=8)
        self.config.parallel_config.tensor_parallel_size = 8
        self.config.parallel_config.world_size_across_dp = 8
        self.config.model_config.hf_text_config.n_routed_experts = 224
        self.config.kv_transfer_config = NS(kv_role="kv_producer")
        self.ns["get_ep_group"] = lambda: NS(world_size=8)
        self.ns["get_tensor_model_parallel_world_size"] = lambda: 8
        self.assertTrue(self.ns["_kimi_megamoe_weights_supported"](self.config))
        for prefill, expected in ((True, "FUSED_MC2"), (False, "MC2"), (True, "FUSED_MC2")):
            with self.state.v2_moe_batch_context(self.config, 8, dummy_run=False, is_profile=False):
                self.record([prefill], 8)
                desc, sync = self.ns["dispatch_cg_and_sync_dp"](None, 1, 8, None, 1, 0)
                self.assertIsNone(sync)
                self.assertEqual(self.forward_context(desc.num_tokens)["moe_comm_type"].name, expected)
        self.collective.assert_not_called()

    def test_v2_eplb_does_not_relax_legacy_placement_guards(self):
        self.config.parallel_config.eplb_config = NS(num_redundant_experts=8)
        for v2, enabled, dynamic, ascend_redundancy, mix in (
            (False, True, False, 0, False),
            (True, False, False, 0, False),
            (True, True, True, 0, False),
            (True, True, False, 8, False),
            (True, True, False, 0, True),
        ):
            with self.subTest(v2=v2, enabled=enabled, dynamic=dynamic, redundancy=ascend_redundancy, mix=mix):
                self.config.use_v2_model_runner = v2
                self.config.parallel_config.enable_eplb = enabled
                self.ascend.eplb_config = NS(dynamic_eplb=dynamic, num_redundant_experts=ascend_redundancy)
                self.ascend.mix_placement = mix
                self.assertFalse(self.ns["_kimi_megamoe_weights_supported"](self.config))

    def test_profile_is_explicit_and_counts_dummy_rows(self):
        with self.state.v2_moe_batch_context(self.config, 16, dummy_run=True, is_profile=True):
            self.ns["dispatch_cg_and_sync_dp"](None, 1, 16, None, 1, 0, need_eager=True)
            context = self.forward_context(16)
            self.assertEqual(context["moe_comm_type"].name, "FUSED_MC2")
            self.assertEqual(context["mc2_mask"].tolist(), [True] * 16)

    def test_prefill_clears_only_the_new_physical_padding(self):
        buffers = NS(input_ids=MagicMock(), positions=MagicMock())
        with self.state.v2_moe_batch_context(self.config, 3, dummy_run=False, is_profile=False):
            self.state.clear_v2_moe_padding(buffers, 3, 8)
            buffers.input_ids.__getitem__.assert_not_called()
            self.record([True], 3)
            self.state.agree_v2_moe_batch([[3, 1, 0, 0, 1, 64], [8, 1, 0, 0, 1, 64]], 1)
            self.state.clear_v2_moe_padding(buffers, 3, 8)
            for buffer in (buffers.input_ids, buffers.positions):
                buffer.__getitem__.assert_called_once_with(slice(3, 8))
                buffer.__getitem__.return_value.zero_.assert_called_once_with()

    def test_capacity_alignment_and_unsupported_configs_fall_back(self):
        for field, value in (
            ("capacity", 7),
            ("pcp", 2),
            ("group_size", 64),
            ("consumer", True),
            ("lora", object()),
            ("allow_megamoe", False),
            ("ubatching", True),
        ):
            with self.subTest(field=field):
                self.setUp()
                if field == "capacity":
                    self.capacity = value
                elif field == "pcp":
                    self.config.parallel_config.prefill_context_parallel_size = value
                elif field == "group_size":
                    self.config.quant_config.quant_description["group_size"] = value
                elif field == "consumer":
                    self.config.kv_transfer_config = NS(is_kv_consumer=True)
                elif field == "lora":
                    self.config.lora_config = value
                with self.state.v2_moe_batch_context(
                    self.config, 8, dummy_run=False, is_profile=False, allow_megamoe=field != "allow_megamoe"
                ):
                    self.record([True], 8)
                    self.ns["dispatch_cg_and_sync_dp"](None, 1, 8, None, 1, 0, allow_ubatching=field == "ubatching")
                    expected = "ALLTOALL" if field in ("capacity", "lora") else "MC2"
                    self.assertEqual(self.forward_context(8)["moe_comm_type"].name, expected)

    def test_peer_without_weight_support_blocks_all_ranks(self):
        with self.state.v2_moe_batch_context(self.config, 8, dummy_run=False, is_profile=False):
            self.record([True], 8)
            result = self.state.agree_v2_moe_batch([[8, 1, 0, 0, 1, 64], [8, 1, 0, 0, 0, 64]], 1)
            self.assertFalse(result.prefill_ready)

    def test_draft_configuration_does_not_inherit_target_phase(self):
        with self.state.v2_moe_batch_context(self.config, 8, dummy_run=False, is_profile=False):
            self.record([True], 8)
            self.ns["dispatch_cg_and_sync_dp"](None, 1, 8, None, 1, 0)
            draft_config = NS(**vars(self.config))
            self.assertEqual(self.forward_context(8, config=draft_config)["moe_comm_type"].name, "MC2")

    def test_nested_contexts_restore_after_exception(self):
        with self.state.v2_moe_batch_context(self.config, 8, dummy_run=False, is_profile=False):
            self.record([True], 8)
            outer = self.state.get_v2_moe_batch_state()
            with (
                self.assertRaisesRegex(RuntimeError, "forward failed"),
                self.state.v2_moe_batch_context(self.config, 1, dummy_run=True, is_profile=False),
            ):
                raise RuntimeError("forward failed")
            self.assertIs(self.state.get_v2_moe_batch_state(), outer)
        self.assertIsNone(self.state.get_v2_moe_batch_state())

    def make_runner(self):
        parent = type(
            "ParentRunner",
            (),
            {
                "gather_batch_req_state": Mock(),
                "execute_model": Mock(),
                "profile_run": Mock(),
            },
        )
        tree = ast.parse((ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text())
        runner_class = next(node for node in tree.body if getattr(node, "name", None) == "NPUModelRunner")
        runner_class.bases = [ast.Name(id="ParentRunner", ctx=ast.Load())]
        runner_class.body = [
            node
            for node in runner_class.body
            if getattr(node, "name", None) in ("gather_batch_req_state", "execute_model", "profile_run")
        ]
        for node in runner_class.body:
            node.decorator_list = []
        self.ns["ParentRunner"] = parent
        self.ns.update(
            nullcontext=nullcontext,
            has_kv_transfer_group=lambda: False,
            should_skip_allreduce_across_dp_group=lambda config: False,
            _start_profiling_chunk_timing=lambda *args: None,
            _finish_profiling_chunk_timing=lambda *args: None,
            lmhead_tp_enable=lambda: False,
            IntermediateTensors=type("IntermediateTensors", (), {}),
            override_mrv2_in_profile_run=lambda enabled: nullcontext(),
            disable_compilation=lambda model: nullcontext(),
        )
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = ast.fix_missing_locations(ast.Module(body=[future, runner_class], type_ignores=[]))
        exec(compile(module, "runner", "exec"), self.ns)
        runner_type = self.ns["NPUModelRunner"]
        runner = runner_type()
        runner.vllm_config = self.config
        runner._uses_flashmla = False
        runner._finegrained_tp_requires_graph = False
        runner.ascend_config = NS(
            scheduler_config=NS(profiling_chunk_config=None), sparse_kv_offload_config=NS(enabled=False)
        )
        runner.model_state = NS(kvpp_is_dummy_run=False, finish_execution=Mock())
        runner.input_buffers = NS(profile_cpp=False)
        runner.kvpp = Mock()
        runner._start_dump_data = Mock()
        runner._finalize_dump_data = Mock()
        runner._maybe_send_draft_kv = Mock()
        runner.is_last_pp_rank = False
        runner.is_pooling_model = False
        runner.adaptive_verification = None
        return runner, parent

    def test_runner_records_phase_after_request_state_adjustments(self):
        runner, parent = self.make_runner()
        for flags in ([True], [False], [True, False]):
            batch = NS(num_tokens=1, is_prefilling_np=NS(all=lambda flags=flags: all(flags)))
            parent.gather_batch_req_state.return_value = batch, None
            with self.state.v2_moe_batch_context(self.config, 1, dummy_run=False, is_profile=False):
                runner.gather_batch_req_state(NS(), False)
                self.assertEqual(self.state.get_v2_moe_batch_state().batch_contains_decode, not all(flags))

    def test_execute_scopes_prefill_then_decode_without_leaking_to_drafter(self):
        runner, parent = self.make_runner()
        scheduled = NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1})
        methods = []
        for prefill in (True, False):
            parent.gather_batch_req_state.return_value = (
                NS(num_tokens=1, is_prefilling_np=NS(all=lambda prefill=prefill: prefill)),
                None,
            )

            def execute(*args, **kwargs):
                runner.gather_batch_req_state(scheduled, False)
                desc, sync = self.ns["dispatch_cg_and_sync_dp"](None, 1, 1, 1, 1, 0)
                methods.append(self.forward_context(desc.num_tokens, sync, desc.cg_mode)["moe_comm_type"].name)

            parent.execute_model.side_effect = execute
            runner.execute_model(scheduled)
            self.assertIsNone(self.state.get_v2_moe_batch_state())
        self.assertEqual(methods, ["FUSED_MC2", "MC2"])
        runner.model_state.finish_execution.assert_called_with(failed=False)

    def test_execute_failure_restores_context(self):
        runner, parent = self.make_runner()
        parent.execute_model.side_effect = RuntimeError("model failed")
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            runner.execute_model(NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}))
        self.assertIsNone(self.state.get_v2_moe_batch_state())
        runner.model_state.finish_execution.assert_called_once_with(failed=True)

    def test_multimodal_producer_uses_loaded_language_model_quantization(self):
        quant_type = self.ns["QuantType"]
        for layouts, expected in (
            ((quant_type.W4A8MXFP,), "FUSED_MC2"),
            ((quant_type.W4A8MXFP, quant_type.NONE), "MC2"),
            ((quant_type.NONE,), "MC2"),
        ):
            with self.subTest(layouts=layouts):
                self.config.model_config.architectures = ["KimiK3ForConditionalGeneration"]
                if hasattr(self.config.model_config, "_ascend_kimi_moe_quant_types"):
                    del self.config.model_config._ascend_kimi_moe_quant_types
                self.config.kv_transfer_config = NS(kv_role="kv_producer")
                self.config.model_config.multimodal_config = NS()
                self.config.model_config.hf_text_config.hidden_size = 7168
                self.config.model_config.hf_config = NS(
                    text_config=self.config.model_config.hf_text_config,
                    vision_config=NS(num_attention_heads=1),
                    media_placeholder_token_id=0,
                )

                def init_language_model(vllm_config, hf_config, prefix, architectures, loaded_layouts=layouts):
                    language_config = copy(vllm_config)
                    language_config.model_config = deepcopy(vllm_config.model_config)
                    language_config.model_config.architectures = architectures
                    for layout in loaded_layouts:
                        self.ns["register_kimi_moe_quant_type"](language_config, layout)
                    return NS(model_config=language_config.model_config, make_empty_intermediate_tensors=Mock())

                vision_module = Mock()
                vision_module.to.return_value = vision_module
                namespace = dict(
                    self.ns,
                    nn=NS(Module=object),
                    current_platform=NS(current_device=lambda: "npu"),
                    is_vit_use_data_parallel=lambda heads: True,
                    MoonViT3dPretrainedModel=Mock(return_value=vision_module),
                    AscendKimiK3MultiModalProjector=Mock(return_value=vision_module),
                    get_rotation_path=lambda config: None,
                    maybe_prefix=lambda prefix, suffix: prefix + "." + suffix,
                    init_vllm_registered_model=init_language_model,
                )
                load_functions(
                    "vllm_ascend/models/kimi_k3.py",
                    {"__init__"},
                    namespace,
                    class_name="AscendKimiK3ForConditionalGeneration",
                )
                wrapper = type(
                    "MultimodalKimi",
                    (),
                    {
                        "__init__": namespace["__init__"],
                        "_maybe_ignore_quant_config": lambda self, config: config,
                        "_mark_tower_model": lambda *args: nullcontext(),
                        "_mark_language_model": lambda *args: nullcontext(),
                    },
                )(self.config)
                self.assertIsNot(wrapper.language_model.model_config, self.config.model_config)
                self.assertEqual(self.config.model_config._ascend_kimi_moe_quant_types, frozenset(layouts))
                with self.state.v2_moe_batch_context(self.config, 8, dummy_run=False, is_profile=False):
                    self.record([True], 8)
                    desc, sync = self.ns["dispatch_cg_and_sync_dp"](None, 1, 8, None, 1, 0)
                    context = self.forward_context(desc.num_tokens, sync, desc.cg_mode)
                    self.assertEqual(context["moe_comm_type"].name, expected)

    def test_kv_consumer_does_not_enter_the_new_phase_collective(self):
        runner, parent = self.make_runner()
        self.config.kv_transfer_config = NS(is_kv_consumer=True)

        def execute(*args, **kwargs):
            self.assertIsNone(self.state.get_v2_moe_batch_state())
            self.ns["dispatch_cg_and_sync_dp"](None, 1, 1, 1, 2, 0)

        parent.execute_model.side_effect = execute
        runner.execute_model(NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1}))
        self.collective.assert_not_called()

    def test_kv_both_prefill_is_not_treated_as_a_decode_only_instance(self):
        runner, parent = self.make_runner()
        self.config.kv_transfer_config = NS(is_kv_consumer=True, is_kv_producer=True, kv_role="kv_both")
        scheduled = NS(total_num_scheduled_tokens=1, num_scheduled_tokens={"request": 1})
        methods = []
        for prefill in (True, False):
            parent.gather_batch_req_state.return_value = (
                NS(num_tokens=1, is_prefilling_np=NS(all=lambda prefill=prefill: prefill)),
                None,
            )

            def execute(*args, **kwargs):
                runner.gather_batch_req_state(scheduled, False)
                desc, sync = self.ns["dispatch_cg_and_sync_dp"](None, 1, 1, 1, 1, 0)
                methods.append(self.forward_context(desc.num_tokens, sync, desc.cg_mode)["moe_comm_type"].name)

            parent.execute_model.side_effect = execute
            runner.execute_model(scheduled)
        self.assertEqual(methods, ["FUSED_MC2", "MC2"])

    def test_kv_role_classification(self):
        for role in ("kv_producer", "kv_consumer", "kv_both"):
            with self.subTest(role=role):
                self.config.kv_transfer_config = NS(kv_role=role)
                self.assertEqual(self.ns["_is_kv_consumer_only"](self.config), role == "kv_consumer")
        self.config.kv_transfer_config = NS(is_kv_consumer=True, is_kv_producer=True)
        self.assertFalse(self.ns["_is_kv_consumer_only"](self.config))

    def test_loaded_config_diagnostic_reports_actual_quantization_and_runtime(self):
        for layouts, available, supported in (
            (frozenset((self.ns["QuantType"].W4A8MXFP,)), True, True),
            (frozenset((self.ns["QuantType"].W4A8,)), True, False),
            (None, True, False),
            (frozenset((self.ns["QuantType"].W4A8MXFP,)), False, True),
        ):
            with self.subTest(layouts=layouts, available=available):
                self.config.model_config._ascend_kimi_moe_quant_types = layouts
                self.ns["is_mega_moe_supported"] = lambda available=available: available
                self.ns["log_kimi_megamoe_config"](self.config)
                message, *arguments = self.ns["logger"].info_once.call_args.args
                output = message % tuple(arguments)
                self.assertIn(f"runtime_available={available}", output)
                self.assertIn(f"weights_supported={supported}", output)
                self.assertIn(f"capable={available and (layouts is None or supported)}", output)
                self.assertIn(f"layouts={sorted(layout.name for layout in layouts or ())}", output)

    def test_phase_diagnostic_reports_eligibility_and_padding_rejection(self):
        for capacity, allow_megamoe, reason in ((64, False, "ineligible_rank"), (7, True, "token_capacity_or_padding")):
            with self.subTest(reason=reason):
                self.capacity = capacity
                with self.state.v2_moe_batch_context(
                    self.config, 8, dummy_run=False, is_profile=False, allow_megamoe=allow_megamoe
                ):
                    self.record([True], 8)
                    state = self.ns["_sync_v2_moe_phase"](8, 1, 0, False)
                    self.assertFalse(state.prefill_ready)
                    self.ns["logger"].info_once.assert_called_with("Kimi V2 MegaMoE phase agreement: %s", reason)
                    details = self.ns["logger"].debug.call_args.args
                    self.assertEqual(details[-1], reason)
                    self.assertEqual(details[-2], 8)

    def test_cpp_dummy_profiles_the_prefill_backend(self):
        runner, parent = self.make_runner()
        runner.input_buffers.profile_cpp = True
        contexts = []

        def execute(*args, **kwargs):
            desc, sync = self.ns["dispatch_cg_and_sync_dp"](None, 1, 8, None, 1, 0)
            contexts.append(self.forward_context(desc.num_tokens, sync, desc.cg_mode))

        parent.execute_model.side_effect = execute
        runner.execute_model(NS(total_num_scheduled_tokens=8), dummy_run=True)
        self.assertEqual(contexts[0]["moe_comm_type"].name, "FUSED_MC2")
        self.assertEqual(contexts[0]["mc2_mask"].tolist(), [True] * 8)
        self.assertIsNone(self.state.get_v2_moe_batch_state())

    def test_profile_reserves_megamoe_capacity_before_parent_profile(self):
        runner, parent = self.make_runner()
        runner.max_num_tokens = 128
        runner.model = object()
        runner.get_model = Mock()
        runner._dummy_run = Mock()
        select = Mock(wraps=self.ns["select_moe_comm_method"])
        self.ns["select_moe_comm_method"] = select
        release_weights = Mock(return_value=1024)
        with patch.dict(sys.modules, {"vllm_ascend.ops.kimi_kda": NS(release_mega_kda_nd_weights=release_weights)}):
            runner.profile_run()
        select.assert_called_once_with(64, self.config, batch_contains_decode=False, v2_prefill_ready=True)
        runner._dummy_run.assert_called_once_with(64, skip_attn=True, skip_eplb=True, is_profile=True)
        parent.profile_run.assert_called_once_with()
        release_weights.assert_called_once_with(runner.model)
        self.assertEqual(runner.mega_kda_nd_released_bytes, 1024)


if __name__ == "__main__":
    unittest.main()
