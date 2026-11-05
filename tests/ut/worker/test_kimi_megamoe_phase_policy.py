# SPDX-License-Identifier: Apache-2.0
"""CPU policy tests using production runner/selector methods and mocked collectives.

Run directly with CPU PyTorch. HCCL and graph replay require NPU integration tests.
"""

import ast
import logging
import unittest
from enum import Enum, IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]


class GraphMode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class Policy(Enum):
    CAPACITY_AND_EXPERT_DENSITY = 1
    FUSED_OR_CAPACITY = 2
    CAPACITY_AND_WORLD_SIZE = 3
    ALLGATHER = 4


def load_policy():
    context = ast.parse((ROOT / "vllm_ascend/ascend_forward_context.py").read_text())
    runner = ast.parse((ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text())
    quant = ast.parse((ROOT / "vllm_ascend/quantization/quant_type.py").read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [n for n in quant.body if isinstance(n, ast.ClassDef)]
    names = {
        "MoECommType",
        "uses_kimi_prefill_megamoe",
        "select_moe_comm_method",
        "_select_capacity_and_world_size_moe_comm_method",
    }
    nodes += [n for n in context.body if getattr(n, "name", None) in names]
    cls = next(n for n in runner.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
    cls.bases = []
    cls.decorator_list = []
    cls.body = [
        n
        for n in cls.body
        if getattr(n, "name", None) in {"_sync_metadata_across_dp", "_determine_batch_execution_and_padding"}
    ]
    nodes += [cls]
    nodes += [
        n
        for n in runner.body
        if getattr(n, "name", None) == "_post_process_cudagraph_mode"
        or isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.startswith("DP_") for t in n.targets)
    ]
    ns = dict(
        torch=torch,
        Enum=Enum,
        CUDAGraphMode=GraphMode,
        MoECommPolicy=Policy,
        logging=logging,
        logger=Mock(),
        HardwareCapability=NS(CANN_MEGAMOE_MXFP="a5"),
        is_mega_moe_supported=lambda: True,
        is_moe_model=lambda c: True,
        get_ep_group=lambda: NS(world_size=32),
        get_mc2_tokens_capacity=lambda: 64,
        should_skip_allreduce_across_dp_group=lambda c, draft: draft,
        use_cann_megamoe=lambda c: True,
        enable_sp=lambda *a: False,
        is_pd_decode_recompute_scheduler_enabled=lambda c: False,
        np=NS(any=lambda x: bool(x.any()), all=lambda x: bool(x.all())),
        BatchDescriptor=lambda n: NS(num_tokens=n, num_reqs=None),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "policy", "exec"), ns)
    ns["_MOE_COMM_SELECTORS"] = {Policy.CAPACITY_AND_WORLD_SIZE: ns["_select_capacity_and_world_size_moe_comm_method"]}
    ns["A5_SUPPORT_MEGA_MOE_QUANT_TYPES"] = {ns["QuantType"].W4A8MXFP}
    return ns


class TestKimiMegaMoEPhase(unittest.TestCase):
    def setUp(self):
        self.ns = load_policy()
        self.ascend = NS(
            enable_fused_mc2=1,
            finegrained_tp_config=NS(
                oproj_tensor_parallel_size=1,
                lmhead_tensor_parallel_size=1,
                embedding_tensor_parallel_size=1,
                mlp_tensor_parallel_size=1,
            ),
        )
        self.config = NS(
            model_config=NS(
                architectures=["KimiK3ForConditionalGeneration"], hf_text_config=NS(num_experts_per_tok=16)
            ),
            parallel_config=NS(enable_expert_parallel=True, world_size_across_dp=32),
            lora_config=None,
            kv_transfer_config=None,
            use_v2_model_runner=False,
        )
        self.ns["get_ascend_config"] = lambda: self.ascend
        self.ns["get_current_hardware_profile"] = lambda: NS(
            supports=lambda cap: True, moe_comm_policy=Policy.CAPACITY_AND_WORLD_SIZE
        )

    def batch(self, phases, tokens, expected, unavailable=()):
        # P=real prefill, D=real decode, M=local mixture, I=idle dummy.
        modes = [
            GraphMode.NONE if p in ("P", "M") or r in unavailable else GraphMode.FULL for r, p in enumerate(phases)
        ]
        packed = torch.tensor(
            [tokens, modes, [p != "I" for p in phases], [p in ("D", "M", "I") for p in phases]], dtype=torch.int32
        )
        outputs = []
        for rank, phase in enumerate(phases):
            r = self.ns["NPUModelRunner"]()
            r.dp_size = len(phases)
            r.dp_rank = rank
            r.ascend_config = self.ascend
            r.vllm_config = NS(**vars(self.config))
            r.vllm_config.parallel_config = NS(
                **vars(self.config.parallel_config),
                data_parallel_size=len(phases),
                data_parallel_rank=rank,
                tensor_parallel_size=1,
            )
            r.vllm_config.observability_config = NS(cudagraph_metrics=False)
            r.parallel_config = r.vllm_config.parallel_config
            r.model_config = NS(is_encoder_decoder=False, is_hybrid=False)
            r.use_dcp = False
            r.uniform_decode_query_len = 1
            computed = [0, 8] if phase == "M" else [0 if phase == "P" else 8]
            r.input_batch = NS(
                num_computed_tokens_cpu=torch.tensor(computed),
                num_prompt_tokens=torch.tensor([8] * len(computed)),
                lora_id_to_lora_request={},
            )
            r._pad_for_sequence_parallelism = lambda n: n
            r.cudagraph_dispatcher = NS(
                dispatch=lambda num_tokens, valid_modes=None, rank=rank, **kw: (
                    next(iter(valid_modes)) if valid_modes else modes[rank],
                    NS(num_tokens=num_tokens, num_reqs=None),
                )
            )
            self.ns["get_dp_group"] = lambda: NS(cpu_group="cpu")

            def reduce(tensor, group, rank=rank):
                self.assertEqual(group, "cpu")
                torch.testing.assert_close(tensor[:, rank], packed[:, rank])
                tensor.copy_(packed)

            self.ns["dist"] = NS(all_reduce=reduce)
            mode, desc, _, counts, _ = r._determine_batch_execution_and_padding(
                tokens[rank],
                len(computed),
                None,
                tokens[rank],
                False,
                is_dummy_run=phase == "I",
                batch_contains_decode=True if phase == "I" else None,
            )
            backend = self.ns["select_moe_comm_method"](
                int(counts.max()),
                r.vllm_config,
                batch_contains_decode=r._synced_batch_contains_decode,
                aclgraph_runtime_mode=mode,
            )
            self.assertEqual(backend.name, expected)
            self.assertEqual(mode, min(modes))
            self.assertEqual(desc.num_tokens, int(counts[rank]))
            outputs.append(counts)
        for counts in outputs[1:]:
            torch.testing.assert_close(outputs[0], counts)
        return outputs[0]

    def test_prefill_idle_counts_and_graph_agree(self):
        for phases, tokens in [(["P", "I"], [1, 8]), (["P", "P", "I"], [4, 8, 1]), (["I", "P", "I"], [1, 16, 1])]:
            for unavailable in ((), tuple(range(len(phases)))):
                with self.subTest(phases=phases, unavailable=unavailable):
                    counts = self.batch(phases, tokens, "FUSED_MC2", unavailable)
                    self.assertEqual(counts.tolist(), [max(tokens)] * len(tokens))

    def test_real_decode_and_mixed_fall_back(self):
        self.batch(["P", "D"], [4, 8], "MC2")
        self.batch(["M", "I"], [4, 1], "MC2")
        self.batch(["D", "I"], [8, 1], "MC2")
        self.batch(["P", "D"], [128, 8], "ALLTOALL")

    def test_capture_v2_and_kv_consumer_reject_megamoe(self):
        select = self.ns["select_moe_comm_method"]
        for mode in (GraphMode.FULL, GraphMode.PIECEWISE):
            self.assertEqual(
                select(8, self.config, batch_contains_decode=False, aclgraph_runtime_mode=mode).name, "MC2"
            )
        for kwargs in ({"use_v2_model_runner": True}, {"kv_transfer_config": NS(is_kv_consumer=True)}):
            c = NS(**(vars(self.config) | kwargs))
            self.assertEqual(select(8, c, batch_contains_decode=False).name, "MC2")
        self.batch(["I", "I"], [8, 8], "MC2")

    def test_non_kimi_and_dense_draft_keep_030_policy(self):
        self.config.model_config.architectures = ["OtherMoE"]
        self.assertEqual(
            self.ns["select_moe_comm_method"](
                8, self.config, batch_contains_decode=True, aclgraph_runtime_mode=GraphMode.FULL
            ).name,
            "FUSED_MC2",
        )
        r = self.ns["NPUModelRunner"]()
        r.dp_size = 2
        r.dp_rank = 0
        r.vllm_config = self.config
        result = r._sync_metadata_across_dp(8, is_draft_model=True)
        self.assertEqual(result[1].tolist(), [8, 8])

    def test_overlap_exception_is_only_for_supported_kimi_a5(self):
        tree = ast.parse((ROOT / "vllm_ascend/ascend_config.py").read_text())
        fn = next(n for n in tree.body if getattr(n, "name", None) == "_supports_kimi_megamoe_overlap")
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), "overlap_policy", "exec"), self.ns)
        allowed = self.ns["_supports_kimi_megamoe_overlap"]
        self.assertTrue(allowed(self.config))
        self.config.model_config.architectures = ["OtherMoE"]
        self.assertFalse(allowed(self.config))
        self.config.model_config.architectures = ["KimiK3ForConditionalGeneration"]
        self.ns["is_mega_moe_supported"] = lambda: False
        self.assertFalse(allowed(self.config))
        self.ns["is_mega_moe_supported"] = lambda: True
        self.ns["get_current_hardware_profile"] = lambda: NS(supports=lambda cap: False)
        self.assertFalse(allowed(self.config))

    def test_310p_override_accepts_dummy_phase_arguments(self):
        tree = ast.parse((ROOT / "vllm_ascend/_310p/model_runner_310p.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner310")
        cls.body = [n for n in cls.body if getattr(n, "name", None) == "_determine_batch_execution_and_padding"]
        cls.decorator_list = []

        class Base:
            def _determine_batch_execution_and_padding(self, **kwargs):
                return kwargs

        namespace = dict(
            NPUModelRunner=Base,
            np=NS(any=lambda x: bool(x.any())),
            AscendAttentionState=NS(ChunkedPrefill=1, PrefillCacheHit=2, SpecDecoding=3, DecodeOnly=4),
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), "runner310", "exec"), namespace)
        runner = namespace["NPUModelRunner310"]()
        runner.input_batch = NS(num_computed_tokens_cpu=torch.tensor([0]), num_prompt_tokens=torch.tensor([8]))
        runner.attn_state = 0
        runner.speculative_config = None
        result = runner._determine_batch_execution_and_padding(
            1, 1, None, 1, False, is_dummy_run=True, batch_contains_decode=False
        )
        self.assertTrue(result["is_dummy_run"])
        self.assertFalse(result["batch_contains_decode"])


if __name__ == "__main__":
    unittest.main()
