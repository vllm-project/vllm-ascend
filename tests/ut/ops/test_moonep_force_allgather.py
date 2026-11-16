# SPDX-License-Identifier: Apache-2.0
"""CPU policy checks for native A5 selection without a forced AllGather mode."""

import ast
import os
import unittest
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ascend_forward_context.py"


class Comm(Enum):
    MC2 = auto()
    FUSED_MC2 = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()
    SHMEM = auto()


class Device(Enum):
    A2 = auto()
    A3 = auto()
    A5 = auto()
    _310P = auto()


class NormalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.enabled = False
        self.keep_mc2 = False
        self.is_moe = True
        self.ep_world_size = 8
        self.soc = Device.A5
        self.shmem_capacity = 4096
        self.config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_experts_per_tok=6)),
            parallel_config=SimpleNamespace(enable_expert_parallel=True, world_size_across_dp=8, pipeline_parallel_size=1),
        )
        self.a2 = Mock(return_value=Comm.ALLGATHER)
        self.a3 = Mock(return_value=Comm.FUSED_MC2)
        self.envs = SimpleNamespace(MOONEP_KEEP_MC2=False)
        namespace = {
            "os": os,
            "VllmConfig": SimpleNamespace,
            "MoECommType": Comm,
            "AscendDeviceType": Device,
            "logger": SimpleNamespace(debug=Mock()),
            "envs": self.envs,
            "is_moe_model": lambda config: self.is_moe,
            "get_ascend_config": lambda: SimpleNamespace(enable_moonep=self.enabled, enable_fused_mc2=0),
            "get_ep_group": lambda: SimpleNamespace(world_size=self.ep_world_size),
            "get_mc2_tokens_capacity": lambda: 256,
            "get_moonep_tokens_capacity": lambda: self.shmem_capacity,
            "get_ascend_device_type": lambda: self.soc,
            "QuantType": SimpleNamespace(NONE=object()),
        }
        policies = {device: object() for device in Device}

        def hardware_profile():
            if self.soc not in policies:
                raise ValueError("Unsupported soc_version")
            return SimpleNamespace(moe_comm_policy=policies[self.soc])

        namespace["get_current_hardware_profile"] = hardware_profile
        namespace["MoECommPolicy"] = SimpleNamespace(ALLGATHER=policies[Device._310P])
        names = {"_select_capacity_and_world_size_moe_comm_method", "select_moe_comm_method"}
        functions = [
            n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
        namespace["_MOE_COMM_SELECTORS"] = {
            policies[Device.A5]: namespace["_select_capacity_and_world_size_moe_comm_method"],
            policies[Device.A2]: self.a2,
            policies[Device.A3]: self.a3,
        }
        self.select_method = namespace["select_moe_comm_method"]

    def select(self, tokens, legacy="1"):
        self.envs.MOONEP_KEEP_MC2 = self.keep_mc2
        with patch.dict(os.environ, {}, clear=True):
            if legacy is not None:
                os.environ["MOONEP_FORCE_ALLGATHER"] = legacy
            return self.select_method(tokens, self.config)

    def test_old_zero_one_and_unset_all_allow_native_mc2(self):
        for legacy in (None, "0", "1"):
            with self.subTest(legacy=legacy):
                self.assertIs(self.select(128, legacy), Comm.MC2)

    def test_large_native_a5_step_uses_alltoall(self):
        for legacy in ("0", "1"):
            with self.subTest(legacy=legacy):
                self.assertIs(self.select(512, legacy), Comm.ALLTOALL)

    def test_natural_allgather_remains_for_small_parallel_world(self):
        self.config.parallel_config.world_size_across_dp = 4
        self.ep_world_size = 4
        self.assertIs(self.select(512), Comm.ALLGATHER)
        self.assertIs(self.select(128), Comm.MC2)

    def test_topk_alias_and_default_still_control_natural_allgather(self):
        hf = self.config.model_config.hf_text_config
        del hf.num_experts_per_tok
        hf.top_k_experts = 8
        self.assertIs(self.select(512), Comm.ALLGATHER)
        del hf.top_k_experts
        self.assertIs(self.select(512), Comm.ALLTOALL)

    def test_native_non_ep_or_single_rank_still_uses_allgather(self):
        self.config.parallel_config.enable_expert_parallel = False
        self.assertIs(self.select(128), Comm.ALLGATHER)
        self.config.parallel_config.enable_expert_parallel = True
        self.ep_world_size = 1
        self.assertIs(self.select(128), Comm.ALLGATHER)

    def test_moonep_overrides_native_choices_when_keep_mc2_is_off(self):
        self.enabled = True
        self.assertIs(self.select(128), Comm.SHMEM)
        self.assertIs(self.select(512), Comm.SHMEM)
        self.config.parallel_config.world_size_across_dp = 4
        self.assertIs(self.select(512), Comm.SHMEM)

    def test_keep_mc2_only_keeps_the_native_mc2_choices(self):
        self.enabled = True
        self.keep_mc2 = True
        self.assertIs(self.select(128), Comm.MC2)
        self.assertIs(self.select(512), Comm.SHMEM)
        self.config.parallel_config.world_size_across_dp = 4
        self.assertIs(self.select(512), Comm.SHMEM)
        self.soc = Device.A3
        self.assertIs(self.select(128), Comm.FUSED_MC2)

    def test_shmem_capacity_overflow_still_raises(self):
        self.enabled = True
        for keep in (False, True):
            with self.subTest(keep=keep):
                self.keep_mc2 = keep
                with self.assertRaisesRegex(RuntimeError, "SHMEM capacity"):
                    self.select(4097)

    def test_moonep_invalid_expert_parallel_still_raises(self):
        self.enabled = True
        self.config.parallel_config.enable_expert_parallel = False
        with self.assertRaisesRegex(RuntimeError, "requires expert parallel"):
            self.select(128)
        self.config.parallel_config.enable_expert_parallel = True
        self.ep_world_size = 1
        with self.assertRaisesRegex(RuntimeError, "requires expert parallel"):
            self.select(128)

    def test_other_device_selection_helpers_remain_independent(self):
        self.soc = Device.A2
        self.assertIs(self.select(128), Comm.ALLGATHER)
        self.a2.assert_called_once()
        self.soc = Device.A3
        self.assertIs(self.select(128), Comm.FUSED_MC2)
        self.a3.assert_called_once()
        self.soc = Device._310P
        self.assertIs(self.select(128), Comm.ALLGATHER)

    def test_dense_model_keeps_its_early_return(self):
        self.is_moe = False
        self.enabled = True
        self.ep_world_size = 1
        self.assertIsNone(self.select(128))

    def test_unsupported_device_still_raises(self):
        self.soc = object()
        with self.assertRaisesRegex(ValueError, "Unsupported soc_version"):
            self.select(128)


if __name__ == "__main__":
    unittest.main()
