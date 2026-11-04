# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU-only empty forward-context metadata boundary regression.

Run directly: python3 tests/ut/test_platform_empty_metadata_cpu.py
This executes the unchanged production hook with a dense MRv2, TP=DP=1
configuration. No model, communication kernel, or NPU runtime is imported.
"""

import __future__

import ast
import math
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

SOURCE = Path(__file__).resolve().parents[2] / "vllm_ascend/platform.py"


def load_platform():
    tree = ast.parse(SOURCE.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUPlatform")
    method = next(
        n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "set_additional_forward_context"
    )
    cls = ast.ClassDef(name="NPUPlatform", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"math": math, "is_moe_model": lambda config: False}
    exec(compile(module, str(SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["NPUPlatform"]


def module(name, **attrs):
    result = ModuleType(name)
    result.__dict__.update(attrs)
    return result


class PlatformEmptyMetadataTest(unittest.TestCase):
    def setUp(self):
        self.platform = load_platform()
        self.config = SimpleNamespace(
            use_v2_model_runner=True,
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        )
        self.modules = {
            "torch_npu": MagicMock(),
            "acl": MagicMock(),
            "vllm.config": module("vllm.config", CUDAGraphMode=SimpleNamespace(NONE=0)),
            "vllm.distributed": module(
                "vllm.distributed",
                get_dp_group=lambda: SimpleNamespace(world_size=1),
                get_tensor_model_parallel_world_size=lambda: 1,
            ),
            "vllm_ascend.ascend_forward_context": module(
                "vllm_ascend.ascend_forward_context",
                _is_decode_only_node=lambda config: False,
                get_mc2_mask=lambda: None,
                get_mrv2_in_profile_run=lambda: False,
                select_moe_comm_method=lambda *args: None,
                sync_v2_extra_kwargs=lambda config: None,
                use_cann_megamoe=lambda config: False,
            ),
            "vllm_ascend.ops.fused_moe.moe_comm_method": module(
                "vllm_ascend.ops.fused_moe.moe_comm_method",
                get_moe_comm_method=lambda kind: None,
            ),
            "vllm_ascend.quantization.utils": module(
                "vllm_ascend.quantization.utils",
                get_dynamic_mx_quant_scale_alg=lambda config: 0,
            ),
        }

    def test_empty_metadata_without_token_count(self):
        with patch.dict(sys.modules, self.modules):
            result = self.platform.set_additional_forward_context(
                attn_metadata={},
                vllm_config=self.config,
                dp_metadata=None,
                num_tokens=None,
            )
        self.assertIsNone(result["num_tokens"])
        self.assertIsNone(result["padded_num_tokens"])
        self.assertEqual(result["dynamic_mx_quant_scale_alg"], 0)

    def test_nonempty_metadata_infers_token_count(self):
        with patch.dict(sys.modules, self.modules):
            result = self.platform.set_additional_forward_context(
                attn_metadata={"layer": SimpleNamespace(num_actual_tokens=3)},
                vllm_config=self.config,
                dp_metadata=None,
                num_tokens=None,
            )
        self.assertEqual(result["num_tokens"], 3)
        self.assertEqual(result["padded_num_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
