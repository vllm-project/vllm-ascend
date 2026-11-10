# SPDX-License-Identifier: Apache-2.0
"""Check empty-rank LoRA milestones without importing NPU dependencies."""

import ast
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call


class TestQuantMoELoRAEmptyOverlap(unittest.TestCase):
    def test_empty_routed_rank_preserves_gmm2_milestone(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/lora/quant_moe.py"
        function = next(
            node
            for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_dynamic_int8_moe_lora"
        )
        function.decorator_list = []
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        MoECommType = Enum("MoECommType", "ALLGATHER ALLTOALL MC2 FUSED_MC2")
        int8 = object()

        for comm_type in (MoECommType.ALLTOALL, MoECommType.ALLGATHER):
            with self.subTest(comm_type=comm_type):
                event = object()
                stream = Mock()
                stream.record_event.return_value = event
                current_stream = Mock(return_value=stream)
                device_operator = Mock()
                torch_npu = Mock()
                quant_method = Mock()
                hidden_states = SimpleNamespace(shape=(0, 4), dtype=object())
                mlp_input = SimpleNamespace(
                    hidden_states=hidden_states,
                    dynamic_scale=None,
                    dynamic_eplb=False,
                    lora_context=object(),
                    expanded_row_idx=object() if comm_type is MoECommType.ALLGATHER else None,
                    topk_ids=object() if comm_type is MoECommType.ALLGATHER else None,
                )
                namespace: dict[str, Any] = {
                    "torch": SimpleNamespace(int8=int8, npu=SimpleNamespace(current_stream=current_stream)),
                    "torch_npu": torch_npu,
                    "DeviceOperator": device_operator,
                    "MoECommType": MoECommType,
                    "_EXTRA_CTX": SimpleNamespace(moe_comm_type=comm_type),
                }
                exec(compile(module, str(source), "exec"), namespace)

                output, output_event = namespace["_apply_dynamic_int8_moe_lora"](mlp_input, quant_method)

                self.assertIs(output, hidden_states)
                self.assertIs(output_event, event)
                current_stream.assert_called_once_with()
                self.assertEqual(stream.mock_calls, [call.record_event()])
                self.assertEqual(device_operator.mock_calls, [])
                self.assertEqual(torch_npu.mock_calls, [])
                self.assertEqual(quant_method.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
