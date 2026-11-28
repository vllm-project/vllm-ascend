# SPDX-License-Identifier: Apache-2.0
"""Check empty-rank LoRA milestones and index cleanup."""

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, call, patch

import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.lora.quant_moe import _apply_dynamic_int8_moe_lora


class TestQuantMoELoRAEmptyOverlap(unittest.TestCase):
    def test_empty_routed_rank_preserves_gmm2_milestone(self):
        for comm_type in (MoECommType.ALLTOALL, MoECommType.ALLGATHER):
            with self.subTest(comm_type=comm_type):
                event = object()
                stream = Mock()
                stream.record_event.return_value = event
                quant_method = Mock()
                hidden_states = torch.empty((0, 4), dtype=torch.bfloat16)
                lora_context = SimpleNamespace(
                    split_lora_indices=torch.tensor([0]),
                    permuted_lora_indices=torch.tensor([0]),
                    exchanged_lora_indices=torch.tensor([0]),
                    gathered_lora_indices=torch.tensor([0]),
                )
                mlp_input = SimpleNamespace(
                    hidden_states=hidden_states,
                    dynamic_scale=None,
                    lora_context=lora_context,
                    expanded_row_idx=object() if comm_type is MoECommType.ALLGATHER else None,
                    topk_ids=object() if comm_type is MoECommType.ALLGATHER else None,
                )
                with (
                    patch("vllm_ascend.lora.quant_moe._EXTRA_CTX", SimpleNamespace(moe_comm_type=comm_type)),
                    patch("vllm_ascend.lora.quant_moe.get_current_vllm_config_or_none", return_value=None),
                    patch(
                        "vllm_ascend.ops.fused_moe.moe_mlp.get_ascend_config",
                        return_value=SimpleNamespace(multistream_overlap_shared_expert=True),
                    ),
                    patch(
                        "vllm_ascend.ops.fused_moe.moe_mlp.torch.npu.current_stream",
                        return_value=stream,
                    ) as current_stream,
                    patch.object(DeviceOperator, "npu_dynamic_quant") as mock_dynamic_quant,
                    patch.object(DeviceOperator, "npu_grouped_matmul_gmm2") as mock_gmm2,
                    patch("vllm_ascend.lora.quant_moe.torch_npu.npu_grouped_matmul", create=True) as mock_gmm,
                ):
                    output, output_event = _apply_dynamic_int8_moe_lora(cast(Any, mlp_input), quant_method)

                self.assertIs(output, hidden_states)
                self.assertIs(output_event, event)
                current_stream.assert_called_once_with()
                self.assertEqual(stream.mock_calls, [call.record_event()])
                self.assertFalse(hasattr(lora_context, "split_lora_indices"))
                self.assertFalse(hasattr(lora_context, "permuted_lora_indices"))
                self.assertFalse(hasattr(lora_context, "exchanged_lora_indices"))
                self.assertFalse(hasattr(lora_context, "gathered_lora_indices"))
                mock_dynamic_quant.assert_not_called()
                mock_gmm2.assert_not_called()
                mock_gmm.assert_not_called()
                self.assertEqual(quant_method.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
