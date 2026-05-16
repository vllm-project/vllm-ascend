#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
import torch_npu
from pytest_mock import MockerFixture

from tests.ut.base import TestBase
from tests.ut.conftest import npu_test
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEPrepareOutput
from vllm_ascend.utils import AscendDeviceType, adapt_patch, enable_custom_op

adapt_patch(True)


def mock_ep_and_mc2_group(mocker):
    mock_group = mocker.MagicMock()
    mock_group.rank_in_group = 0
    mock_group.rank = 0
    mock_group.world_size = 4
    mock_group.device_group = "mock_group_ep"
    mock_group.all_to_all = MagicMock(return_value=torch.randn(8, 8))
    return mock_group


def mock_dp_and_tp_group(mocker):
    mock_group = mocker.MagicMock()
    mock_group.rank_in_group = 0
    mock_group.world_size = 2
    mock_group.device_group = "mock_group"
    mock_group.all_gather = MagicMock(return_value=torch.randn(10, 32))
    return mock_group


def _require_ascend_custom_op(op_name: str):
    try:
        custom_op_enabled = enable_custom_op()
    except Exception as exc:
        pytest.skip(f"requires vllm_ascend custom ops: {exc}")
    if not custom_op_enabled:
        pytest.skip("requires vllm_ascend custom ops")
    try:
        getattr(torch.ops._C_ascend, op_name)
    except AttributeError:
        pytest.skip(f"requires torch.ops._C_ascend.{op_name}")


def _sort_topk_by_ids(topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_ids, order = torch.sort(topk_ids.to(torch.int64), dim=-1)
    sorted_weights = topk_weights.gather(1, order)
    return sorted_weights, sorted_ids.to(torch.int32)


@pytest.fixture(autouse=True)
def setup_vllm_config_mock(mocker: MockerFixture):
    mock_hf_config = MagicMock()
    mock_hf_config.model_type = "llama"

    mock_model_config = MagicMock()
    mock_model_config.hf_config = mock_hf_config

    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config = mock_model_config
    mock_vllm_config.parallel_config = MagicMock(tensor_parallel_size=2)
    mock_vllm_config.scheduler_config = MagicMock(max_num_seqs=4)
    mock_vllm_config.model_config.max_model_len = 2048

    mocker.patch("vllm_ascend.ops.fused_moe.fused_moe.get_current_vllm_config", return_value=mock_vllm_config)


@pytest.fixture
def mock_dist_env(mocker: MockerFixture):
    mock_moe_comm_method = MagicMock()

    def mock_prepare(hidden_states, router_logits, **kwargs):
        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=kwargs.get("mc2_mask"),
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )

    mock_moe_comm_method.prepare.side_effect = mock_prepare

    mock_fused_experts_result = torch.randn(16, 2)
    mock_moe_comm_method.fused_experts.return_value = mock_fused_experts_result

    def mock_finalize(hidden_states, **kwargs):
        return hidden_states

    mock_moe_comm_method.finalize.side_effect = mock_finalize
    dp_metadata = MagicMock(num_tokens_across_dp_cpu=[5, 5])
    mock_weight_prefetch_method = MagicMock()
    mock_forward_context_obj = MagicMock(
        moe_comm_method=mock_moe_comm_method,
        moe_comm_type=MoECommType.MC2,
        max_tokens_across_dp=10,
        dp_metadata=dp_metadata,
        mc2_mask=torch.zeros(16, dtype=torch.bool),
        padded_num_tokens=16,
        with_quant=False,
    )

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_ep_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.token_dispatcher.get_ep_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_mc2_group", return_value=mock_ep_and_mc2_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_tp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.model_executor.layers.fused_moe.layer.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch("vllm.model_executor.layers.fused_moe.config.get_dp_group", return_value=mock_dp_and_tp_group(mocker)),
        patch(
            "vllm_ascend.ops.fused_moe.fused_moe.get_ascend_config",
            return_value=MagicMock(enable_multistream_moe=False, expert_map_path=None),
        ),
        patch(
            "vllm_ascend.ops.fused_moe.fused_moe.init_eplb_config",
            return_value=(torch.tensor([0, 1, 2, -1, -1, -1, -1, -1]), None, 0),
        ),
        patch("vllm_ascend.ops.fused_moe.fused_moe.get_forward_context", return_value=mock_forward_context_obj),
        patch("vllm_ascend.ascend_forward_context.get_forward_context", return_value=mock_forward_context_obj),
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.MC2CommImpl._get_token_dispatcher", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.AlltoAllCommImpl._get_token_dispatcher", return_value=None),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.AllGatherCommImpl._get_token_dispatcher", return_value=None),
        patch(
            "vllm_ascend.ops.fused_moe.experts_selector.get_weight_prefetch_method",
            return_value=mock_weight_prefetch_method,
        ),
    ):
        yield {
            "mock_forward_context_obj": mock_forward_context_obj,
            "mock_moe_comm_method": mock_moe_comm_method,
        }


@pytest.fixture
def mock_moe_env(mocker: MockerFixture):
    with (
        patch(
            "torch_npu.npu_moe_init_routing",
            return_value=(torch.randn(8, 2), torch.randint(0, 8, (8, 2)), torch.tensor([0, 1, 2, 4, 6, 2, 7, 1])),
        ),
        patch("torch_npu.npu_moe_compute_expert_tokens", return_value=(torch.randn(8, 2))),
        patch("torch_npu.npu_moe_distribute_dispatch", return_value=(torch.randn(16, 2))),
        patch("torch_npu.npu_moe_distribute_combine", return_value=(torch.randn(16, 2))),
        patch("torch_npu.npu_grouped_matmul", return_value=([torch.randn(16, 2)])),
        patch("torch_npu.npu_swiglu", return_value=(torch.randn(16, 2))),
        patch("torch_npu.npu_moe_finalize_routing", return_value=(torch.randn(16, 2))),
    ):
        if hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            with (
                patch("torch_npu.npu_moe_distribute_dispatch_v2", return_value=(torch.randn(16, 2))),
                patch("torch_npu.npu_moe_distribute_combine_v2", return_value=(torch.randn(16, 2))),
            ):
                yield
        else:
            yield


class TestExpertsSelector:
    @pytest.mark.parametrize("num_experts", [256, 128])
    def test_select_experts(self, mock_dist_env, mock_moe_env, num_experts):
        x = torch.randn(8, 2)
        router_logits = torch.randn(8, 2)
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            num_experts=num_experts,
        )

        assert topk_weights.shape == (8, 2)
        assert topk_ids.shape == (8, 2)

    @pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
    @pytest.mark.parametrize("renormalize", [True, False])
    def test_select_experts_with_different_scoring_func(self, mock_dist_env, mock_moe_env, scoring_func, renormalize):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def simple_custom_routing(hidden_states, gating_output, topk, renormalize, num_experts):
            if scoring_func == "softmax":
                weights = gating_output.softmax(dim=-1)
            else:
                weights = gating_output.sigmoid()
            topk_weights, topk_ids = weights.topk(topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            return topk_weights, topk_ids.to(torch.int32)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=renormalize,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=simple_custom_routing,
            scoring_func=scoring_func,
            e_score_correction_bias=None,
            num_experts=num_experts,
        )

        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)
        assert topk_weights.dtype == hidden_states.dtype
        assert topk_ids.dtype == torch.int32

        if renormalize:
            weight_sum = topk_weights.sum(dim=-1)
            torch.testing.assert_close(weight_sum, torch.ones_like(weight_sum), rtol=1e-4, atol=1e-4)

    def test_select_experts_with_grouped_topk(self, mock_dist_env, mock_moe_env):
        _require_ascend_custom_op("moe_gating_top_k")

        num_tokens = 16
        num_experts = 8
        hidden_size = 32
        num_expert_group = 4
        topk_group = 2

        hidden_states = torch.randn(num_tokens, hidden_size, device="npu")
        router_logits = torch.randn(num_tokens, num_experts, device="npu")

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=True,
            renormalize=True,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            num_experts=num_experts,
        )

        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)

    def test_select_experts_with_e_score_correction_bias(self, mock_dist_env, mock_moe_env):
        _require_ascend_custom_op("moe_gating_top_k")

        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size, device="npu")
        router_logits = torch.randn(num_tokens, num_experts, device="npu")
        e_score_correction_bias = torch.randn(num_experts, device="npu")

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_experts,
        )

        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)

    def test_select_experts_with_custom_routing_function(self, mock_dist_env, mock_moe_env):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def custom_routing(hidden_states, gating_output, topk, renormalize, num_experts):
            weights = gating_output.softmax(dim=-1)
            topk_weights, topk_ids = weights.topk(topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            return topk_weights, topk_ids.to(torch.int32)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=custom_routing,
            scoring_func="softmax",
            e_score_correction_bias=None,
            num_experts=num_experts,
        )

        assert topk_weights.shape == (num_tokens, 2)
        assert topk_ids.shape == (num_tokens, 2)

    def test_select_experts_weight_sum_range(self, mock_dist_env, mock_moe_env):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def simple_routing(hidden_states, gating_output, topk, renormalize, num_experts):
            weights = gating_output.softmax(dim=-1)
            topk_weights, topk_ids = weights.topk(topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            return topk_weights, topk_ids.to(torch.int32)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=simple_routing,
            scoring_func="softmax",
            e_score_correction_bias=None,
            num_experts=num_experts,
        )

        assert (topk_weights >= 0).all()
        assert (topk_weights <= 1).all()

        weight_sum = topk_weights.sum(dim=-1)
        torch.testing.assert_close(weight_sum, torch.ones_like(weight_sum), rtol=1e-4, atol=1e-4)

    def test_select_experts_expert_id_range(self, mock_dist_env, mock_moe_env):
        num_tokens = 16
        num_experts = 8
        hidden_size = 32

        hidden_states = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def simple_routing(hidden_states, gating_output, topk, renormalize, num_experts):
            weights = gating_output.softmax(dim=-1)
            topk_weights, topk_ids = weights.topk(topk, dim=-1)
            if renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            return topk_weights, topk_ids.to(torch.int32)

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=simple_routing,
            scoring_func="softmax",
            e_score_correction_bias=None,
            num_experts=num_experts,
        )

        assert (topk_ids >= 0).all()
        assert (topk_ids < num_experts).all()

    @patch("vllm_ascend.ops.fused_moe.experts_selector.get_weight_prefetch_method")
    @patch("vllm_ascend.ops.fused_moe.experts_selector.check_npu_moe_gating_top_k", return_value=False)
    def test_select_experts_native_softmax_matches_expected(self, _, mock_get_weight_prefetch_method):
        hidden_states = torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.5, 1.5, -0.5, 1.0]], dtype=torch.float32)
        router_logits = torch.tensor([[3.0, 1.0, 0.0, 2.0], [0.0, 4.0, 1.0, 2.0]], dtype=torch.float32)

        prefetch_method = MagicMock()
        mock_get_weight_prefetch_method.return_value = prefetch_method

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            num_experts=4,
        )

        expected_probs = router_logits.softmax(dim=-1)
        expected_weights, expected_ids = expected_probs.topk(2, dim=-1)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

        prefetch_method.maybe_prefetch_moe_weight_preprocess.assert_called_once_with(hidden_states, "gate_up")
        torch.testing.assert_close(topk_weights, expected_weights.to(hidden_states.dtype))
        assert torch.equal(topk_ids, expected_ids.to(torch.int32))

    def test_select_experts_grouped_topk_bias_uses_original_weights(self):
        _require_ascend_custom_op("moe_gating_top_k")

        hidden_states = torch.tensor([[1.0, 0.0, -1.0, 2.0]], dtype=torch.float32, device="npu")
        router_logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]], dtype=torch.float32, device="npu")
        e_score_correction_bias = torch.tensor([-10.0, -10.0, 5.0, 5.0], dtype=torch.float32, device="npu")

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=True,
            renormalize=False,
            topk_group=1,
            num_expert_group=2,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=e_score_correction_bias,
            num_experts=4,
        )

        original_weights = router_logits.softmax(dim=-1)
        actual_weights, actual_ids = _sort_topk_by_ids(topk_weights.cpu(), topk_ids.cpu())
        expected_ids = torch.tensor([[2, 3]], dtype=torch.int32)
        expected_weights = original_weights[:, 2:4].to("cpu")

        assert torch.equal(actual_ids, expected_ids)
        torch.testing.assert_close(actual_weights, expected_weights)

    @patch("vllm_ascend.ops.fused_moe.experts_selector.get_weight_prefetch_method", return_value=MagicMock())
    @patch("vllm_ascend.ops.fused_moe.experts_selector.check_npu_moe_gating_top_k", return_value=False)
    def test_select_experts_invalid_scoring_func_raises(self, _, __):
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 4)

        with pytest.raises(ValueError, match="Unsupported scoring function"):
            select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=2,
                use_grouped_topk=False,
                renormalize=True,
                topk_group=None,
                num_expert_group=None,
                custom_routing_function=None,
                scoring_func="unsupported",
                e_score_correction_bias=None,
                num_experts=4,
            )


@npu_test(num_npus=2, npu_type="a3")
class TestFusedMoENPUOpsAccuracy:
    def test_moe_gating_top_k_matches_native_softmax(self, mock_dist_env):
        _require_ascend_custom_op("moe_gating_top_k")

        hidden_states = torch.randn(4, 16, dtype=torch.float32, device="npu")
        router_logits_cpu = torch.tensor(
            [[3.0, 1.0, 0.0, 2.0], [0.0, 4.0, 1.0, 2.0], [1.0, 0.5, 5.0, -1.0], [2.5, -0.5, 1.5, 0.25]],
            dtype=torch.float32,
        )
        router_logits = router_logits_cpu.to("npu")

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            topk_group=None,
            num_expert_group=None,
            custom_routing_function=None,
            scoring_func="softmax",
            e_score_correction_bias=None,
            num_experts=4,
        )

        expected_weights, expected_ids = router_logits_cpu.softmax(dim=-1).topk(2, dim=-1)
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
        actual_weights, actual_ids = _sort_topk_by_ids(topk_weights.cpu(), topk_ids.cpu())
        expected_weights, expected_ids = _sort_topk_by_ids(expected_weights, expected_ids.to(torch.int32))

        assert torch.equal(actual_ids, expected_ids)
        torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-3, atol=1e-3)

    def test_npu_swiglu_matches_torch_reference(self):
        gate_up_cpu = torch.tensor(
            [[1.0, -2.0, 0.5, 3.0], [-0.5, 2.0, 1.5, -1.0], [4.0, -3.0, -2.0, 0.25]],
            dtype=torch.float16,
        )
        gate_up = gate_up_cpu.to("npu")

        actual = torch_npu.npu_swiglu(gate_up)
        left, right = gate_up_cpu.chunk(2, dim=-1)
        expected = F.silu(left.float()) * right.float()

        torch.testing.assert_close(actual.cpu().float(), expected, rtol=2e-3, atol=2e-3)

    def test_npu_grouped_matmul_matches_torch_reference(self):
        hidden_states_cpu = torch.tensor(
            [[1.0, 2.0, -1.0], [0.5, -0.5, 1.0], [2.0, 0.0, 1.0], [-1.0, 3.0, 0.5]],
            dtype=torch.float16,
        )
        weight_cpu = torch.tensor(
            [[[1.0, 0.0], [0.5, -1.0], [-0.5, 2.0]], [[-1.0, 1.5], [2.0, 0.25], [0.0, -0.5]]],
            dtype=torch.float16,
        )
        group_list = torch.tensor([2, 2], dtype=torch.int64, device="npu")

        actual = torch_npu.npu_grouped_matmul(
            x=[hidden_states_cpu.to("npu")],
            weight=[weight_cpu.to("npu")],
            split_item=3,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.float16,
        )[0]
        expected = torch.cat(
            [
                hidden_states_cpu[:2].float() @ weight_cpu[0].float(),
                hidden_states_cpu[2:].float() @ weight_cpu[1].float(),
            ],
            dim=0,
        )

        torch.testing.assert_close(actual.cpu().float(), expected, rtol=2e-3, atol=2e-3)


class TestZeroExpertsCompute(TestBase):
    def test_zero_experts_compute_identity_type(self):
        num_experts = 8
        num_tokens = 4
        top_k = 2
        hidden_size = 16

        expert_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32)
        expert_scales = torch.ones(num_tokens, top_k, dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert result_indices.shape == expert_indices.shape
        assert result_scales.shape == expert_scales.shape
        assert result_hidden.shape == hidden_states.shape

    def test_zero_experts_compute_with_zero_experts(self):
        num_experts = 4
        num_tokens = 3
        hidden_size = 8

        expert_indices = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
        expert_scales = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.4, 0.6]], dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert result_indices.shape == expert_indices.shape
        assert result_scales.shape == expert_scales.shape
        assert result_hidden.shape == hidden_states.shape

    def test_zero_experts_compute_normal_experts_masked(self):
        num_experts = 4
        num_tokens = 2
        top_k = 2
        hidden_size = 8

        expert_indices = torch.tensor([[0, 5], [6, 7]], dtype=torch.int32)
        expert_scales = torch.ones(num_tokens, top_k, dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, _ = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        normal_expert_mask = expert_indices >= num_experts
        for i in range(num_tokens):
            for j in range(top_k):
                if normal_expert_mask[i, j]:
                    assert result_scales[i, j] == 0.0
                    assert result_indices[i, j] == 0

    def test_zero_experts_compute_output_sum(self):
        num_experts = 2
        num_tokens = 2
        hidden_size = 4

        expert_indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        expert_scales = torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        _, _, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert result_hidden.shape == (num_tokens, hidden_size)

    def test_zero_experts_compute_all_zero_experts(self):
        num_experts = 4
        num_tokens = 2
        top_k = 2
        hidden_size = 8

        expert_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        expert_scales = torch.ones(num_tokens, top_k, dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert torch.equal(result_indices, expert_indices)
        assert torch.equal(result_scales, expert_scales)
        assert result_hidden.shape == hidden_states.shape

    def test_zero_experts_compute_mixed_experts(self):
        num_experts = 3
        num_tokens = 2
        hidden_size = 8

        expert_indices = torch.tensor([[0, 1, 4], [2, 5, 6]], dtype=torch.int32)
        expert_scales = torch.tensor([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]], dtype=torch.float32)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.float32)

        result_indices, result_scales, _ = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=num_experts,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        assert result_indices.shape == expert_indices.shape
        assert result_scales.shape == expert_scales.shape

    def test_zero_experts_compute_identity_values_match_expected(self):
        expert_indices = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32)
        expert_scales = torch.tensor([[0.25, 0.75], [0.60, 0.40]], dtype=torch.float32)
        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        result_indices, result_scales, result_hidden = zero_experts_compute(
            expert_indices=expert_indices,
            expert_scales=expert_scales,
            num_experts=2,
            zero_expert_type="identity",
            hidden_states=hidden_states,
        )

        expected_indices = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
        expected_scales = torch.tensor([[0.25, 0.0], [0.0, 0.40]], dtype=torch.float32)
        expected_hidden = torch.tensor([[0.75, 1.50], [1.80, 2.40]], dtype=torch.float32)

        assert torch.equal(result_indices, expected_indices)
        torch.testing.assert_close(result_scales, expected_scales)
        torch.testing.assert_close(result_hidden, expected_hidden)
