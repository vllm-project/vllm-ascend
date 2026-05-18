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
import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytest_mock import MockerFixture

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendFusedMoE,
    AscendMoERunner,
    AscendUnquantizedFusedMoEMethod,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEPrepareOutput,
    MoEQuantParams,
    MoEWeights,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import AscendDeviceType, adapt_patch

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


def mock_npu_format_cast(weight_data, format):
    return weight_data


def build_mlp_compute_input_fixture(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor | list[torch.Tensor],
    w2: torch.Tensor | list[torch.Tensor],
    group_list: torch.Tensor,
    with_quant: bool,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor | None = None,
    topk_scales: torch.Tensor | None = None,
    w1_scale: torch.Tensor | list[torch.Tensor] | None = None,
    w2_scale: torch.Tensor | list[torch.Tensor] | None = None,
    w1_scale_bias: torch.Tensor | None = None,
    w2_scale_bias: torch.Tensor | None = None,
    w1_offset: torch.Tensor | None = None,
    w2_offset: torch.Tensor | None = None,
    fusion: bool = False,
    activation: str = "silu",
    need_trans: bool = True,
    dynamic_eplb: bool = False,
) -> MoEMlpComputeInput:
    return MoEMlpComputeInput(
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=group_list_type,
        dynamic_scale=dynamic_scale,
        topk_scales=topk_scales,
        weights=MoEWeights(
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_scale_bias=w1_scale_bias,
            w2_scale_bias=w2_scale_bias,
            w1_offset=w1_offset,
            w2_offset=w2_offset,
        ),
        quant=MoEQuantParams(quant_type=QuantType.W8A8 if with_quant else QuantType.NONE),
        fusion=fusion,
        activation=activation,
        need_trans=need_trans,
        dynamic_eplb=dynamic_eplb,
    )


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
def default_moe_config():
    return {"num_experts": 8, "top_k": 2, "hidden_size": 512, "intermediate_size": 1024}


@pytest.fixture
def moe_method(mock_dist_env):
    moe = MagicMock()
    moe.moe_parallel_config.return_value = MagicMock(ep_size=4)
    moe.moe_parallel_config.use_ep = False
    moe.moe_parallel_config.dp_size = 1
    return AscendUnquantizedFusedMoEMethod(moe)


def test_ascend_unquantized_skips_upstream_modular_kernel_init():
    method = AscendUnquantizedFusedMoEMethod.maybe_make_prepare_finalize

    assert method(object()) is None


class Device(TypedDict):
    device_id: int
    device_expert: list[int]


class Layer(TypedDict):
    layer_id: int
    device_count: int
    device_list: list[Device]


class MockData(TypedDict):
    moe_layer_count: int
    layer_list: list[Layer]


class MockQuantMethod(nn.Module):
    def __init__(self, shared_experts, num_tokens):
        super().__init__()
        if shared_experts:
            self.apply = MagicMock(return_value=(torch.randn(num_tokens, 32), torch.randn(num_tokens, 10)))
        else:
            self.apply = MagicMock(return_value=(torch.randn(num_tokens, 32)))


def _drop_self(signature: inspect.Signature) -> list[inspect.Parameter]:
    params = list(signature.parameters.values())
    if params and params[0].name == "self":
        return params[1:]
    return params


def _format_signature_mismatch(method_name: str, issues: list[str]) -> str:
    return f"{method_name} signature is not aligned with vLLM parent: " + "; ".join(issues)


def _assert_child_signature_accepts_parent_interface(child_method, parent_method):
    child_params = _drop_self(inspect.signature(child_method))
    parent_params = _drop_self(inspect.signature(parent_method))
    child_by_name = {
        param.name: param
        for param in child_params
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    child_has_var_positional = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in child_params)
    child_has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in child_params)
    issues: list[str] = []

    for parent_param in parent_params:
        if parent_param.kind == inspect.Parameter.VAR_POSITIONAL:
            if not child_has_var_positional:
                issues.append("child is missing *args from parent")
            continue

        if parent_param.kind == inspect.Parameter.VAR_KEYWORD:
            if not child_has_var_keyword:
                issues.append("child is missing **kwargs from parent")
            continue

        child_param = child_by_name.get(parent_param.name)
        if child_param is None:
            if parent_param.kind == inspect.Parameter.KEYWORD_ONLY:
                if not child_has_var_keyword:
                    issues.append(f"missing keyword-only parameter {parent_param.name!r}")
            elif not child_has_var_positional and not child_has_var_keyword:
                issues.append(f"missing parameter {parent_param.name!r}")
            continue

        if parent_param.kind != child_param.kind:
            issues.append(
                f"parameter {parent_param.name!r} has kind {child_param.kind!s}, expected {parent_param.kind!s}"
            )

    parent_param_names = {param.name for param in parent_params}
    for child_param in child_params:
        if child_param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if child_param.name in parent_param_names:
            continue
        if child_param.default is inspect.Parameter.empty:
            issues.append(f"extra parameter {child_param.name!r} must be optional")

    assert not issues, _format_signature_mismatch(parent_method.__qualname__, issues)


def _method_uses_super(method) -> bool:
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return False

    tree = ast.parse(textwrap.dedent(source))
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"
        for node in ast.walk(tree)
    )


class TestVllmParentInterfaceCompatibility:
    @pytest.mark.parametrize(
        "child_cls,parent_cls,method_name",
        [
            (AscendUnquantizedFusedMoEMethod, fused_moe_module.UnquantizedFusedMoEMethod, "__init__"),
            (
                AscendUnquantizedFusedMoEMethod,
                fused_moe_module.UnquantizedFusedMoEMethod,
                "process_weights_after_loading",
            ),
            (AscendUnquantizedFusedMoEMethod, fused_moe_module.UnquantizedFusedMoEMethod, "apply"),
            (AscendMoERunner, fused_moe_module.MoERunner, "__init__"),
            (AscendMoERunner, fused_moe_module.MoERunner, "forward_impl"),
            (AscendMoERunner, fused_moe_module.MoERunner, "_forward_impl"),
            (AscendFusedMoE, fused_moe_module.FusedMoE, "__init__"),
            (AscendFusedMoE, fused_moe_module.FusedMoE, "forward"),
            (AscendFusedMoE, fused_moe_module.FusedMoE, "forward_impl"),
            (AscendFusedMoE, fused_moe_module.FusedMoE, "maybe_all_reduce_tensor_model_parallel"),
        ],
    )
    def test_overridden_method_signature_accepts_parent_interface(self, child_cls, parent_cls, method_name):
        child_method = getattr(child_cls, method_name)
        if not _method_uses_super(child_method):
            pytest.skip(
                f"{child_cls.__name__}.{method_name} does not call "
                "super(), so parent interface alignment is not "
                "required"
            )

        if not hasattr(parent_cls, method_name):
            pytest.fail(
                f"{child_cls.__name__}.{method_name} calls super(), but {parent_cls.__name__} has no {method_name}"
            )

        _assert_child_signature_accepts_parent_interface(
            child_method,
            getattr(parent_cls, method_name),
        )


class TestAscendUnquantizedFusedMoEMethod:
    def _build_layer(self, *, has_bias=True, zero_expert_num=0):
        layer = MagicMock()
        layer.w13_weight = nn.Parameter(torch.randn(2, 3, 4))
        layer.w2_weight = nn.Parameter(torch.randn(2, 4, 3))
        layer.w13_bias = torch.randn(2, 4) if has_bias else None
        layer.w2_bias = torch.randn(2, 3) if has_bias else None
        layer.zero_expert_num = zero_expert_num
        layer.zero_expert_type = "identity" if zero_expert_num > 0 else None
        layer.n_shared_experts = 0
        layer.moe_config = SimpleNamespace(num_logical_experts=None)
        layer.layer_id = 3
        layer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enable_return_routed_experts=False))
        return layer

    @pytest.mark.parametrize("enable_fused_mc2", [True, False])
    def test_process_weights_after_loading_transposes_and_formats(self, monkeypatch, enable_fused_mc2):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method._maybe_pad_weight = MagicMock(side_effect=lambda weight: weight)
        layer = self._build_layer()
        original_w13 = layer.w13_weight.detach().clone()
        original_w2 = layer.w2_weight.detach().clone()
        format_cast = MagicMock(side_effect=lambda weight, _: weight)
        maybe_trans_nz = MagicMock(side_effect=lambda weight: weight)

        monkeypatch.setattr(fused_moe_module.envs_ascend, "VLLM_ASCEND_ENABLE_FUSED_MC2", enable_fused_mc2)
        monkeypatch.setattr(fused_moe_module.torch_npu, "npu_format_cast", format_cast)
        monkeypatch.setattr(fused_moe_module, "maybe_trans_nz", maybe_trans_nz)

        method.process_weights_after_loading(layer)

        torch.testing.assert_close(layer.w13_weight, original_w13.transpose(1, 2).contiguous())
        torch.testing.assert_close(layer.w2_weight, original_w2.transpose(1, 2).contiguous())
        if enable_fused_mc2:
            assert format_cast.call_count == 2
            maybe_trans_nz.assert_not_called()
        else:
            assert maybe_trans_nz.call_count == 2
            format_cast.assert_not_called()

    @pytest.mark.parametrize("moe_comm_type", [MoECommType.MC2, MoECommType.FUSED_MC2])
    def test_apply_builds_fused_experts_input(self, monkeypatch, moe_comm_type):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=True)
        method.dynamic_eplb = False
        layer = self._build_layer(has_bias=True)
        hidden_states = torch.randn(2, 4, dtype=torch.float16)
        router_logits = torch.randn(2, 4)
        topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = torch.ones_like(hidden_states)
        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(moe_comm_type=moe_comm_type, moe_comm_method=moe_comm_method),
        )
        select_experts_mock = MagicMock(return_value=(topk_weights, topk_ids))
        monkeypatch.setattr(fused_moe_module, "select_experts", select_experts_mock)

        result = method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=2,
            router_logits=router_logits,
            renormalize=True,
            num_experts=4,
            apply_router_weight_on_input=True,
            activation="gelu",
            pertoken_scale=torch.ones(2),
            mc2_mask=torch.tensor([True, False]),
        )

        torch.testing.assert_close(result, torch.ones_like(hidden_states))
        select_experts_mock.assert_called_once()
        fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        assert fused_input.hidden_states is hidden_states
        torch.testing.assert_close(fused_input.topk_weights, topk_weights.to(hidden_states.dtype))
        assert torch.equal(fused_input.topk_ids, topk_ids)
        assert fused_input.weights.w1_bias is layer.w13_bias
        assert fused_input.weights.w2_bias is layer.w2_bias
        assert fused_input.routing.apply_router_weight_on_input
        assert fused_input.activation == "gelu"
        if moe_comm_type == MoECommType.FUSED_MC2:
            assert fused_input.weights.w1[0] is layer.w13_weight
            assert fused_input.weights.w2[0] is layer.w2_weight
            assert isinstance(fused_input.weights.w1_scale, list)
            assert isinstance(fused_input.weights.w2_scale, list)
        else:
            assert fused_input.weights.w1 is layer.w13_weight
            assert fused_input.weights.w2 is layer.w2_weight
            assert fused_input.weights.w1_scale is None
            assert fused_input.weights.w2_scale is None

    def test_apply_adds_zero_expert_result_and_force_balances(self, monkeypatch):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=False)
        method.dynamic_eplb = True
        layer = self._build_layer(has_bias=False, zero_expert_num=1)
        hidden_states = torch.randn(2, 4)
        topk_weights = torch.ones(2, 2)
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        zero_hidden = torch.full_like(hidden_states, 3.0)
        routed_hidden = torch.full_like(hidden_states, 5.0)
        expected = routed_hidden + zero_hidden
        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = routed_hidden

        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(moe_comm_type=MoECommType.MC2, moe_comm_method=moe_comm_method),
        )
        monkeypatch.setattr(fused_moe_module, "select_experts", MagicMock(return_value=(topk_weights, topk_ids)))
        zero_experts_mock = MagicMock(return_value=(topk_ids, topk_weights, zero_hidden))
        monkeypatch.setattr(fused_moe_module, "zero_experts_compute", zero_experts_mock)
        monkeypatch.setattr(torch, "rand", MagicMock(return_value=torch.tensor([[0.2, 0.1], [0.4, 0.3]])))

        result = method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=2,
            router_logits=torch.randn(2, 2),
            renormalize=False,
            num_experts=2,
            enable_force_load_balance=True,
        )

        torch.testing.assert_close(result, expected)
        zero_experts_mock.assert_called_once()
        fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        assert fused_input.dynamic_eplb
        assert fused_input.weights.w1_bias is None
        assert fused_input.weights.w2_bias is None


class TestAscendMoERunner:
    @pytest.mark.parametrize(
        "moe_comm_type, expected",
        [
            (MoECommType.ALLTOALL, True),
            (MoECommType.MC2, True),
            (MoECommType.FUSED_MC2, True),
            (MoECommType.ALLGATHER, False),
        ],
    )
    def test_runner_reduction_properties(self, monkeypatch, moe_comm_type, expected):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        monkeypatch.setattr(fused_moe_module, "_EXTRA_CTX", SimpleNamespace(moe_comm_type=moe_comm_type))

        assert runner.use_dp_chunking is False
        if hasattr(type(runner), "_fused_output_is_reduced"):
            assert runner._fused_output_is_reduced is expected
        if hasattr(runner, "_maybe_reduce_shared_expert_output"):
            assert runner._maybe_reduce_shared_expert_output("shared") == "shared"

    @pytest.mark.parametrize("has_shared_experts", [False, True])
    def test_forward_impl_delegates_to_layer(self, monkeypatch, has_shared_experts):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        shared_experts = MagicMock() if has_shared_experts else None
        shared_experts_owner = next(
            (cls for cls in type(runner).__mro__ if "shared_experts" in cls.__dict__),
            AscendMoERunner,
        )
        monkeypatch.setattr(shared_experts_owner, "shared_experts", property(lambda _: shared_experts), raising=False)
        layer = MagicMock()
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 3)
        layer.forward_impl.return_value = "routed"
        layer.shared_forward_impl.return_value = ("shared", "routed")

        result = runner.forward_impl(layer, hidden_states, router_logits, None)

        if has_shared_experts:
            assert result == ("shared", "routed")
            layer.shared_forward_impl.assert_called_once_with(hidden_states, router_logits)
            layer.forward_impl.assert_not_called()
        else:
            assert result == "routed"
            layer.forward_impl.assert_called_once_with(hidden_states, router_logits)
            layer.shared_forward_impl.assert_not_called()


class TestAscendFusedMoE:
    def _build_layer(self):
        layer = AscendFusedMoE.__new__(AscendFusedMoE)
        layer.quant_method = MagicMock()
        layer.ensure_moe_quant_config_init = MagicMock()
        layer.runner = MagicMock()
        layer.moe_load = torch.zeros(2, dtype=torch.int64)
        layer.multi_stage = False
        layer.log2phy = torch.tensor([1, 0])
        return layer

    def test_simple_helpers(self, monkeypatch):
        layer = self._build_layer()
        layer.quant_method.quant_method = SimpleNamespace(quant_type=QuantType.W8A8)
        layer.update_expert_map(torch.tensor([0, -1]))
        assert torch.equal(layer._expert_map, torch.tensor([0, -1]))
        assert torch.equal(layer.get_log2phy_map(), torch.tensor([1, 0]))
        assert layer._get_quant_type() == QuantType.W8A8

        layer.clear_moe_load()
        assert torch.equal(layer.moe_load, torch.zeros_like(layer.moe_load))
        layer.multi_stage = True
        layer.load_counter = torch.tensor(4)
        layer.clear_moe_load()
        assert layer.load_counter.item() == 0

        maybe_all_reduce = MagicMock(return_value="reduced")
        monkeypatch.setattr(
            fused_moe_module.torch.ops,
            "vllm",
            SimpleNamespace(maybe_all_reduce_tensor_model_parallel=maybe_all_reduce),
            raising=False,
        )
        assert layer.maybe_all_reduce_tensor_model_parallel(torch.ones(1)) == "reduced"

    def test_forward_delegates_to_runner(self):
        layer = self._build_layer()
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 3)
        layer.runner.forward.return_value = "forwarded"

        assert layer.forward(hidden_states, router_logits) == "forwarded"
        layer.ensure_moe_quant_config_init.assert_called_once()
        layer.runner.forward.assert_called_once_with(hidden_states, router_logits)

    @pytest.mark.parametrize("return_with_event", [True, False])
    def test_forward_impl_prepare_apply_finalize(self, monkeypatch, return_with_event):
        layer = self._build_layer()
        layer.enable_npugraph_ex_static_kernel = True
        layer.multistream_overlap_gate = False
        layer.enable_shared_expert_dp = False
        layer.quant_type = QuantType.NONE
        layer.top_k = 2
        layer.renormalize = True
        layer.use_grouped_topk = False
        layer.moe_config = SimpleNamespace(num_experts=4)
        layer._expert_map = None
        layer.topk_group = None
        layer.num_expert_group = None
        layer.custom_routing_function = None
        layer.scoring_func = "softmax"
        layer._original_routed_scaling_factor = 1.0
        layer.routed_scaling_factor = 1.0
        layer.e_score_correction_bias = None
        layer.activation = "silu"
        layer.apply_router_weight_on_input = False
        layer.global_redundant_expert_num = 0
        layer.dynamic_eplb = True
        layer.reduce_results = True
        forward_context = SimpleNamespace(moe_layer_index=5, all_moe_layers=[0, 1])
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 4)
        prepared_hidden = hidden_states + 1
        prepared_logits = router_logits + 1
        prepare_output = MoEPrepareOutput(
            hidden_states=prepared_hidden,
            router_logits=prepared_logits,
            mc2_mask=torch.tensor([True, False]),
            padded_hidden_states_shape=torch.Size([4, 4]),
            pertoken_scale=torch.ones(2),
        )
        moe_comm_method = MagicMock()
        moe_comm_method.prepare.return_value = prepare_output
        moe_comm_method.finalize.side_effect = lambda hidden_states, **_: (hidden_states + 2)
        before_dispatch_evt = MagicMock()
        before_combine_evt = MagicMock()
        layer.quant_method.apply.return_value = FusedExpertsResult(
            routed_out=torch.ones_like(hidden_states),
            before_dispatch_evt=before_dispatch_evt,
            before_combine_evt=before_combine_evt,
            expert_tokens=torch.tensor([2, 5]),
            group_list_type=0,
        )
        monkeypatch.setattr(fused_moe_module, "get_forward_context", MagicMock(return_value=forward_context))
        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(in_profile_run=True, moe_comm_method=moe_comm_method, flash_comm_v1_enabled=True),
        )

        result = layer.forward_impl(hidden_states, router_logits, return_with_event=return_with_event)

        assert forward_context.moe_layer_index == 1
        moe_comm_method.prepare.assert_called_once_with(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=True,
            enable_shared_expert_dp=False,
            quant_type=QuantType.NONE,
        )
        apply_kwargs = layer.quant_method.apply.call_args.kwargs
        assert apply_kwargs["x"] is prepared_hidden
        assert apply_kwargs["router_logits"] is prepared_logits
        assert apply_kwargs["num_experts"] == 4
        assert apply_kwargs["enable_force_load_balance"] is True
        assert torch.equal(apply_kwargs["mc2_mask"], prepare_output.mc2_mask)
        torch.testing.assert_close(layer.moe_load, torch.tensor([2, 3]))
        if return_with_event:
            assert result.routed_out.shape == hidden_states.shape
            assert result.before_dispatch_evt is before_dispatch_evt
            assert result.before_combine_evt is before_combine_evt
        else:
            torch.testing.assert_close(result, torch.ones_like(hidden_states) + 2)

    def test_forward_impl_dynamic_eplb_multi_stage(self, monkeypatch):
        layer = self._build_layer()
        layer.enable_npugraph_ex_static_kernel = False
        layer.multistream_overlap_gate = False
        layer.enable_shared_expert_dp = False
        layer.quant_type = QuantType.NONE
        layer.top_k = 1
        layer.renormalize = False
        layer.use_grouped_topk = False
        layer.moe_config = SimpleNamespace(num_experts=2)
        layer._expert_map = None
        layer.topk_group = None
        layer.num_expert_group = None
        layer.custom_routing_function = None
        layer.scoring_func = "softmax"
        layer._original_routed_scaling_factor = 1.0
        layer.routed_scaling_factor = 1.0
        layer.e_score_correction_bias = None
        layer.activation = "silu"
        layer.apply_router_weight_on_input = False
        layer.global_redundant_expert_num = 0
        layer.dynamic_eplb = True
        layer.multi_stage = True
        layer.moe_load = torch.zeros((2, 2), dtype=torch.int32)
        layer.load_counter = torch.tensor([1], dtype=torch.int64)
        layer.num_iter = 2
        layer.reduce_results = False
        moe_comm_method = MagicMock()
        moe_comm_method.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.ones(2, 4),
            router_logits=torch.ones(2, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        moe_comm_method.finalize.side_effect = lambda hidden_states, **_: (hidden_states)
        layer.quant_method.apply.return_value = FusedExpertsResult(
            routed_out=torch.ones(2, 4),
            expert_tokens=torch.tensor([4, 6]),
            group_list_type=1,
        )
        monkeypatch.setattr(fused_moe_module, "get_forward_context", MagicMock(return_value=SimpleNamespace()))
        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(in_profile_run=False, moe_comm_method=moe_comm_method, flash_comm_v1_enabled=False),
        )

        layer.forward_impl(torch.zeros(2, 4), torch.zeros(2, 2))

        assert torch.equal(layer.moe_load[1], torch.tensor([4, 6], dtype=torch.int32))
        assert layer.load_counter.item() == 2


class TestAscendFusedMoESharedExperts:
    def test_properties_and_forward_delegate(self, monkeypatch):
        layer = AscendFusedMoE.__new__(AscendFusedMoE)
        if not hasattr(type(layer), "gate"):
            pytest.skip("Current AscendFusedMoE does not expose gate property")
        layer._gate = MagicMock()
        layer.use_overlapped = True
        assert layer.gate is layer._gate
        layer.use_overlapped = False
        assert layer.gate is None
        assert layer.is_internal_router is False
        assert layer.use_dp_chunking is False

        monkeypatch.setattr(fused_moe_module.AscendFusedMoE, "forward", MagicMock(return_value="routed"))
        layer._shared_experts = None
        assert layer.forward(torch.ones(1, 2), torch.ones(1, 2)) == "routed"

        fused_moe_module.AscendFusedMoE.forward.return_value = "forwarded"
        layer._shared_experts = MagicMock()
        assert layer.forward(torch.ones(1, 2), torch.ones(1, 2)) == "forwarded"

    def test_shared_experts_split_with_expert_gate(self):
        layer = AscendFusedMoE.__new__(AscendFusedMoE)
        if not hasattr(layer, "_shared_experts_part1"):
            pytest.skip("Current AscendFusedMoE does not split shared experts")
        hidden_states = torch.tensor([[1.0, -1.0]])
        gate_up = torch.tensor([[2.0, -2.0]])
        down_out = torch.tensor([[3.0, 4.0]])
        gate_out = torch.tensor([[0.0, 2.0]])
        shared_experts = MagicMock()
        shared_experts.gate_up_proj.return_value = (gate_up, None)
        shared_experts.act_fn.side_effect = lambda tensor: tensor + 1
        shared_experts.down_proj.return_value = (down_out, None)
        shared_experts.expert_gate.return_value = (gate_out, None)
        layer._shared_experts = shared_experts

        part1_out = layer._shared_experts_part1(hidden_states)
        part2_out = layer._shared_experts_part2(hidden_states, part1_out)

        torch.testing.assert_close(part1_out, gate_up)
        torch.testing.assert_close(part2_out, F.sigmoid(gate_out) * down_out)

    @pytest.mark.parametrize("has_shared_experts", [False, True])
    def test_shared_forward_impl_routes_shared_output(self, monkeypatch, has_shared_experts):
        layer = AscendFusedMoE.__new__(AscendFusedMoE)
        if not hasattr(layer, "shared_forward_impl"):
            pytest.skip("Current AscendFusedMoE has no shared_forward_impl")
        layer.shared_multistream_overlap_gate = False
        layer._shared_experts = MagicMock() if has_shared_experts else None
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 3)
        fused_result = fused_moe_module.FusedMoEResult(
            routed_out=torch.ones(2, 4),
            before_dispatch_evt=MagicMock(),
            before_combine_evt=MagicMock(),
        )
        monkeypatch.setattr(
            fused_moe_module.torch.npu,
            "current_stream",
            MagicMock(return_value=MagicMock(record_event=MagicMock(return_value=MagicMock()))),
        )
        monkeypatch.setattr(fused_moe_module.AscendFusedMoE, "forward_impl", MagicMock(return_value=fused_result))
        layer._forward_shared_experts = MagicMock(return_value="shared_out")

        result = layer.shared_forward_impl(hidden_states, router_logits)

        if has_shared_experts:
            assert result == ("shared_out", fused_result.routed_out)
            layer._forward_shared_experts.assert_called_once()
        else:
            torch.testing.assert_close(result, fused_result.routed_out)
