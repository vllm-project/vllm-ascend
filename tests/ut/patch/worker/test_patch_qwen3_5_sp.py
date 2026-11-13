# SPDX-License-Identifier: Apache-2.0
"""Dense sequence parallelism for the Qwen3.5 family (FlashComm SP switch)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CompilationMode
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ops import linear_op
from vllm_ascend.patch.worker import patch_qwen3_5
from vllm_ascend.utils import enable_dense_sp, enable_sp


def _make_vllm_config(
    *,
    tensor_parallel_size: int = 2,
    pipeline_parallel_size: int = 1,
    model_type: str = "qwen3_5_text",
    enable_flashcomm1: bool | None = True,
    use_sequence_parallel_moe: bool = False,
):
    additional_config = {} if enable_flashcomm1 is None else {"enable_flashcomm1": enable_flashcomm1}
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        additional_config=additional_config,
    )


class TestEnableDenseSp:
    def test_enabled_by_flashcomm_switch(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        assert enable_dense_sp(_make_vllm_config())

    def test_enabled_by_env_var(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "1")
        assert enable_dense_sp(_make_vllm_config(enable_flashcomm1=None))

    @pytest.mark.parametrize("env_value", [None, "0", "false"])
    def test_disabled_without_switch(self, monkeypatch, env_value):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        if env_value is not None:
            monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", env_value)
        assert not enable_dense_sp(_make_vllm_config(enable_flashcomm1=False))

    def test_falsey_string_switch_stays_disabled(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        config = _make_vllm_config()
        config.additional_config = {"enable_flashcomm1": "false"}
        assert not enable_dense_sp(config)

    def test_disabled_for_tensor_parallel_size_one(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        assert not enable_dense_sp(_make_vllm_config(tensor_parallel_size=1))

    def test_disabled_for_pipeline_parallel(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        assert not enable_dense_sp(_make_vllm_config(pipeline_parallel_size=2))

    @pytest.mark.parametrize("model_type", ["qwen3_5_moe_text", "qwen3_next", "llama_text"])
    def test_disabled_for_other_model_families(self, monkeypatch, model_type):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        assert not enable_dense_sp(_make_vllm_config(model_type=model_type))

    def test_disabled_without_model_config(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        config = _make_vllm_config()
        config.model_config = None
        assert not enable_dense_sp(config)

    def test_enable_sp_covers_dense_models(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        assert enable_sp(_make_vllm_config(use_sequence_parallel_moe=False))

    def test_enable_sp_stays_off_without_dense_sp(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        assert not enable_sp(_make_vllm_config(enable_flashcomm1=False))

    def test_enable_sp_covers_sp_moe_models(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        config = _make_vllm_config(
            model_type="qwen3_5_moe_text",
            enable_flashcomm1=False,
            use_sequence_parallel_moe=True,
        )
        assert enable_sp(config)
        assert not enable_dense_sp(config)


def _make_layer(layer_type: str = "full_attention", mlp=None, upstream_sp: bool = False):
    layer = object.__new__(patch_qwen3_5.AscendQwen3_5DecoderLayer)
    layer.layer_type = layer_type
    layer.use_attn_reduce_scatter_for_moe = upstream_sp
    if layer_type == "linear_attention":
        layer.linear_attn = SimpleNamespace(out_proj=SimpleNamespace(reduce_results=True, custom_op=None))
    else:
        layer.self_attn = SimpleNamespace(
            qkv_proj=SimpleNamespace(custom_op=None),
            o_proj=SimpleNamespace(reduce_results=True, custom_op=None),
        )
    layer.mlp = (
        mlp
        if mlp is not None
        else SimpleNamespace(
            gate_up_proj=SimpleNamespace(custom_op=None),
            down_proj=SimpleNamespace(reduce_results=True, custom_op=None),
        )
    )
    return layer


def _run_layer_init(layer, vllm_config, layer_type, dense_sp: bool):
    def _fake_original_init(_layer, _vllm_config, _layer_type, prefix=""):
        del _layer, _vllm_config, _layer_type, prefix

    with (
        patch.object(patch_qwen3_5, "_QWEN3_5_LAYER_ORIGINAL_INIT", _fake_original_init),
        patch.object(patch_qwen3_5, "enable_dense_sp", return_value=dense_sp),
    ):
        patch_qwen3_5._qwen3_5_decoder_layer_init_with_dense_sp(layer, vllm_config, layer_type)


class TestDenseSpLayerInit:
    def test_full_attention_layer_flips_to_sequence_parallel(self):
        layer = _make_layer("full_attention")
        _run_layer_init(layer, _make_vllm_config(), "full_attention", dense_sp=True)
        assert layer.use_attn_reduce_scatter_for_moe is True
        assert layer.self_attn.o_proj.reduce_results is False
        assert layer.mlp.down_proj.reduce_results is False

    def test_linear_attention_layer_flips_out_proj(self):
        layer = _make_layer("linear_attention")
        _run_layer_init(layer, _make_vllm_config(), "linear_attention", dense_sp=True)
        assert layer.use_attn_reduce_scatter_for_moe is True
        assert layer.linear_attn.out_proj.reduce_results is False
        assert layer.mlp.down_proj.reduce_results is False

    def test_noop_without_dense_sp(self):
        layer = _make_layer("full_attention")
        _run_layer_init(layer, _make_vllm_config(), "full_attention", dense_sp=False)
        assert layer.use_attn_reduce_scatter_for_moe is False
        assert layer.self_attn.o_proj.reduce_results is True
        assert layer.mlp.down_proj.reduce_results is True

    def test_noop_when_upstream_sp_moe_already_active(self):
        layer = _make_layer("full_attention", upstream_sp=True)
        layer.self_attn.o_proj.reduce_results = False
        layer.mlp.down_proj.reduce_results = True  # MoE block keeps its own layout
        _run_layer_init(layer, _make_vllm_config(), "full_attention", dense_sp=True)
        assert layer.use_attn_reduce_scatter_for_moe is True
        assert layer.self_attn.o_proj.reduce_results is False
        assert layer.mlp.down_proj.reduce_results is True


class _ForwardLayer(SimpleNamespace):
    """Test double that avoids nn.Module metaclass issues on CI."""

    def forward(self, hidden_states, residual, positions=None, **kwargs):
        return patch_qwen3_5.AscendQwen3_5DecoderLayer.forward(self, hidden_states, residual, positions, **kwargs)


class _CallableProjection:
    """Callable projection double: its forward dispatches to the installed
    custom op, mirroring AscendColumn/RowParallelLinear.forward."""

    def __init__(self, op):
        self.custom_op = op

    def __call__(self, input_):
        return self.custom_op.apply(input_)


def _make_forward_layer(layer_type: str = "full_attention", use_sp: bool = True, mlp=None):
    def _input_norm(hidden, residual=None):
        return (hidden, hidden) if residual is None else (hidden, residual)

    layer = _ForwardLayer(
        layer_type=layer_type,
        use_attn_reduce_scatter_for_moe=use_sp,
        layer_scale=False,
        input_layernorm=_input_norm,
        post_attention_layernorm=lambda hidden, residual: (hidden, residual),
    )
    if layer_type == "linear_attention":
        layer.linear_attn = MagicMock(return_value=torch.full((4, 2), 3.0))
    else:
        layer.self_attn = MagicMock(return_value=torch.full((4, 2), 3.0))
    if mlp is not None:
        layer.mlp = mlp
    else:
        # Default MLP must be callable through the dense-MLP helper:
        # gate_up_proj / down_proj return (tensor, None) tuples.
        layer.mlp = SimpleNamespace(
            gate_up_proj=lambda x: (torch.full((4, 2), 1.0), None),
            act_fn=lambda x: x,
            down_proj=lambda x: (torch.full((4, 2), 3.0), None),
        )
    return layer


class TestSpDecoderLayerForward:
    def test_full_attention_gathers_before_and_scatters_after(self):
        layer = _make_forward_layer("full_attention", use_sp=True)
        hidden = torch.zeros(2, 2)
        residual = torch.zeros(2, 2)
        gathered = torch.full((4, 2), 1.0)

        with (
            patch.object(patch_qwen3_5, "sp_all_gather", return_value=gathered) as all_gather,
            patch.object(patch_qwen3_5, "sp_reduce_scatter", side_effect=lambda x: x[:2]) as reduce_scatter,
        ):
            output, out_residual = layer.forward(
                hidden_states=hidden,
                residual=residual,
                positions=torch.zeros(4, dtype=torch.long),
            )

        # sp_all_gather runs twice per SP layer: once for attention and once
        # for the dense MLP gather. Both start from the 2-row shard.
        assert all_gather.call_count == 2
        # Attention ran on the gathered full-token layout.
        attn_input = layer.self_attn.call_args.kwargs["hidden_states"]
        assert attn_input.shape[0] == 4
        assert attn_input.data_ptr() == gathered.data_ptr()
        # sp_reduce_scatter also runs twice: once after attention, once after MLP.
        assert reduce_scatter.call_count == 2
        # The final output and residual both have the shard's row count.
        assert output.shape[0] == 2
        assert out_residual.shape[0] == 2

    def test_linear_attention_gathers_before_and_scatters_after(self):
        layer = _make_forward_layer("linear_attention", use_sp=True)
        gathered = torch.full((4, 2), 1.0)

        with (
            patch.object(patch_qwen3_5, "sp_all_gather", return_value=gathered),
            patch.object(patch_qwen3_5, "sp_reduce_scatter", side_effect=lambda x: x[:2]),
        ):
            layer.forward(
                hidden_states=torch.zeros(2, 2),
                residual=torch.zeros(2, 2),
                positions=torch.zeros(4, dtype=torch.long),
            )

        assert layer.linear_attn.call_args.kwargs["hidden_states"].data_ptr() == gathered.data_ptr()

    def test_dense_mlp_runs_on_the_shard(self):
        layer = _make_forward_layer("full_attention", use_sp=True)
        wrapped_output = torch.full((2, 2), 7.0)

        with (
            patch.object(patch_qwen3_5, "sp_all_gather", return_value=torch.full((4, 2), 1.0)),
            patch.object(patch_qwen3_5, "sp_reduce_scatter", side_effect=lambda x: x[:2]),
            patch.object(
                patch_qwen3_5,
                "_sequence_parallel_dense_mlp_forward",
                return_value=wrapped_output,
            ) as dense_mlp,
        ):
            output, _ = layer.forward(
                hidden_states=torch.zeros(2, 2),
                residual=torch.zeros(2, 2),
                positions=torch.zeros(4, dtype=torch.long),
            )

        dense_mlp.assert_called_once()
        assert dense_mlp.call_args.args[0] is layer.mlp
        assert dense_mlp.call_args.args[1].shape[0] == 2
        assert output is wrapped_output

    def test_moe_mlp_receives_already_sequence_parallel(self):
        moe_calls = {}

        class _FakeMoEBlock:
            def __call__(self, hidden_states, already_sequence_parallel=False, **kwargs):
                moe_calls["already_sequence_parallel"] = already_sequence_parallel
                return torch.full((2, 2), 5.0)

        moe = _FakeMoEBlock()
        layer = _make_forward_layer("full_attention", use_sp=True, mlp=moe)

        with (
            patch.object(patch_qwen3_5, "sp_all_gather", return_value=torch.full((4, 2), 1.0)),
            patch.object(patch_qwen3_5, "sp_reduce_scatter", side_effect=lambda x: x[:2]),
            patch.object(patch_qwen3_5, "Qwen3NextSparseMoeBlock", _FakeMoEBlock),
        ):
            layer.forward(
                hidden_states=torch.zeros(2, 2),
                residual=torch.zeros(2, 2),
                positions=torch.zeros(4, dtype=torch.long),
            )

        assert moe_calls["already_sequence_parallel"] is True

    def test_non_sp_forward_keeps_upstream_layout(self):
        mlp_calls = []

        def _mlp(x):
            mlp_calls.append(x)
            return torch.full((4, 2), 5.0)

        layer = _make_forward_layer("full_attention", use_sp=False, mlp=_mlp)

        with (
            patch.object(patch_qwen3_5, "sp_all_gather") as all_gather,
            patch.object(patch_qwen3_5, "sp_reduce_scatter") as reduce_scatter,
        ):
            output, out_residual = layer.forward(
                hidden_states=torch.zeros(4, 2),
                residual=torch.zeros(4, 2),
                positions=torch.zeros(4, dtype=torch.long),
            )

        all_gather.assert_not_called()
        reduce_scatter.assert_not_called()
        assert len(mlp_calls) == 1
        assert output.shape[0] == 4
        assert out_residual.shape[0] == 4


class TestSequenceParallelDenseMlp:
    def test_gathers_around_tp_sharded_weights_in_order(self):
        mlp = SimpleNamespace(
            gate_up_proj=MagicMock(return_value=(torch.full((4, 2), 1.0), None)),
            act_fn=MagicMock(side_effect=lambda x: x),
            down_proj=MagicMock(return_value=(torch.full((4, 2), 3.0), None)),
        )
        shard = torch.zeros(2, 2)
        gathered = torch.full((4, 2), 1.0)
        order = []

        def _all_gather(x):
            order.append("all_gather")
            assert x is shard
            return gathered

        def _reduce_scatter(x):
            order.append("reduce_scatter")
            assert x is not None and x.shape[0] == 4
            return x[:2]

        def _gate_up(x):
            order.append("gate_up_proj")
            return (torch.full((4, 2), 1.0), None)

        def _act(x):
            order.append("act_fn")
            return x

        def _down(x):
            order.append("down_proj")
            return (torch.full((4, 2), 3.0), None)

        mlp.gate_up_proj.side_effect = _gate_up
        mlp.act_fn.side_effect = _act
        mlp.down_proj.side_effect = _down

        with (
            patch.object(patch_qwen3_5, "sp_all_gather", side_effect=_all_gather),
            patch.object(patch_qwen3_5, "sp_reduce_scatter", side_effect=_reduce_scatter),
        ):
            output = patch_qwen3_5._sequence_parallel_dense_mlp_forward(mlp, shard)

        assert order == ["all_gather", "gate_up_proj", "act_fn", "down_proj", "reduce_scatter"]
        assert output.shape[0] == 2


class TestPatchInstallation:
    def test_decoder_layer_forward_and_init_are_installed(self):
        assert patch_qwen3_5.Qwen3_5DecoderLayer.forward is patch_qwen3_5.AscendQwen3_5DecoderLayer.forward
        assert patch_qwen3_5.Qwen3_5DecoderLayer.__init__ is patch_qwen3_5._qwen3_5_decoder_layer_init_with_dense_sp

    def test_installed_init_calls_through_to_upstream(self):
        calls = []

        def _fake_original_init(layer, vllm_config, layer_type, prefix=""):
            calls.append((vllm_config, layer_type, prefix))

        layer = _make_layer("full_attention")
        config = _make_vllm_config()
        with (
            patch.object(patch_qwen3_5, "_QWEN3_5_LAYER_ORIGINAL_INIT", _fake_original_init),
            patch.object(patch_qwen3_5, "enable_dense_sp", return_value=False),
        ):
            patch_qwen3_5.Qwen3_5DecoderLayer.__init__(layer, config, "full_attention", prefix="model.layers.0")

        assert calls == [(config, "full_attention", "model.layers.0")]


class _FakeDeviceOperator:
    """Single-process stand-in for the fused matmul-collective kernels.

    The fakes delegate to the module-level ``sp_all_gather`` /
    ``sp_reduce_scatter`` that each test patches, so the fused path and the
    fallback path compute from the same staged collectives and stay directly
    comparable.
    """

    calls: list = []

    @staticmethod
    def npu_all_gather_base_mm(x1, x2, hcom, world_size, *, bias=None, **kwargs):
        _FakeDeviceOperator.calls.append(("all_gather_mm", x1, x2, world_size, bias))
        output = patch_qwen3_5.sp_all_gather(x1) @ x2
        return output if bias is None else output + bias

    @staticmethod
    def npu_mm_reduce_scatter_base(x1, x2, hcom, world_size, *, reduce_op="sum", **kwargs):
        _FakeDeviceOperator.calls.append(("mm_reduce_scatter", x1, x2, world_size))
        return patch_qwen3_5.sp_reduce_scatter(x1 @ x2)


class _FakeQuantMethod:
    """Plain GEMM stand-in for ``quant_method.apply``."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        output = x @ self.weight.t()
        return output if bias is None else output + bias


def _make_fake_tp_group(world_size: int = 2) -> MagicMock:
    group = MagicMock()
    group.world_size = world_size
    group.rank_in_group = 0
    group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "hcom-0"
    return group


def _make_column_proj(weight: torch.Tensor, *, bias: torch.Tensor | None = None, quant_method=None) -> SimpleNamespace:
    return SimpleNamespace(
        prefix="model.layers.0.self_attn.qkv_proj",
        bias=bias,
        skip_bias_add=False,
        return_bias=True,
        quant_method=quant_method if quant_method is not None else _FakeQuantMethod(weight),
        weight=weight,
        gather_output=False,
        custom_op=None,
    )


def _make_row_proj(weight: torch.Tensor, *, bias: torch.Tensor | None = None, quant_method=None) -> SimpleNamespace:
    return SimpleNamespace(
        prefix="model.layers.0.self_attn.o_proj",
        bias=bias,
        skip_bias_add=False,
        return_bias=True,
        quant_method=quant_method if quant_method is not None else _FakeQuantMethod(weight),
        weight=weight,
        input_is_parallel=True,
        reduce_results=False,
        input_size_per_partition=weight.shape[1],
        custom_op=None,
    )


class TestDenseSpFusedColumnOp:
    @pytest.fixture(autouse=True)
    def _fused_env(self, monkeypatch: pytest.MonkeyPatch):
        _FakeDeviceOperator.calls = []
        monkeypatch.setattr(patch_qwen3_5, "get_tp_group", lambda: _make_fake_tp_group(2))
        monkeypatch.setattr(patch_qwen3_5, "DeviceOperator", _FakeDeviceOperator)
        monkeypatch.setattr(patch_qwen3_5, "_DENSE_SP_FUSED_DTYPES", (torch.float64,))

    def test_fuses_all_gather_into_gemm(self, monkeypatch):
        gen = torch.Generator().manual_seed(0)
        weight = torch.randn(5, 4, generator=gen, dtype=torch.float64)
        gathered = torch.randn(6, 4, generator=gen, dtype=torch.float64)
        shard = torch.randn(3, 4, generator=gen, dtype=torch.float64)
        monkeypatch.setattr(patch_qwen3_5, "sp_all_gather", lambda x: gathered)

        op = patch_qwen3_5.DenseSpAllGatherMmOp(_make_column_proj(weight))
        output, output_bias = op.apply(shard)

        assert [call[0] for call in _FakeDeviceOperator.calls] == ["all_gather_mm"]
        _, x1, x2, world_size, bias = _FakeDeviceOperator.calls[0]
        assert x1 is shard
        assert torch.equal(x2, weight.t())
        assert world_size == 2
        assert bias is None
        # The fused op returns the padded full-token rows; the caller trims.
        assert output.shape[0] == gathered.shape[0]
        torch.testing.assert_close(output, gathered @ weight.t())
        assert output_bias is None

    def test_bias_is_passed_into_the_fused_gemm(self, monkeypatch):
        gen = torch.Generator().manual_seed(1)
        weight = torch.randn(5, 4, generator=gen, dtype=torch.float64)
        bias = torch.randn(5, generator=gen, dtype=torch.float64)
        gathered = torch.randn(6, 4, generator=gen, dtype=torch.float64)
        monkeypatch.setattr(patch_qwen3_5, "sp_all_gather", lambda x: gathered)

        op = patch_qwen3_5.DenseSpAllGatherMmOp(_make_column_proj(weight, bias=bias))
        output, output_bias = op.apply(torch.randn(3, 4, generator=gen, dtype=torch.float64))

        torch.testing.assert_close(output, gathered @ weight.t() + bias)
        assert output_bias is None

    def test_ineligible_dtype_falls_back_to_gather_plus_gemm(self, monkeypatch):
        gen = torch.Generator().manual_seed(2)
        # float32 is not in _DENSE_SP_FUSED_DTYPES (only float64 in this test
        # suite), so the structural guard rejects the fusion.
        weight = torch.randn(5, 4, generator=gen, dtype=torch.float32)
        gathered = torch.randn(6, 4, generator=gen, dtype=torch.float32)
        monkeypatch.setattr(patch_qwen3_5, "sp_all_gather", lambda x: gathered)

        op = patch_qwen3_5.DenseSpAllGatherMmOp(_make_column_proj(weight))
        output, _ = op.apply(torch.randn(3, 4, generator=gen, dtype=torch.float32))

        assert _FakeDeviceOperator.calls == []
        torch.testing.assert_close(output, gathered @ weight.t())


class TestDenseSpFusedRowOp:
    @pytest.fixture(autouse=True)
    def _fused_env(self, monkeypatch: pytest.MonkeyPatch):
        _FakeDeviceOperator.calls = []
        fake_group = _make_fake_tp_group(2)
        monkeypatch.setattr(patch_qwen3_5, "get_tp_group", lambda: fake_group)
        # CustomLinearOp.tp_rank resolves the group from linear_op's module.
        monkeypatch.setattr(linear_op, "get_tp_group", lambda: fake_group)
        monkeypatch.setattr(patch_qwen3_5, "DeviceOperator", _FakeDeviceOperator)
        monkeypatch.setattr(patch_qwen3_5, "_DENSE_SP_FUSED_DTYPES", (torch.float64,))
        monkeypatch.setattr(patch_qwen3_5, "sp_reduce_scatter", lambda x: x[: x.shape[0] // 2])

    def test_zero_pads_then_fuses(self, monkeypatch):
        gen = torch.Generator().manual_seed(3)
        weight = torch.randn(4, 6, generator=gen, dtype=torch.float64)
        partial_input = torch.randn(5, 6, generator=gen, dtype=torch.float64)  # 5 % tp(2) != 0

        op = patch_qwen3_5.DenseSpMmReduceScatterOp(_make_row_proj(weight))
        output, output_bias = op.apply(partial_input)

        assert [call[0] for call in _FakeDeviceOperator.calls] == ["mm_reduce_scatter"]
        _, x1, x2, world_size = _FakeDeviceOperator.calls[0]
        assert x1.shape[0] == 6  # zero-padded to the TP multiple
        assert torch.equal(x1[5:], torch.zeros(1, 6, dtype=torch.float64))
        assert torch.equal(x2, weight.t())
        assert world_size == 2
        assert output.shape[0] == 3
        # The zero-padded row contributes nothing: rows 0..2 of the padded
        # GEMM equal rows 0..2 of the unpadded one.
        torch.testing.assert_close(output, (partial_input @ weight.t())[:3])
        assert output_bias is None

    def test_aligned_input_fused_matches_fallback(self, monkeypatch):
        gen = torch.Generator().manual_seed(4)
        weight = torch.randn(4, 6, generator=gen, dtype=torch.float64)
        x = torch.randn(6, 6, generator=gen, dtype=torch.float64)

        fused_op = patch_qwen3_5.DenseSpMmReduceScatterOp(_make_row_proj(weight))
        fused, _ = fused_op.apply(x)

        # Force the fallback path by swapping the dtype allow-list to exclude
        # float64, so the structural guard rejects the same input.
        monkeypatch.setattr(patch_qwen3_5, "_DENSE_SP_FUSED_DTYPES", (torch.float16,))
        fallback, _ = fused_op.apply(x)

        torch.testing.assert_close(fused, fallback)
        # Fused once, fallback once.
        assert [call[0] for call in _FakeDeviceOperator.calls] == ["mm_reduce_scatter"]

    def test_bias_lands_once_on_the_local_shard(self, monkeypatch):
        gen = torch.Generator().manual_seed(5)
        weight = torch.randn(4, 6, generator=gen, dtype=torch.float64)
        bias = torch.randn(4, generator=gen, dtype=torch.float64)
        x = torch.randn(6, 6, generator=gen, dtype=torch.float64)

        op = patch_qwen3_5.DenseSpMmReduceScatterOp(_make_row_proj(weight, bias=bias))
        output, output_bias = op.apply(x)

        torch.testing.assert_close(output, (x @ weight.t())[:3] + bias)
        assert output_bias is None

    def test_ineligible_dtype_falls_back_to_partial_gemm_plus_scatter(self, monkeypatch):
        gen = torch.Generator().manual_seed(6)
        # float32 is not in the fused dtype allow-list, so the structural
        # guard rejects the fusion and the op falls back to the unfused path.
        weight = torch.randn(4, 6, generator=gen, dtype=torch.float32)
        x = torch.randn(6, 6, generator=gen, dtype=torch.float32)

        op = patch_qwen3_5.DenseSpMmReduceScatterOp(_make_row_proj(weight))
        output, _ = op.apply(x)

        assert _FakeDeviceOperator.calls == []
        torch.testing.assert_close(output, (x @ weight.t())[:3])


class TestInstallDenseSpFusedOps:
    @pytest.fixture(autouse=True)
    def _install_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(patch_qwen3_5, "get_tp_group", lambda: _make_fake_tp_group(2))
        monkeypatch.setattr(patch_qwen3_5.torch_npu, "npu_all_gather_base_mm", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(patch_qwen3_5.torch_npu, "npu_mm_reduce_scatter_base", lambda *a, **k: None, raising=False)

    def _eligible_proj(self, **overrides):
        weight = torch.randn(4, 6, dtype=torch.bfloat16)
        kwargs = {"quant_method": UnquantizedLinearMethod()}
        kwargs.update(overrides)
        return _make_row_proj(weight, **kwargs)

    def test_installs_fused_row_op_on_eligible_projection(self):
        proj = self._eligible_proj()
        assert patch_qwen3_5._install_dense_sp_fused_op(proj, patch_qwen3_5.DenseSpMmReduceScatterOp) is True
        assert isinstance(proj.custom_op, patch_qwen3_5.DenseSpMmReduceScatterOp)

    def test_skips_for_unsupported_tp_size(self, monkeypatch):
        monkeypatch.setattr(patch_qwen3_5, "get_tp_group", lambda: _make_fake_tp_group(16))
        proj = self._eligible_proj()
        assert patch_qwen3_5._install_dense_sp_fused_op(proj, patch_qwen3_5.DenseSpMmReduceScatterOp) is False
        assert proj.custom_op is None

    def test_skips_for_quantized_projection(self):
        proj = self._eligible_proj(quant_method=_FakeQuantMethod(torch.randn(4, 6, dtype=torch.bfloat16)))
        assert patch_qwen3_5._install_dense_sp_fused_op(proj, patch_qwen3_5.DenseSpMmReduceScatterOp) is False
        assert proj.custom_op is None

    def test_skips_when_a_custom_op_is_already_installed(self):
        proj = self._eligible_proj()
        proj.custom_op = object()
        assert patch_qwen3_5._install_dense_sp_fused_op(proj, patch_qwen3_5.DenseSpMmReduceScatterOp) is False

    def test_skips_column_projection_with_gather_output(self):
        weight = torch.randn(4, 6, dtype=torch.bfloat16)
        proj = _make_column_proj(weight, quant_method=UnquantizedLinearMethod())
        proj.gather_output = True
        assert patch_qwen3_5._install_dense_sp_fused_op(proj, patch_qwen3_5.DenseSpAllGatherMmOp) is False

    def test_skips_when_the_fused_kernel_is_unavailable(self, monkeypatch):
        monkeypatch.delattr(patch_qwen3_5.torch_npu, "npu_mm_reduce_scatter_base", raising=False)
        proj = self._eligible_proj()
        assert patch_qwen3_5._install_dense_sp_fused_op(proj, patch_qwen3_5.DenseSpMmReduceScatterOp) is False


class TestDenseSpLayerInitFusedOps:
    @pytest.fixture(autouse=True)
    def _init_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(patch_qwen3_5, "get_tp_group", lambda: _make_fake_tp_group(2))
        monkeypatch.setattr(patch_qwen3_5, "enable_dense_sp", lambda vllm_config=None: True)
        monkeypatch.setattr(patch_qwen3_5.torch_npu, "npu_all_gather_base_mm", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(patch_qwen3_5.torch_npu, "npu_mm_reduce_scatter_base", lambda *a, **k: None, raising=False)

    def _make_layer_with_fusable_projections(self, layer_type: str):
        layer = _make_layer(layer_type)
        quant_method = UnquantizedLinearMethod()
        if layer_type == "linear_attention":
            layer.linear_attn.out_proj = _make_row_proj(
                torch.randn(4, 6, dtype=torch.bfloat16), quant_method=quant_method
            )
        else:
            layer.self_attn.qkv_proj = _make_column_proj(
                torch.randn(6, 4, dtype=torch.bfloat16), quant_method=quant_method
            )
            layer.self_attn.o_proj = _make_row_proj(torch.randn(4, 6, dtype=torch.bfloat16), quant_method=quant_method)
        layer.mlp.gate_up_proj = _make_column_proj(torch.randn(6, 4, dtype=torch.bfloat16), quant_method=quant_method)
        layer.mlp.down_proj = _make_row_proj(torch.randn(4, 6, dtype=torch.bfloat16), quant_method=quant_method)
        return layer

    def test_full_attention_installs_all_four_fused_ops(self):
        layer = self._make_layer_with_fusable_projections("full_attention")
        _run_layer_init(layer, _make_vllm_config(), "full_attention", dense_sp=True)

        assert isinstance(layer.self_attn.qkv_proj.custom_op, patch_qwen3_5.DenseSpAllGatherMmOp)
        assert isinstance(layer.self_attn.o_proj.custom_op, patch_qwen3_5.DenseSpMmReduceScatterOp)
        assert isinstance(layer.mlp.gate_up_proj.custom_op, patch_qwen3_5.DenseSpAllGatherMmOp)
        assert isinstance(layer.mlp.down_proj.custom_op, patch_qwen3_5.DenseSpMmReduceScatterOp)

    def test_gdn_installs_only_the_output_projection(self):
        layer = self._make_layer_with_fusable_projections("linear_attention")
        _run_layer_init(layer, _make_vllm_config(), "linear_attention", dense_sp=True)

        assert isinstance(layer.linear_attn.out_proj.custom_op, patch_qwen3_5.DenseSpMmReduceScatterOp)
        # The GDN in-projections keep the decoder-level gather.
        assert not hasattr(layer.linear_attn, "in_proj_qkvz")

    def test_no_fused_ops_without_dense_sp(self):
        layer = self._make_layer_with_fusable_projections("full_attention")
        _run_layer_init(layer, _make_vllm_config(), "full_attention", dense_sp=False)

        assert layer.self_attn.qkv_proj.custom_op is None
        assert layer.self_attn.o_proj.custom_op is None
        assert layer.mlp.gate_up_proj.custom_op is None
        assert layer.mlp.down_proj.custom_op is None

    def test_half_installed_mlp_pair_is_unwound(self):
        layer = self._make_layer_with_fusable_projections("full_attention")
        # Make down_proj ineligible (an already-installed custom op) while
        # gate_up_proj stays eligible.
        sentinel = object()
        layer.mlp.down_proj.custom_op = sentinel
        _run_layer_init(layer, _make_vllm_config(), "full_attention", dense_sp=True)

        assert layer.mlp.gate_up_proj.custom_op is None
        assert layer.mlp.down_proj.custom_op is sentinel


class TestFusedDecoderFlow:
    """With fused boundary projections the decoder-level collectives vanish."""

    @pytest.fixture(autouse=True)
    def _fused_env(self, monkeypatch: pytest.MonkeyPatch):
        _FakeDeviceOperator.calls = []
        fake_group = _make_fake_tp_group(2)
        monkeypatch.setattr(patch_qwen3_5, "get_tp_group", lambda: fake_group)
        monkeypatch.setattr(linear_op, "get_tp_group", lambda: fake_group)
        monkeypatch.setattr(patch_qwen3_5, "DeviceOperator", _FakeDeviceOperator)
        monkeypatch.setattr(patch_qwen3_5, "_DENSE_SP_FUSED_DTYPES", (torch.float64,))

    def _fused_column(self, out_features: int, in_features: int) -> "_CallableProjection":
        proj = _make_column_proj(torch.randn(out_features, in_features, dtype=torch.float64))
        return _CallableProjection(patch_qwen3_5.DenseSpAllGatherMmOp(proj))

    def _fused_row(self, out_features: int, in_features: int) -> "_CallableProjection":
        proj = _make_row_proj(torch.randn(out_features, in_features, dtype=torch.float64))
        return _CallableProjection(patch_qwen3_5.DenseSpMmReduceScatterOp(proj))

    def test_all_fused_layer_skips_decoder_collectives(self):
        gather_rows = []
        scatter_rows = []

        layer = _make_forward_layer("full_attention", use_sp=True)
        # self_attn is a MagicMock, so the decoder only inspects its qkv_proj
        # / o_proj custom ops; the MLP projections run for real below. The
        # float64 return keeps the fused dtype guard satisfied.
        layer.self_attn = MagicMock(return_value=torch.zeros(4, 2, dtype=torch.float64))
        layer.self_attn.qkv_proj = self._fused_column(6, 4)
        layer.self_attn.o_proj = self._fused_row(2, 6)
        layer.mlp.gate_up_proj = self._fused_column(6, 2)
        layer.mlp.down_proj = self._fused_row(2, 6)

        def _gather(x):
            gather_rows.append(x.shape[0])
            return x.repeat(2, 1)

        def _scatter(x):
            scatter_rows.append(x.shape[0])
            return x[: x.shape[0] // 2]

        with (
            patch.object(patch_qwen3_5, "sp_all_gather", _gather),
            patch.object(patch_qwen3_5, "sp_reduce_scatter", _scatter),
        ):
            output, residual = layer.forward(
                hidden_states=torch.zeros(4, 2, dtype=torch.float64),
                residual=torch.zeros(4, 2, dtype=torch.float64),
                positions=torch.zeros(4, dtype=torch.long),
            )

        # The unfused layout would run one decoder-level gather plus one
        # MLP-helper gather (and the same for the scatters); with the fused
        # ops owning the boundary collectives only the kernels' staged
        # collectives run: the MLP's gather over the 4-row shard and its
        # scatter over the 8 padded rows.
        assert gather_rows == [4]
        assert scatter_rows == [8]
        assert [call[0] for call in _FakeDeviceOperator.calls] == ["all_gather_mm", "mm_reduce_scatter"]
        assert output.shape == (4, 2)
        assert residual.shape == (4, 2)

    def test_attention_trims_padded_rows_behind_fused_qkv(self):
        class _Rotary:
            def __call__(self, positions, q, k):
                return q, k

        attention = SimpleNamespace(
            qkv_proj=self._fused_column(8, 8),
            config=SimpleNamespace(model_type="qwen3_5_text", rms_norm_eps=1e-6),
            attn_output_gate=False,
            q_size=4,
            kv_size=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=2,
            q_norm=MagicMock(side_effect=lambda x: x),
            k_norm=MagicMock(side_effect=lambda x: x),
            rotary_emb=_Rotary(),
            attn=MagicMock(return_value=torch.zeros(5, 4, dtype=torch.float64)),
            o_proj=MagicMock(return_value=(torch.zeros(5, 4, dtype=torch.float64), None)),
        )

        # 5 real tokens, TP=2: the fused gather produces 6 padded rows and
        # the patched attention must trim back to 5 before RoPE.
        with (
            patch.object(patch_qwen3_5, "sp_all_gather", lambda x: x.repeat(2, 1)),
            patch.object(patch_qwen3_5, "sp_reduce_scatter", lambda x: x[: x.shape[0] // 2]),
        ):
            out = patch_qwen3_5.AscendQwen3NextAttention.forward(
                attention,
                positions=torch.zeros(5, dtype=torch.long),
                hidden_states=torch.randn(3, 8, dtype=torch.float64),
            )

        q, k, v = attention.attn.call_args.args
        assert q.shape[0] == 5
        assert k.shape[0] == 5
        assert v.shape[0] == 5
        assert out.shape[0] == 5


class TestDenseSpGraphModePreserved:
    """Dense SP keeps the user's cudagraph_mode unchanged: the fused
    matmul-collective projections are opaque custom ops that compile, warmup,
    capture and replay on the same path, so FULL graphs are safe."""

    @staticmethod
    def _make_mode_config(
        *,
        cudagraph_mode,
        enforce_eager=False,
        enable_flashcomm1=True,
        quantization_config=None,
    ):
        text_config = SimpleNamespace(model_type="qwen3_5_text", quantization_config=quantization_config)
        return SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=cudagraph_mode,
                mode=CompilationMode.VLLM_COMPILE,
                splitting_ops=[],
            ),
            model_config=SimpleNamespace(
                enforce_eager=enforce_eager,
                is_encoder_decoder=False,
                hf_quant_config=quantization_config,
                hf_config=text_config,
                hf_text_config=text_config,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
                use_sequence_parallel_moe=False,
            ),
            additional_config={} if enable_flashcomm1 is None else {"enable_flashcomm1": enable_flashcomm1},
            speculative_config=None,
        )

    @staticmethod
    def _run_update_compilation_modes(vllm_config):
        from vllm_ascend.platform import _update_compilation_modes

        ascend_config = SimpleNamespace(
            ascend_compilation_config={},
            xlite_graph_config=SimpleNamespace(enabled=False),
        )
        _update_compilation_modes(vllm_config, ascend_config)
        return vllm_config.compilation_config.cudagraph_mode

    def test_full_keeps_full(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        mode = self._run_update_compilation_modes(self._make_mode_config(cudagraph_mode=CUDAGraphMode.FULL))
        assert mode == CUDAGraphMode.FULL

    def test_full_and_piecewise_keeps_full_and_piecewise(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        mode = self._run_update_compilation_modes(
            self._make_mode_config(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)
        )
        assert mode == CUDAGraphMode.FULL_AND_PIECEWISE

    def test_no_flashcomm_keeps_full(self, monkeypatch):
        monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
        mode = self._run_update_compilation_modes(
            self._make_mode_config(cudagraph_mode=CUDAGraphMode.FULL, enable_flashcomm1=False)
        )
        assert mode == CUDAGraphMode.FULL
