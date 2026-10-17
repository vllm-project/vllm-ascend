# SPDX-License-Identifier: Apache-2.0
"""Dense sequence parallelism for the Qwen3.5 family (FlashComm SP switch)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

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
        layer.linear_attn = SimpleNamespace(out_proj=SimpleNamespace(reduce_results=True))
    else:
        layer.self_attn = SimpleNamespace(o_proj=SimpleNamespace(reduce_results=True))
    layer.mlp = mlp if mlp is not None else SimpleNamespace(down_proj=SimpleNamespace(reduce_results=True))
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
