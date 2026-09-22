from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CompilationConfig, VllmConfig
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig
from vllm.v1.worker.gpu.model_states import interface

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker import encoder_acl_graph
from vllm_ascend.worker.encoder_acl_graph import (
    EncoderAclGraphManager,
    get_encoder_forward_context,
    get_encoder_graph_params,
    maybe_compute_actual_seq_lengths,
    set_encoder_graph_params,
    update_encoder_graph_params,
    update_encoder_graph_workspace,
)


def _reset_encoder_acl_graph_state() -> None:
    encoder_acl_graph._encoder_graph_params = None
    encoder_acl_graph._reset_encoder_forward_context()


@pytest.fixture(autouse=True)
def _reset_state():
    _reset_encoder_acl_graph_state()
    yield
    _reset_encoder_acl_graph_state()


@pytest.mark.parametrize(
    "cu_seqlens, num_tokens, expected",
    [
        (torch.tensor([0, 4, 16], dtype=torch.int32), 8, [4, 16]),
    ],
)
def test_maybe_compute_actual_seq_lengths_eager(cu_seqlens, num_tokens, expected):
    actual_q, actual_kv = maybe_compute_actual_seq_lengths(
        cu_seqlens,
        num_tokens,
        num_tokens,
        cudagraph_mm_encoder=False,
    )
    assert actual_q == expected
    assert actual_kv == expected


def test_maybe_compute_actual_seq_lengths_eager_unequal_q_kv():
    """Molmo-style uniform cross-attention: scale KV endpoints by kv/q ratio."""
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int32)
    actual_q, actual_kv = maybe_compute_actual_seq_lengths(
        cu_seqlens,
        2,
        8,
        cudagraph_mm_encoder=False,
    )
    assert actual_q == [1, 2]
    assert actual_kv == [4, 8]


@pytest.mark.parametrize(
    "cu_seqlens, num_tokens, expected",
    [
        (torch.tensor([0, 4, 8], dtype=torch.int32), 8, [4, 8]),
        (torch.tensor([0, 4, 16], dtype=torch.int32), 8, [4, 8]),
    ],
)
def test_maybe_compute_actual_seq_lengths_graph(cu_seqlens, num_tokens, expected):
    actual_q, actual_kv = maybe_compute_actual_seq_lengths(
        cu_seqlens,
        num_tokens,
        num_tokens,
        cudagraph_mm_encoder=True,
    )
    assert actual_q == expected
    assert actual_kv == expected


def test_update_encoder_graph_params_cu_seqlens():
    set_encoder_graph_params([2048])
    params = get_encoder_graph_params()
    query = MagicMock()
    query.shape = [8, 4, 72]
    key = MagicMock()
    key.shape = [8, 4, 72]
    packed = (
        query,
        key,
        MagicMock(),
        None,
        None,
        128,
        4,
        4,
        0.125,
        MagicMock(),
        MagicMock(),
    )
    graph_key = ("default", 2048)
    params.handles[graph_key] = [1, 2]
    params.events[graph_key] = [MagicMock(), MagicMock()]
    params.attn_params[graph_key] = [packed, packed]
    params.workspaces[graph_key] = MagicMock()

    ctx = get_encoder_forward_context()
    ctx.cu_seqlens_cpu = torch.tensor([0, 4, 8], dtype=torch.int32)

    captured = {}

    def fake_out(**kwargs):
        captured["actual_seq_lengths"] = kwargs["actual_seq_lengths"]

    fake_fia = SimpleNamespace(out=fake_out)
    with (
        patch("vllm_ascend.worker.encoder_acl_graph.torch.npu.stream"),
        patch("vllm_ascend.worker.encoder_acl_graph.torch.npu.graph_task_update_begin"),
        patch("vllm_ascend.worker.encoder_acl_graph.torch.npu.graph_task_update_end"),
        patch(
            "vllm_ascend.worker.encoder_acl_graph.torch_npu.npu_fused_infer_attention_score",
            fake_fia,
        ),
        patch(
            "vllm_ascend.worker.encoder_acl_graph.maybe_compute_actual_seq_lengths",
            wraps=maybe_compute_actual_seq_lengths,
        ) as compute_lengths,
    ):
        update_encoder_graph_params(MagicMock(), 2048)

    assert captured["actual_seq_lengths"] == [4, 8]
    assert compute_lengths.call_count == 1


def _make_manager():
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = CompilationConfig()
    mm_config = MagicMock()
    mm_config.get_limit_per_prompt.return_value = 0
    mm_config.mm_encoder_tp_mode = "tensor"
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.multimodal_config = mm_config
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.tensor_parallel_size = 1

    model = MagicMock()
    model.get_encoder_cudagraph_config.return_value = EncoderCudaGraphConfig(
        modalities=["image"],
        buffer_keys=["cu_seqlens"],
        out_hidden_size=64,
        padding_logics={},
        max_frames_per_video=1,
    )
    model.get_encoder_cudagraph_budget_range.return_value = (64, 2048)
    return EncoderAclGraphManager(vllm_config, "npu", "bfloat16", model), model


def test_capture_graph_params():
    mgr, _ = _make_manager()
    mgr.token_budgets = [2048]
    mgr.path_token_budgets = {"default": [2048]}

    assert not mgr.is_captured()

    with patch("vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager.capture", return_value=None):
        mgr.capture()

    params = get_encoder_graph_params()
    assert params is not None
    assert ("default", 2048) in params.events
    assert mgr.is_captured()

    mgr.clear()
    assert not mgr.is_captured()
    assert get_encoder_graph_params() is None


def test_capture_failure_clears_graph_state():
    mgr, _ = _make_manager()
    with (
        patch(
            "vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager.capture",
            side_effect=RuntimeError("capture failed"),
        ),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        mgr.capture()

    assert not mgr.is_captured()
    assert get_encoder_graph_params() is None


def test_dual_path_graph_params_are_isolated():
    set_encoder_graph_params({"global": [128, 256], "local": [0, 128]})

    params = get_encoder_graph_params()
    assert params is not None
    assert set(params.handles) == {
        ("global", 128),
        ("global", 256),
        ("local", 128),
    }
    assert params.handles[("global", 128)] is not params.handles[("local", 128)]


def test_capture_axis_graph_params_are_isolated():
    set_encoder_graph_params({"default": [128]})
    key_a = ("default", 128, ((14, 14),))
    key_b = ("default", 128, ((28, 28),))
    encoder_acl_graph._ensure_graph_params(key_a)
    encoder_acl_graph._ensure_graph_params(key_b)

    params = get_encoder_graph_params()
    assert params is not None
    assert params.handles[key_a] is not params.handles[key_b]
    assert params.workspaces[key_a] is None
    assert params.workspaces[key_b] is None
    workspace_a = torch.empty(1)
    update_encoder_graph_workspace(128, workspace_a, axis_keys=((14, 14),))
    assert params.workspaces[key_a] is workspace_a
    assert params.workspaces[key_b] is None


def test_mrv2_model_state_uses_ascend_encoder_graph_manager():
    # Importing the MRV2 patch replaces the binding used inside ModelState,
    # rather than only the source encoder_cudagraph module.
    from vllm_ascend.patch.worker.patch_v2 import patch_model_state  # noqa: F401

    assert interface.EncoderCudaGraphManager is EncoderAclGraphManager


def test_capture_budget_graph_npu():
    mgr, model = _make_manager()
    set_encoder_graph_params({"default": [2048]})
    mgr.max_batch_size = 2
    mgr.max_frames_per_batch = 0
    capture_values = {"cu_seqlens": torch.zeros(3, dtype=torch.int32)}
    model.prepare_encoder_cudagraph_capture_inputs.return_value = MagicMock(
        values=capture_values,
    )
    model.encoder_cudagraph_forward.return_value = torch.zeros(2, 64)

    fake_graph = MagicMock()
    with (
        patch("vllm_ascend.worker.encoder_acl_graph.torch.npu.NPUGraph", return_value=fake_graph),
        patch("vllm_ascend.worker.encoder_acl_graph.torch.npu.graph"),
        patch(
            "vllm_ascend.worker.encoder_acl_graph.weak_ref_tensors",
            side_effect=lambda tensor: tensor,
        ) as weak_ref,
    ):
        mgr._capture_budget_graph(2048, **({} if vllm_version_is("0.29.0") else {"axis_keys": ()}))

    graph_meta = mgr._get_graph_set("default")[2048]
    assert graph_meta.graph is fake_graph
    assert graph_meta.input_buffers is capture_values
    assert isinstance(graph_meta.output_buffer, torch.Tensor)
    weak_ref.assert_called_once_with(model.encoder_cudagraph_forward.return_value)


@pytest.mark.skipif(
    not encoder_acl_graph._ENCODER_SUPPORTS_CAPTURE_AXES,
    reason="This vLLM version does not support encoder capture axes",
)
def test_capture_budget_graph_with_axis_keys():
    mgr, model = _make_manager()
    set_encoder_graph_params({"default": [128]})
    axis_keys = ((14, 14),)
    model.prepare_encoder_cudagraph_capture_inputs.return_value = MagicMock(
        values={"cu_seqlens": torch.zeros(3, dtype=torch.int32)},
    )
    model.encoder_cudagraph_forward.return_value = torch.zeros(2, 64)

    with (
        patch("vllm_ascend.worker.encoder_acl_graph.torch.npu.NPUGraph"),
        patch("vllm_ascend.worker.encoder_acl_graph.torch.npu.graph"),
        patch("vllm_ascend.worker.encoder_acl_graph.weak_ref_tensors", side_effect=lambda value: value),
    ):
        mgr._capture_budget_graph(128, axis_keys=axis_keys)

    key = ("default", 128, axis_keys)
    assert key in get_encoder_graph_params().handles
    assert mgr._get_graph_set("default")[(128, axis_keys)].axis_keys == axis_keys
    assert model.prepare_encoder_cudagraph_capture_inputs.call_args.args[-1] == axis_keys


def test_replay_selects_capture_axis_graph():
    mgr, model = _make_manager()
    axis_keys = ((14, 14),)
    graph_meta = MagicMock()
    graph_meta.input_buffers = {"cu_seqlens": torch.zeros(3, dtype=torch.int32)}
    graph_meta.output_buffer = torch.zeros(2, 64)
    mgr._get_graph_set("default")[(128, axis_keys)] = graph_meta
    mgr._get_item_specs = MagicMock(return_value=[MagicMock()])
    model.prepare_encoder_cudagraph_replay_buffers.return_value = MagicMock(
        values={"cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32)},
    )

    current_stream = MagicMock(name="current_stream")
    update_stream = MagicMock(name="update_stream")
    call_order = []
    update_stream.wait_stream.side_effect = lambda stream: call_order.append(("wait_stream", stream))
    graph_meta.graph.replay.side_effect = lambda: call_order.append(("replay", None))

    with (
        patch("vllm_ascend.worker.encoder_acl_graph.torch.npu.Stream", return_value=update_stream),
        patch(
            "vllm_ascend.worker.encoder_acl_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        patch("vllm_ascend.worker.encoder_acl_graph.update_encoder_graph_params") as update,
    ):
        result = mgr._run_budget_graph({}, 128, axis_keys=axis_keys)

    graph_meta.graph.replay.assert_called_once()
    update.assert_called_once_with(mgr.update_stream, 128, path="default", axis_keys=axis_keys)
    assert call_order == [("wait_stream", current_stream), ("replay", None)]
    assert result is graph_meta.output_buffer
