from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CompilationConfig, VllmConfig

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker import encoder_acl_graph
from vllm_ascend.worker.encoder_acl_graph import (
    EncoderAclGraphManager,
    get_encoder_forward_context,
    get_encoder_graph_params,
    maybe_compute_actual_seq_lengths,
    set_encoder_graph_params,
    update_encoder_graph_params,
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
    params.handles[2048] = [1]
    params.events[2048] = [MagicMock()]
    params.attn_params[2048] = [packed]
    params.workspaces[2048] = MagicMock()

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
    ):
        update_encoder_graph_params(MagicMock(), 2048)

    assert captured["actual_seq_lengths"] == [4, 8]


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
    model.get_encoder_cudagraph_config.return_value = MagicMock(
        modalities=["image"],
        buffer_keys=["cu_seqlens"],
        out_hidden_size=64,
        enable_dual_path_graph=False,
        padding_logics={},
        max_frames_per_video=1,
    )
    model.get_encoder_cudagraph_budget_range.return_value = (64, 2048)
    return EncoderAclGraphManager(vllm_config, "npu", "bfloat16", model), model


def test_capture_graph_params():
    mgr, _ = _make_manager()
    mgr.token_budgets = [2048]

    with patch("vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager.capture", return_value=None):
        mgr.capture()

    params = get_encoder_graph_params()
    assert params is not None
    assert 2048 in params.events


def test_capture_budget_graph_npu():
    mgr, model = _make_manager()
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
            side_effect=lambda tensors: tensors,
        ),
    ):
        mgr._capture_budget_graph(2048)

    if vllm_version_is("0.23.0"):
        graph_meta = mgr.budget_graphs[2048]
    else:
        graph_meta = mgr._get_graph_set("default")[2048]
    assert graph_meta.graph is fake_graph
    assert graph_meta.input_buffers is capture_values
