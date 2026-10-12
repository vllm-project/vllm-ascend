from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_states.encoder_decoder import (
    EncoderDecoderAttnMetadata,
    EncoderDecoderModelState,
)

import vllm_ascend.worker.v2.model_states as model_states
import vllm_ascend.worker.v2.model_states.encoder_decoder as encoder_decoder
from vllm_ascend.worker.v2.model_states.encoder_decoder import (
    AscendEncoderDecoderModelState,
)


class _CrossAttention:
    pass


class _EncoderDecoderModel:
    def modules(self):
        return [_CrossAttention()]


def test_cross_attention_model_selects_ascend_encoder_decoder_state(monkeypatch):
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=False))
    model = _EncoderDecoderModel()
    encoder_cache = object()
    device = torch.device("cpu")
    state_cls = MagicMock()

    monkeypatch.setattr(model_states, "CrossAttention", _CrossAttention)
    monkeypatch.setattr(
        model_states,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda _: True),
    )
    monkeypatch.setattr(encoder_decoder, "AscendEncoderDecoderModelState", state_cls)

    state = model_states.init_asecnd_model_state(
        vllm_config,
        model,
        encoder_cache,
        device,
    )

    assert state is state_cls.return_value
    state_cls.assert_called_once_with(vllm_config, model, encoder_cache, device)


def test_ascend_encoder_decoder_state_reuses_upstream_input_lifecycle():
    assert issubclass(AscendEncoderDecoderModelState, EncoderDecoderModelState)

    state = AscendEncoderDecoderModelState.__new__(AscendEncoderDecoderModelState)
    encoder_output = torch.tensor([1.0])
    state.encoder_outputs = []
    state.encoder_runner = SimpleNamespace(
        prepare_mm_inputs=MagicMock(return_value=(None, {"input_features": object()})),
        timed_encoder_operation=MagicMock(return_value=nullcontext()),
        execute_mm_encoder=MagicMock(return_value=[encoder_output]),
    )
    input_batch = SimpleNamespace(req_ids=["request-1"])
    scheduled_encoder_inputs = {"request-1": [0]}

    if hasattr(state, "prepare_inputs_embeds"):
        result = state.prepare_inputs_embeds(
            scheduled_encoder_inputs,
            input_batch,
            req_states=SimpleNamespace(),
        )
    else:
        result = state.get_mm_embeddings(
            scheduled_encoder_inputs,
            input_batch,
            req_states=SimpleNamespace(),
        )

    assert result is None
    assert state.prepare_inputs(input_batch, SimpleNamespace()) == {"encoder_outputs": [encoder_output]}
    assert state.encoder_outputs == []
    state.encoder_runner.execute_mm_encoder.assert_called_once()


@pytest.mark.parametrize(
    ("cudagraph_mode", "expected_num_reqs", "expected_num_input_tokens"),
    [
        (CUDAGraphMode.NONE, 2, 5),
        (CUDAGraphMode.FULL, 4, 8),
    ],
)
def test_ascend_encoder_decoder_state_builds_ascend_attention_metadata(
    monkeypatch,
    cudagraph_mode,
    expected_num_reqs,
    expected_num_input_tokens,
):
    state = AscendEncoderDecoderModelState.__new__(AscendEncoderDecoderModelState)
    state.max_model_len = 32
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=2,
        cp_kv_cache_interleave_size=1,
    )
    state.vllm_config = SimpleNamespace(parallel_config=parallel_config)
    encoder_seq_lens = {
        0: (
            torch.tensor([7, 11], dtype=torch.int32),
            np.array([7, 11], dtype=np.int32),
        )
    }
    state._get_encoder_seq_lens = MagicMock(return_value=encoder_seq_lens)
    expected_metadata = {"layer.0": object()}
    build_attn_metadata = MagicMock(return_value=expected_metadata)
    monkeypatch.setattr(encoder_decoder, "build_attn_metadata", build_attn_metadata)

    input_batch = SimpleNamespace(
        req_ids=["request-1", "request-2"],
        num_reqs=2,
        num_reqs_after_padding=4,
        num_tokens=5,
        num_tokens_after_padding=8,
        query_start_loc_np=np.array([0, 2, 5, 5, 5], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2, 5, 5, 5], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
        seq_lens=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
        seq_lens_np=np.array([2, 3, 0, 0], dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
        is_prefilling_np=np.array([True, True, False, False]),
        dcp_local_seq_lens=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
        positions=torch.arange(8, dtype=torch.int32),
        attn_state=None,
    )
    attn_groups = [[SimpleNamespace()]]
    block_tables = (torch.zeros((4, 1), dtype=torch.int32),)
    slot_mappings = torch.zeros((1, 8), dtype=torch.int32)
    kv_cache_config = SimpleNamespace()

    result = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=cudagraph_mode,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
    )

    assert result is expected_metadata
    assert state.attn_metadata is expected_metadata
    state._get_encoder_seq_lens.assert_called_once_with(
        input_batch.req_ids,
        attn_groups,
        False,
        expected_num_reqs,
    )
    kwargs = build_attn_metadata.call_args.kwargs
    assert kwargs["num_reqs"] == expected_num_reqs
    assert kwargs["num_actual_reqs"] == 2
    assert kwargs["num_tokens"] == expected_num_input_tokens
    assert kwargs["num_actual_tokens"] == 5
    assert kwargs["num_input_tokens"] == expected_num_input_tokens
    assert kwargs["positions"] is input_batch.positions
    assert kwargs["seq_lens_cpu_upper_bound"] is input_batch.seq_lens_cpu_upper_bound
    assert kwargs["dcp_local_seq_lens"] is input_batch.dcp_local_seq_lens
    assert kwargs["parallel_config"] is parallel_config
    model_metadata = kwargs["model_specific_attn_metadata"]
    assert isinstance(model_metadata, EncoderDecoderAttnMetadata)
    assert model_metadata.encoder_seq_lens is encoder_seq_lens


def test_ascend_encoder_decoder_state_rejects_dbo():
    state = AscendEncoderDecoderModelState.__new__(AscendEncoderDecoderModelState)

    with pytest.raises(AssertionError, match="DBO is not supported"):
        state.prepare_attn(
            input_batch=SimpleNamespace(),
            cudagraph_mode=CUDAGraphMode.NONE,
            block_tables=(),
            slot_mappings=torch.tensor([]),
            attn_groups=[],
            kv_cache_config=SimpleNamespace(),
            ubatch_idx=1,
        )
