# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.worker.v2.model_states import default as state_module


class TokenModel(torch.nn.Module):
    def __init__(self, requires_ids=True):
        super().__init__()
        self.requires_raw_input_tokens = requires_ids

    def forward(self, input_ids=None, positions=None, **kwargs):
        return input_ids


@pytest.fixture
def make_state(monkeypatch):
    def initialize(self, config, model, encoder_cache, device):
        self.vllm_config = config
        self.model_config = config.model_config
        self.model = model
        self.max_num_reqs = 4
        self.max_num_tokens = 32

    monkeypatch.setattr(DefaultModelState, "__init__", initialize)
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    monkeypatch.setattr(
        state_module,
        "get_ascend_config",
        lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
    )
    monkeypatch.setattr(state_module, "build_attn_metadata", lambda **kwargs: {})

    def create(requires_ids=True, pp_size=2):
        model = TokenModel(requires_ids)
        config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=128),
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, pipeline_parallel_size=pp_size),
        )
        state = state_module.AscendModelState(config, model, None, torch.device("cpu"))
        return state, model

    return create


def batch(raw_ids, *, is_dummy=False):
    num_reqs = 4 if is_dummy else 1
    num_tokens = 24 if is_dummy else 6
    query = np.arange(num_reqs + 1, dtype=np.int32) * 6
    return SimpleNamespace(
        is_dummy=is_dummy,
        input_ids=raw_ids[:24],
        positions=torch.arange(24, dtype=torch.int64),
        num_tokens=num_tokens,
        num_tokens_after_padding=24,
        num_reqs=num_reqs,
        num_reqs_after_padding=4,
        req_ids=[str(index) for index in range(num_reqs)],
        query_start_loc=torch.from_numpy(query),
        query_start_loc_np=query,
        seq_lens=torch.full((4,), 6, dtype=torch.int32),
        seq_lens_np=np.full(4, 6, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.full((4,), 6, dtype=torch.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=bool),
        num_scheduled_tokens=np.full(num_reqs, 6, dtype=np.int32),
        dcp_local_seq_lens=None,
        attn_state=None,
    )


@pytest.mark.parametrize("is_dummy", [False, True])
def test_nonfirst_pp_real_and_profile_forwards_restore_native_padded_ids(make_state, is_dummy):
    state, model = make_state()
    native_ids = torch.arange(32, dtype=torch.int32) + 100
    current_batch = batch(native_ids, is_dummy=is_dummy)
    inputs = state.prepare_inputs(current_batch, None)
    # Native MRV2 clears IDs on non-first PP ranks after state preparation.
    inputs.update(input_ids=None, positions=current_batch.positions)

    restored = model(**inputs)

    torch.testing.assert_close(restored, native_ids[:24])
    assert restored.shape == current_batch.positions.shape
    assert restored.data_ptr() == native_ids.data_ptr()
    assert restored.dtype == native_ids.dtype


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_full_capture_binds_dummy_batch_after_prepare_dummy_inputs(make_state, mode):
    state, model = make_state()
    state.prepare_inputs(batch(torch.arange(32, dtype=torch.int32)), None)
    native_ids = torch.arange(32, dtype=torch.int32) + 200
    capture_batch = batch(native_ids, is_dummy=True)
    # The graph manager builds model kwargs before its dummy attention batch.
    inputs = state.prepare_dummy_inputs(4, 24)
    inputs.update(input_ids=None, positions=capture_batch.positions)
    state.prepare_attn(capture_batch, mode, (), torch.empty(0), [], None, for_capture=True)

    captured_ids = model(**inputs)

    assert captured_ids.data_ptr() == native_ids.data_ptr()
    torch.testing.assert_close(captured_ids, native_ids[:24])
    # Replay uses the same native input-buffer address after its values change.
    native_ids.add_(300)
    state.prepare_inputs(batch(native_ids), None)
    torch.testing.assert_close(captured_ids, native_ids[:24])
    assert model(**inputs).data_ptr() == captured_ids.data_ptr()


def test_non_none_forward_ids_are_preserved(make_state):
    state, model = make_state()
    state.prepare_inputs(batch(torch.arange(32, dtype=torch.int32)), None)
    explicit = torch.tensor([91, 92, 93], dtype=torch.int64)

    assert model(input_ids=explicit, positions=torch.arange(3)) is explicit
    assert model(explicit, torch.arange(3)) is explicit


def test_positional_none_ids_use_the_prepared_native_view(make_state):
    state, model = make_state()
    native_ids = torch.arange(32, dtype=torch.int32)
    state.prepare_inputs(batch(native_ids), None)

    output = model(None, torch.arange(24))

    assert output.data_ptr() == native_ids.data_ptr()
    torch.testing.assert_close(output, native_ids[:24])


@pytest.mark.parametrize("requires_ids,pp_size", [(False, 2), (True, 1)])
def test_models_without_pp_raw_id_requirements_keep_none_inputs(make_state, requires_ids, pp_size):
    state, model = make_state(requires_ids=requires_ids, pp_size=pp_size)

    assert state.prepare_inputs(SimpleNamespace(), None) == {}
    assert not model._forward_pre_hooks
    assert model(input_ids=None, positions=torch.arange(3)) is None


def test_missing_positions_and_empty_unprepared_forwards_remain_safe(make_state):
    _, model = make_state()

    assert model(input_ids=None, positions=None) is None
    assert model(input_ids=None, positions=torch.empty(0, dtype=torch.int64)) is None


def test_unprepared_or_truncated_raw_ids_fail_without_inventing_tokens(make_state):
    state, model = make_state()
    with pytest.raises(RuntimeError, match="not prepared"):
        model(input_ids=None, positions=torch.arange(3))
    state.prepare_inputs(batch(torch.arange(32, dtype=torch.int32)), None)
    with pytest.raises(ValueError, match="padded model positions"):
        model(input_ids=None, positions=torch.arange(25))
