"""Exercise shared state staging with real CPU tensor copies and reordered slots."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_copy_spec
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from vllm_ascend.worker.v2.model_states import mamba_hybrid as module


@pytest.fixture
def make_state(monkeypatch):
    monkeypatch.setattr(MambaHybridModelState, "initialize_kv_cache", lambda *args: None)
    monkeypatch.setattr(MambaHybridModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(module, "_allocate_int8_cache_tensor", lambda size, *_: torch.zeros(size, dtype=torch.uint8))
    spec = SimpleNamespace(mamba_type="test", block_size=4)
    monkeypatch.setattr(module.mamba_utils, "_get_mamba_spec_for_layer", lambda *_: spec)
    connector = Mock()
    monkeypatch.setattr(module, "get_kv_transfer_group", lambda: connector)

    def create(*, profiling=False, reuse=True, align=True, pp=1):
        state = module.AscendMambaHybridModelState.__new__(module.AscendMambaHybridModelState)
        state.max_num_reqs = 3
        state.device = torch.device("cpu")
        state._align_mode = align
        state.recoverssm = None
        state._layerwise_is_profiling = profiling
        state._mamba_group_ids = [0]
        state._mamba_spec = spec
        state._mamba_state_copy_funcs = {"test": (get_conv_copy_spec, object())}
        state._layerwise_mamba_copy = None
        state._mamba_ctx = SimpleNamespace(
            state_base_addrs=torch.zeros(8, dtype=torch.int64),
            state_block_strides=torch.zeros(8, dtype=torch.int64),
            state_dim_row_stride=torch.zeros(8, dtype=torch.int64),
            state_group_indices=torch.zeros(8, dtype=torch.int32),
            block_table_ptrs=torch.zeros(1, dtype=torch.int64),
            block_table_stride_req=6,
            num_groups=1,
            is_initialized=True,
        )
        names = [f"model.layers.{i}.attn" for i in range(4)]

        def caches():
            return (torch.zeros(12, 2, 3), torch.zeros(12, 2, 2))

        shared = caches()
        original = {names[0]: shared, names[1]: shared, names[2]: caches(), names[3]: caches()}
        layout = SimpleNamespace(
            has_layer_reuse=reuse,
            prefetch_layer_map={1: 0},
            layer_cache_specs={
                i: SimpleNamespace(main=SimpleNamespace(layer_name=name)) for i, name in enumerate(names)
            },
        )
        config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=names)],
            layerwise_reuse_layout=layout,
            layerwise_draft_layer_names=(names[3],),
        )
        state.vllm_config = SimpleNamespace(
            num_speculative_tokens=2,
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
            compilation_config=SimpleNamespace(
                static_forward_context={name: SimpleNamespace(kv_cache=value) for name, value in original.items()}
            ),
        )
        state.initialize_kv_cache(config, ())
        return state, names, original, connector, config

    return create


def prepare_batch(state):
    mapping = torch.tensor([2, 0], dtype=torch.int32)
    tables = (torch.tensor([[0, 4, 5, 6, 7, 8], [0, 7, 8, 9, 10, 11]], dtype=torch.int32),)
    state._mamba_batch = (("a", "b"), mapping, (2, 0), tables)
    state._mamba_state_idx_gpu = torch.ones(3, dtype=torch.int32)
    state._mamba_src_col_gpu = torch.ones(3, dtype=torch.int32)
    state._mamba_capture_pending = set(state._mamba_reused)
    state._mamba_accepted = False
    return mapping


def test_initialization_is_bounded_and_preserves_draft(make_state):
    state, names, original, connector, _ = make_state()
    assert set(state._mamba_staging) == set(names[:3])
    assert [cache.shape[0] for cache in state._mamba_staging[names[0]]] == [12, 12]
    assert [cache.shape[0] for cache in state._mamba_staging[names[2]]] == [3, 3]
    assert state.layerwise_mamba_backing.numel() == 3 * (4 * 2 + 1) * (6 + 4) * 4
    assert original[names[0]][0].data_ptr() == original[names[1]][0].data_ptr()
    assert state._mamba_staging[names[0]][0].data_ptr() != state._mamba_staging[names[1]][0].data_ptr()
    assert state._mamba_postprocess_ctx is not state._mamba_ctx
    connector.configure_mamba_state.assert_called_once_with(state)


@pytest.mark.parametrize("profiling,reuse", [(True, True), (False, False)])
def test_inactive_or_profiling_state_does_not_configure_connector(make_state, profiling, reuse):
    state, _, _, connector, _ = make_state(profiling=profiling, reuse=reuse)
    connector.configure_mamba_state.assert_not_called()
    assert bool(state._mamba_staging) == reuse


@pytest.mark.parametrize("kwargs", [{"align": False}, {"pp": 2}])
def test_unsupported_copy_contract_fails_before_staging(make_state, kwargs):
    with pytest.raises(NotImplementedError):
        make_state(**kwargs)


def test_capture_survives_alias_overwrite_and_restores_reordered_slots(make_state):
    state, names, original, _, _ = make_state()
    prepare_batch(state)
    for cache in original[names[0]]:
        cache.copy_(torch.arange(12).reshape(12, 1, 1) + torch.ones_like(cache) * 10)
    state.capture_mamba_layer(names[0])
    saved = [cache.clone() for cache in state._mamba_staging[names[0]]]
    for cache in original[names[1]]:
        cache.add_(100)
    state.capture_mamba_layer(names[1])
    assert not state._mamba_capture_pending
    assert all(torch.equal(a, b) for a, b in zip(saved, state._mamba_staging[names[0]]))
    state._mamba_valid[[2, 0]] = True
    state._copy_layer_candidates(names[0], restore=True)
    assert torch.all(original[names[0]][0][4] == 14)
    assert torch.all(original[names[0]][0][7] == 17)
    assert torch.all(original[names[0]][1][6] == 16)
    assert torch.all(original[names[0]][1][9] == 19)
    assert torch.count_nonzero(saved[1][5:9]) == 0  # Unused request slot remains untouched.


def test_acceptance_uses_staged_context_then_publishes_only_ready_checkpoint(make_state, monkeypatch):
    state, names, _, _, _ = make_state()
    mapping = prepare_batch(state)
    state._mamba_capture_pending.clear()

    def upstream_accept(self, indices, accepted, computed):
        assert self._mamba_ctx is self._mamba_postprocess_ctx
        assert indices is mapping
        for name in names[:2]:
            for cache in self._mamba_staging[name]:
                cache[2 * self._mamba_slots + 1].fill_(42)

    monkeypatch.setattr(MambaHybridModelState, "postprocess_state", upstream_accept)
    original_ctx = state._mamba_ctx
    state.postprocess_state(mapping, torch.tensor([1, 1]), torch.tensor([10, 0, 8]))
    assert state._mamba_ctx is original_ctx
    for checkpoint in state.layerwise_mamba_checkpoints[names[0]]:
        assert torch.all(checkpoint[2] == 42)
        assert torch.count_nonzero(checkpoint[0]) == 0
    assert state.get_mamba_checkpoints() == ({"a": 8, "b": 0}, {"a": 2, "b": 0})


def test_publication_and_acceptance_require_completed_layer_capture(make_state):
    state, _, _, _, _ = make_state()
    mapping = prepare_batch(state)
    with pytest.raises(RuntimeError, match="preceded acceptance"):
        state.get_mamba_checkpoints()
    with pytest.raises(RuntimeError, match="every shared layer"):
        state.postprocess_state(mapping, 1, torch.zeros(3, dtype=torch.int64))


def test_external_reload_and_request_reuse_invalidate_only_their_slot(make_state):
    state, _, _, _, _ = make_state()
    prepare_batch(state)
    state._mamba_valid.fill_(True)
    state._mamba_checkpoint_tokens.fill_(8)
    state.prepare_mamba_state_load(
        [
            SimpleNamespace(req_id="a", load_spec=SimpleNamespace(can_load=True, layerwise_previous_step=True)),
            SimpleNamespace(req_id="b", load_spec=SimpleNamespace(can_load=True, layerwise_previous_step=False)),
        ]
    )
    assert state._mamba_valid.tolist() == [False, True, True]
    state.add_request(2, None)
    assert state._mamba_valid.tolist() == [False, True, False]
    assert state._mamba_checkpoint_tokens.tolist() == [0, 8, 0]


def test_failed_save_prevents_reusing_staging_storage(make_state):
    state, _, _, _, config = make_state()
    prepare_batch(state)
    state.finish_mamba_save(failed=True)
    state._get_mamba_group_info = lambda _: ([0], state._mamba_spec)
    with pytest.raises(RuntimeError, match="still have readers or failed"):
        state.preprocess_state(SimpleNamespace(num_reqs=2), (), config, torch.zeros(3))
