import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state
from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
    AscendMambaHybridModelState,
)


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        block_size=16,
        shapes=((2, 3), (2, 2)),
        dtypes=(torch.float16, torch.float32),
    )


def _kv_cache_config(
    spec: MambaSpec,
    *,
    num_blocks: int = 3,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * spec.page_size_bytes,
                shared_by=["linear_attn"],
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["linear_attn"],
                kv_cache_spec=spec,
            )
        ],
    )


def _group(spec: MambaSpec):
    return SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        layer_names=["linear_attn"],
    )


def test_mamba_model_state_inherits_upstream_state_management():
    assert issubclass(AscendMambaHybridModelState, MambaHybridModelState)
    # preprocess_state is overridden on Ascend to defer the align pre-copy
    # behind layerwise KV pool loads; postprocess stays upstream.
    assert AscendMambaHybridModelState.preprocess_state is not MambaHybridModelState.preprocess_state
    assert AscendMambaHybridModelState.postprocess_state is MambaHybridModelState.postprocess_state


def _make_defer_state(kv_cache_config):
    """Build an AscendMambaHybridModelState without running __init__."""
    state = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    state.model = SimpleNamespace(get_mamba_state_copy_func=lambda: (MagicMock(), MagicMock()))
    state._align_mode = True
    state._mamba_group_ids = [0]
    state._mamba_spec = _mamba_spec()
    state._mamba_state_idx_gpu = torch.zeros(4, dtype=torch.int32)
    state._mamba_src_col_gpu = torch.full((4,), -1, dtype=torch.int32)
    state._mamba_src_off_gpu = torch.zeros(4, dtype=torch.int32)
    state._mamba_ctx = None
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32)
    state._layerwise_mamba_copy = None
    state._layer_state_ranges = None
    state._layer_state_ranges = state._get_layer_state_ranges(kv_cache_config)
    return state


@patch("vllm_ascend.worker.v2.model_states.mamba_hybrid.preprocess_mamba_align_fused_kernel")
@patch("vllm_ascend.worker.v2.model_states.mamba_hybrid.get_kv_transfer_group")
@patch(
    "vllm_ascend.worker.v2.model_states.mamba_hybrid.has_kv_transfer_group",
    return_value=True,
)
def test_layerwise_connector_defers_mamba_pre_copy(mock_has_group, mock_get_group, mock_decision_kernel):
    connector = SimpleNamespace(
        prepare_mamba_state_copy=MagicMock(return_value=True),
        finish_mamba_state_copy=MagicMock(),
    )
    mock_get_group.return_value = connector
    spec = _mamba_spec()
    kv_cache_config = _kv_cache_config(spec)
    state = _make_defer_state(kv_cache_config)
    ctx = MagicMock()
    state._ensure_align_ctx = MagicMock(return_value=ctx)
    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.arange(2, dtype=torch.int32),
        query_start_loc=torch.zeros(3, dtype=torch.int32),
    )
    num_computed = torch.zeros(4, dtype=torch.int32)

    state.preprocess_state(input_batch, (), kv_cache_config, num_computed)

    # Decision kernel still runs; the bulk pre-copy launch is skipped.
    mock_decision_kernel.__getitem__.assert_called_once()
    ctx.run_fused_precopy.assert_not_called()
    connector.prepare_mamba_state_copy.assert_called_once_with(state)
    assert state._layerwise_mamba_copy is not None
    assert state._layerwise_mamba_copy.pending_layers == {"linear_attn"}

    # Next step's preprocess validates every deferred layer executed.
    with patch("vllm.v1.worker.mamba_utils.precopy_mamba_align_fused_kernel"):
        state.do_mamba_copy_for_layer("linear_attn")
    state.preprocess_state(input_batch, (), kv_cache_config, num_computed)
    connector.finish_mamba_state_copy.assert_called_once()


@patch("vllm.v1.worker.mamba_utils.precopy_mamba_align_fused_kernel")
def test_do_mamba_copy_for_layer_launches_one_sliced_kernel(mock_precopy_kernel):
    spec = _mamba_spec()
    kv_cache_config = _kv_cache_config(spec)
    state = _make_defer_state(kv_cache_config)
    ctx = MagicMock()
    state._layerwise_mamba_copy = SimpleNamespace(
        ctx=ctx,
        num_reqs=2,
        idx_mapping=torch.arange(2, dtype=torch.int32),
        pending_layers={"linear_attn"},
    )

    # Non-mamba layers and repeats are no-ops.
    state.do_mamba_copy_for_layer("layers.0.self_attn")
    mock_precopy_kernel.__getitem__.assert_not_called()
    state.do_mamba_copy_for_layer("linear_attn")
    state.do_mamba_copy_for_layer("linear_attn")
    mock_precopy_kernel.__getitem__.assert_called_once_with((2, 2))

    # The kernel sees one layer's (layer, state-type) metadata rows per
    # request row: grid == (num_reqs, num_state_types).
    launch = mock_precopy_kernel.__getitem__.return_value
    launch.assert_called_once()
    _, kwargs = launch.call_args
    assert kwargs["COPY_BLOCK_SIZE"] == 1024
    assert kwargs["HAS_IDX_MAPPING"] is True
    assert state._layerwise_mamba_copy.pending_layers == set()


@patch("vllm_ascend.worker.v2.model_states.mamba_hybrid.get_kv_transfer_group")
def test_deferred_copy_missing_layer_raises(mock_get_group):
    mock_get_group.return_value = SimpleNamespace(
        finish_mamba_state_copy=MagicMock(),
    )
    spec = _mamba_spec()
    kv_cache_config = _kv_cache_config(spec)
    state = _make_defer_state(kv_cache_config)
    state._layerwise_mamba_copy = SimpleNamespace(
        ctx=MagicMock(),
        num_reqs=1,
        idx_mapping=torch.arange(1, dtype=torch.int32),
        pending_layers={"linear_attn"},
    )

    with pytest.raises(RuntimeError, match="linear_attn"):
        state._finish_previous_layerwise_mamba_copy()


def test_prepare_inputs_propagates_padded_request_count():
    model_runner_path = Path(__file__).resolve().parents[3] / "vllm_ascend" / "worker" / "v2" / "model_runner.py"
    module = ast.parse(model_runner_path.read_text(encoding="utf-8"))
    prepare_inputs = next(
        node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "prepare_inputs"
    )

    assignments = {
        target.id: node.value
        for node in ast.walk(prepare_inputs)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert ast.unparse(assignments["query_start_loc"]) == ("self.input_buffers.query_start_loc[:num_reqs_padded + 1]")
    assert ast.unparse(assignments["seq_lens"]) == "self.input_buffers.seq_lens[:num_reqs_padded]"

    input_batch = next(
        node
        for node in ast.walk(prepare_inputs)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "AscendInputBatch"
    )
    keywords = {keyword.arg: keyword.value for keyword in input_batch.keywords}
    padded_count = keywords["num_reqs_after_padding"]
    assert isinstance(padded_count, ast.Name)
    assert padded_count.id == "num_reqs_padded"


@patch(
    "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
    return_value=SimpleNamespace(kv_transfer_config=None),
)
def test_mamba_cache_reshape_returns_contiguous_state_tensors(_mock_config):
    spec = _mamba_spec()
    kv_cache_config = _kv_cache_config(spec)

    raw_caches = _allocate_kv_cache(
        kv_cache_config,
        shared_layers={},
        device=torch.device("cpu"),
    )
    raw_cache = raw_caches["linear_attn"]
    assert isinstance(raw_cache, torch.Tensor)
    assert raw_cache.numel() == 3 * spec.page_size_bytes

    caches = _reshape_kv_cache_v2(
        attn_groups=[_group(spec)],
        kv_cache_raw_tensors=raw_caches,
        cache_dtype="auto",
        kernel_block_sizes=[spec.block_size],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    state_tensors = caches["linear_attn"]
    assert isinstance(state_tensors, list)
    assert len(state_tensors) == len(spec.shapes)

    conv_state, ssm_state = state_tensors
    assert conv_state.shape == (3, 2, 3)
    assert ssm_state.shape == (3, 2, 2)
    assert conv_state.dtype == torch.float16
    assert ssm_state.dtype == torch.float32
    assert conv_state.is_contiguous()
    assert ssm_state.is_contiguous()
    assert conv_state.data_ptr() == raw_cache.data_ptr()
    assert ssm_state.data_ptr() - raw_cache.data_ptr() == (conv_state.numel() * conv_state.element_size())


@patch(
    "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
    return_value=SimpleNamespace(kv_transfer_config=None),
)
def test_hybrid_cache_exposes_attention_views_and_mamba_states(_mock_config):
    attention_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
        page_size_padded=20,
    )
    mamba_spec = MambaSpec(
        block_size=4,
        shapes=((2,), (4,)),
        dtypes=(torch.float16, torch.float16),
        page_size_padded=20,
    )
    assert attention_spec.real_page_size_bytes == 16
    assert attention_spec.page_size_bytes == 20
    assert mamba_spec.page_size_bytes == 20

    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=40,
                shared_by=["full_attn", "linear_attn"],
            ),
            # Hybrid models can have an attention-only slot (for example an
            # MTP layer). It must still use the common single-tensor layout.
            KVCacheTensor(size=40, shared_by=["mtp_attn"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["full_attn", "mtp_attn"],
                kv_cache_spec=attention_spec,
            ),
            KVCacheGroupSpec(
                layer_names=["linear_attn"],
                kv_cache_spec=mamba_spec,
            ),
        ],
    )
    raw_caches = _allocate_kv_cache(
        kv_cache_config,
        shared_layers={},
        device=torch.device("cpu"),
    )
    raw_cache = raw_caches["linear_attn"]
    assert isinstance(raw_cache, torch.Tensor)
    assert raw_caches["full_attn"] is raw_cache
    assert isinstance(raw_caches["mtp_attn"], torch.Tensor)

    backend = MagicMock()
    backend.get_kv_cache_shape.return_value = (2, 2, 4, 1, 1)
    attention_group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=attention_spec,
        layer_names=["full_attn", "mtp_attn"],
        backend=backend,
    )
    mamba_group = SimpleNamespace(
        kv_cache_group_id=1,
        kv_cache_spec=mamba_spec,
        layer_names=["linear_attn"],
    )
    caches = _reshape_kv_cache_v2(
        attn_groups=[attention_group, mamba_group],
        kv_cache_raw_tensors=raw_caches,
        cache_dtype="auto",
        kernel_block_sizes=[4, 4],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )

    key_cache, value_cache = caches["full_attn"]
    mtp_key_cache, mtp_value_cache = caches["mtp_attn"]
    mamba_states = caches["linear_attn"]
    assert isinstance(mamba_states, list)
    conv_state, ssm_state = mamba_states
    assert conv_state.shape == (2, 2)
    assert ssm_state.shape == (2, 4)
    assert conv_state.is_contiguous()
    assert ssm_state.is_contiguous()
    assert conv_state.data_ptr() == raw_cache.data_ptr()
    assert ssm_state.data_ptr() - raw_cache.data_ptr() == (conv_state.numel() * conv_state.element_size())
    assert key_cache.data_ptr() == ssm_state.data_ptr()
    assert value_cache.data_ptr() - raw_cache.data_ptr() == 24
    assert key_cache.is_contiguous()
    assert value_cache.is_contiguous()
    assert mtp_key_cache.shape == key_cache.shape
    assert mtp_value_cache.shape == value_cache.shape


@patch(
    "vllm_ascend.worker.v2.attn_utils._get_attention_kv_cache_dims",
    return_value=(4, 4),
)
@patch(
    "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
    return_value=SimpleNamespace(kv_transfer_config=None),
)
def test_attention_cache_reshape_uses_virtual_kernel_block_count(
    _mock_config,
    _mock_cache_dims,
):
    spec = AscendMLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    assert spec.page_size_bytes == 1024

    num_blocks = 3
    raw_cache = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    backend = MagicMock()
    backend.get_kv_cache_shape.side_effect = (
        lambda num_kernel_blocks, block_size, _num_heads, _head_size, _cache_dtype: (
            num_kernel_blocks,
            block_size,
            1,
            8,
        )
    )
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        layer_names=["mla_attn"],
        backend=backend,
    )

    caches = _reshape_kv_cache_v2(
        attn_groups=[group],
        kv_cache_raw_tensors={"mla_attn": raw_cache},
        cache_dtype="auto",
        kernel_block_sizes=[4],
        shared_kv_cache_layers={},
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=raw_cache.numel(),
                    shared_by=["mla_attn"],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=["mla_attn"],
                    kv_cache_spec=spec,
                )
            ],
        ),
    )

    key_cache, value_cache = caches["mla_attn"]
    num_kernel_blocks = num_blocks * spec.block_size // 4
    assert key_cache.shape == (num_kernel_blocks, 4, 1, 4)
    assert value_cache.shape == key_cache.shape
    assert key_cache.is_contiguous()
    assert value_cache.is_contiguous()
    assert backend.get_kv_cache_shape.call_args.args[0] == num_kernel_blocks


@patch("vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config")
def test_get_kv_cache_spec_keeps_mamba_layers(mock_get_layers):
    spec = _mamba_spec()
    mamba_layer = MagicMock()
    mamba_layer.kv_sharing_target_layer_name = None
    mamba_layer.get_kv_cache_spec.return_value = spec
    mock_get_layers.return_value = {"linear_attn": mamba_layer}

    assert get_kv_cache_spec(MagicMock()) == {"linear_attn": spec}


@patch("vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config")
def test_mamba_spec_follows_aligned_attention_spec(
    mock_get_layers,
):
    attention_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=4,
        shapes=((2,), (4,)),
        dtypes=(torch.float16, torch.float16),
        page_size_padded=20,
    )

    class FakeAttention:
        kv_sharing_target_layer_name = None

        def get_kv_cache_spec(self, _vllm_config):
            return attention_spec

    mamba_layer = MagicMock()
    mamba_layer.kv_sharing_target_layer_name = None
    mamba_layer.get_kv_cache_spec.return_value = mamba_spec
    mock_get_layers.return_value = {
        "linear_attn": mamba_layer,
        "full_attn": FakeAttention(),
    }

    specs = get_kv_cache_spec(MagicMock())

    assert list(specs) == ["full_attn", "linear_attn"]
    assert specs["full_attn"].page_size_bytes == 20
    assert specs["full_attn"].indexes_kv_by_block_stride is True


@patch("vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config")
def test_get_kv_cache_spec_aligns_nondivisible_attention_and_mamba_pages(
    mock_get_layers,
):
    small_attention_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.float16,
    )
    large_attention_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=5,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=4,
        shapes=((2,), (4,)),
        dtypes=(torch.float16, torch.float16),
        page_size_padded=20,
    )
    assert small_attention_spec.page_size_bytes == 48
    assert large_attention_spec.page_size_bytes == 80
    assert mamba_spec.page_size_bytes == 20

    class FakeAttention:
        kv_sharing_target_layer_name = None

        def __init__(self, spec):
            self.spec = spec

        def get_kv_cache_spec(self, _vllm_config):
            return self.spec

    mamba_layer = MagicMock()
    mamba_layer.kv_sharing_target_layer_name = None
    mamba_layer.get_kv_cache_spec.return_value = mamba_spec
    mock_get_layers.return_value = {
        "small_attn": FakeAttention(small_attention_spec),
        "linear_attn": mamba_layer,
        "large_attn": FakeAttention(large_attention_spec),
    }

    specs = get_kv_cache_spec(MagicMock())

    assert {spec.page_size_bytes for spec in specs.values()} == {80}
    assert specs["small_attn"].indexes_kv_by_block_stride is True
    assert specs["large_attn"].indexes_kv_by_block_stride is True
    assert specs["linear_attn"].page_size_padded == 80


@patch("vllm_ascend.worker.v2.model_states.mamba_hybrid.AscendMambaHybridModelState")
def test_hybrid_model_selects_mamba_model_state(mock_mamba_state):
    vllm_config = MagicMock()
    vllm_config.model_config.is_hybrid = True
    model = torch.nn.Module()
    encoder_cache = MagicMock()
    device = torch.device("cpu")

    state = init_asecnd_model_state(
        vllm_config,
        model,
        encoder_cache,
        device,
    )

    assert state is mock_mamba_state.return_value
    mock_mamba_state.assert_called_once_with(
        vllm_config,
        model,
        encoder_cache,
        device,
    )
