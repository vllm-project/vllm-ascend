import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


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
    assert AscendMambaHybridModelState.preprocess_state is MambaHybridModelState.preprocess_state
    assert AscendMambaHybridModelState.postprocess_state is MambaHybridModelState.postprocess_state


def test_partial_hit_copy_uses_logical_blocks_for_attention_and_mamba():
    runner = object.__new__(NPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(num_blocks=3)
    attention_cache = torch.arange(18, dtype=torch.float32).view(6, 3)
    conv_state = torch.arange(12, dtype=torch.float32).view(3, 4)
    recurrent_state = torch.arange(6, dtype=torch.float32).view(3, 2)
    runner.kv_caches = [attention_cache, [conv_state, recurrent_state]]
    expected_attention = attention_cache.clone()
    expected_conv = conv_state.clone()
    expected_recurrent = recurrent_state.clone()

    runner._copy_kv_cache_blocks_for_partial_hits([(0, 1), (1, 2)])

    torch.testing.assert_close(attention_cache[2:4], expected_attention[0:2])
    torch.testing.assert_close(attention_cache[4:6], expected_attention[2:4])
    torch.testing.assert_close(conv_state[1], expected_conv[0])
    torch.testing.assert_close(conv_state[2], expected_conv[1])
    torch.testing.assert_close(recurrent_state[1], expected_recurrent[0])
    torch.testing.assert_close(recurrent_state[2], expected_recurrent[1])


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
