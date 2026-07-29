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

from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state


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
    offset: int = 0,
    block_stride: int = 0,
) -> KVCacheConfig:
    size = (
        num_blocks * block_stride
        if block_stride
        else num_blocks * spec.page_size_bytes
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=size,
                shared_by=["linear_attn"],
                offset=offset,
                block_stride=block_stride,
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


@patch(
    "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
    return_value=SimpleNamespace(kv_transfer_config=None),
)
def test_mamba_cache_allocate_and_reshape(_mock_config):
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
    conv_state, ssm_state = caches["linear_attn"]
    assert conv_state.shape == (3, 2, 3)
    assert ssm_state.shape == (3, 2, 2)
    assert conv_state.is_contiguous()
    assert ssm_state.is_contiguous()
    assert ssm_state.data_ptr() - raw_cache.data_ptr() == 3 * 2 * 3 * 2


@patch(
    "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
    return_value=SimpleNamespace(kv_transfer_config=None),
)
def test_packed_mamba_cache_uses_layer_offset_and_block_stride(_mock_config):
    spec = _mamba_spec()
    kv_cache_config = _kv_cache_config(
        spec,
        offset=8,
        block_stride=64,
    )
    raw_caches = _allocate_kv_cache(
        kv_cache_config,
        shared_layers={},
        device=torch.device("cpu"),
    )
    raw_cache = raw_caches["linear_attn"]
    assert isinstance(raw_cache, torch.Tensor)

    caches = _reshape_kv_cache_v2(
        attn_groups=[_group(spec)],
        kv_cache_raw_tensors=raw_caches,
        cache_dtype="auto",
        kernel_block_sizes=[spec.block_size],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    conv_state, ssm_state = caches["linear_attn"]
    assert conv_state.data_ptr() - raw_cache.data_ptr() == 8
    assert ssm_state.data_ptr() - raw_cache.data_ptr() == 8 + 2 * 3 * 2
    assert conv_state.stride(0) * conv_state.element_size() == 64
    assert ssm_state.stride(0) * ssm_state.element_size() == 64


@patch(
    "vllm_ascend.worker.v2.attn_utils.get_current_vllm_config",
    return_value=SimpleNamespace(kv_transfer_config=None),
)
def test_hybrid_attention_and_mamba_share_aligned_cache(_mock_config):
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
    conv_state, ssm_state = caches["linear_attn"]
    assert conv_state.data_ptr() == raw_cache.data_ptr()
    assert ssm_state.data_ptr() == key_cache.data_ptr()
    assert value_cache.data_ptr() - raw_cache.data_ptr() == 24
    assert mtp_key_cache.shape == key_cache.shape
    assert mtp_value_cache.shape == value_cache.shape


@patch("vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config")
def test_get_kv_cache_spec_keeps_mamba_layers(mock_get_layers):
    spec = _mamba_spec()
    mamba_layer = MagicMock()
    mamba_layer.kv_sharing_target_layer_name = None
    mamba_layer.get_kv_cache_spec.return_value = spec
    mock_get_layers.return_value = {"linear_attn": mamba_layer}

    assert get_kv_cache_spec(MagicMock()) == {"linear_attn": spec}


@patch("vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config")
def test_get_kv_cache_spec_aligns_attention_and_orders_mamba_last(
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

    with patch("vllm_ascend.worker.v2.attn_utils.Attention", FakeAttention):
        specs = get_kv_cache_spec(MagicMock())

    assert list(specs) == ["full_attn", "linear_attn"]
    assert specs["full_attn"].page_size_bytes == 20


@patch(
    "vllm_ascend.worker.v2.model_states.mamba_hybrid."
    "AscendMambaHybridModelState"
)
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
