from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.worker.utils import AscendKVBlockZeroer


def _attention_group(*, group_id=0, layer_names=None):
    spec = FullAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
    )
    return SimpleNamespace(
        kv_cache_group_id=group_id,
        kv_cache_spec=spec,
        layer_names=["full_attn"] if layer_names is None else layer_names,
    )


def _context():
    shape = (2, 4, 1, 2)
    return {
        "full_attn": SimpleNamespace(
            kv_cache=(torch.zeros(shape), torch.zeros(shape)),
        )
    }


def test_init_meta_uses_flat_kernel_size_for_virtual_blocks():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)

    zeroer.init_meta(
        [_attention_group()],
        kernel_block_sizes=[4],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=_context(),
    )

    assert zeroer._meta is not None
    segment_addresses, page_size_el, block_size, num_segments = zeroer._meta
    assert segment_addresses.numel() == 2
    assert page_size_el == 16
    assert block_size == 16
    assert num_segments == 2


def test_init_meta_skips_group_without_kernel_size():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)

    zeroer.init_meta(
        [_attention_group(group_id=2)],
        kernel_block_sizes=[4],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=_context(),
    )

    assert zeroer._meta is None


def test_init_meta_skips_runner_only_attention_layer():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)

    zeroer.init_meta(
        [_attention_group()],
        kernel_block_sizes=[4],
        cache_dtype="auto",
        runner_only_attn_layers={"full_attn"},
        static_forward_context=_context(),
    )

    assert zeroer._meta is None


def test_init_meta_ignores_mamba_groups():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    spec = MambaSpec(
        block_size=8,
        shapes=((3,), (5,)),
        dtypes=(torch.float16, torch.float16),
        page_size_padded=32,
        mamba_cache_mode="align",
    )
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        layer_names=["linear_attn"],
    )

    zeroer.init_meta(
        [group],
        kernel_block_sizes=[8],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={},
    )

    assert zeroer._meta is None


def test_init_meta_deduplicates_shared_cache_pointers():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    shared_cache = torch.zeros((2, 4, 1, 2))
    context = {
        "full_attn": SimpleNamespace(kv_cache=(shared_cache, shared_cache)),
    }

    zeroer.init_meta(
        [_attention_group()],
        kernel_block_sizes=[4],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=context,
    )

    assert zeroer._meta is not None
    segment_addresses, _, _, num_segments = zeroer._meta
    assert segment_addresses.numel() == 1
    assert num_segments == 1


def test_init_meta_rejects_nonuniform_attention_page_sizes():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    first_group = _attention_group(group_id=0)
    second_spec = FullAttentionSpec(
        block_size=12,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float16,
    )
    second_group = SimpleNamespace(
        kv_cache_group_id=1,
        kv_cache_spec=second_spec,
        layer_names=["other_attn"],
    )
    shape = (2, 4, 1, 2)
    context = {
        "full_attn": SimpleNamespace(
            kv_cache=(torch.zeros(shape), torch.zeros(shape)),
        ),
        "other_attn": SimpleNamespace(
            kv_cache=(torch.zeros(shape), torch.zeros(shape)),
        ),
    }

    with pytest.raises(AssertionError, match="Non-uniform page sizes"):
        zeroer.init_meta(
            [first_group, second_group],
            kernel_block_sizes=[4, 4],
            cache_dtype="auto",
            runner_only_attn_layers=set(),
            static_forward_context=context,
        )
