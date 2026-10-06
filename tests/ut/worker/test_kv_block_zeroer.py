#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.worker import utils as worker_utils
from vllm_ascend.worker.utils import AscendKVBlockZeroer


class _FakeKernel:
    def __init__(self) -> None:
        self.grid: tuple[int, ...] | None = None
        self.args: tuple[object, ...] | None = None
        self.kwargs: dict[str, object] | None = None

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., None]:
        self.grid = grid

        def launch(*args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        return launch


def test_physical_block_tensors_support_mixed_pages_and_deduplicate():
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    small = torch.empty(4 * 32, dtype=torch.int8)
    large = torch.empty(4 * 64, dtype=torch.int8)
    zeroer.init_meta(
        attn_groups_iter=[],
        kernel_block_sizes=[],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={},
        physical_block_tensors=[small, small, large],
        num_blocks=4,
    )
    assert zeroer._meta is not None
    addresses, sizes, max_chunks, block_size, n_segs = zeroer._meta
    assert addresses.tolist() == [small.data_ptr(), large.data_ptr()]
    assert sizes.tolist() == [8, 16]
    assert (max_chunks, block_size, n_segs) == (2, 8, 2)


@pytest.mark.parametrize("num_blocks", [None, 0, -1])
def test_physical_block_tensors_require_positive_block_count(num_blocks):
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        zeroer.init_meta([], [], "auto", set(), {}, physical_block_tensors=[], num_blocks=num_blocks)


@pytest.mark.parametrize(
    "tensor,blocks,error",
    [
        (torch.empty(8, 8).t(), 4, "contiguous whole blocks"),
        (torch.empty(15, dtype=torch.int8), 4, "contiguous whole blocks"),
        (torch.empty(12, dtype=torch.int8), 4, "4-byte aligned"),
    ],
)
def test_physical_block_tensors_reject_invalid_storage(tensor, blocks, error):
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    with pytest.raises(ValueError, match=error):
        zeroer.init_meta([], [], "auto", set(), {}, physical_block_tensors=[tensor], num_blocks=blocks)


def test_init_meta_supports_non_uniform_page_sizes() -> None:
    """MLA K/V cache segments may have different sizes per logical block."""
    device = torch.device("cpu")
    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    k_cache = torch.zeros((4, 16384), dtype=torch.int32, device=device)
    v_cache = torch.zeros((4, 4096), dtype=torch.int32, device=device)
    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
        ),
        kv_cache_group_id=0,
        layer_names=["mla_layer"],
    )

    zeroer.init_meta(
        attn_groups_iter=[group],
        kernel_block_sizes=[128],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={"mla_layer": SimpleNamespace(kv_cache=(k_cache, v_cache))},
    )

    assert zeroer._meta is not None
    _, seg_page_sizes, max_chunks, block_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == [16384, 4096]
    assert block_size == 4096
    assert max_chunks == 4
    assert n_segs == 2


def test_init_meta_preserves_uniform_page_size_behavior() -> None:
    """The per-segment metadata path must retain the former uniform layout."""
    device = torch.device("cpu")
    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    cache = torch.zeros((4, 16384), dtype=torch.int32, device=device)
    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
        ),
        kv_cache_group_id=0,
        layer_names=["attn_layer"],
    )

    zeroer.init_meta(
        attn_groups_iter=[group],
        kernel_block_sizes=[128],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={"attn_layer": SimpleNamespace(kv_cache=(cache, cache.clone()))},
    )

    assert zeroer._meta is not None
    _, seg_page_sizes, max_chunks, block_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == [16384, 16384]
    assert block_size == 8192
    assert max_chunks == 2
    assert n_segs == 2


def test_zero_block_ids_dispatches_per_segment_page_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper must pass per-segment metadata to the Triton kernel."""
    zeroer = AscendKVBlockZeroer(torch.device("cpu"), pin_memory=False)
    seg_addrs = torch.tensor([101, 202], dtype=torch.uint64)
    seg_page_sizes = torch.tensor([16384, 4096], dtype=torch.int64)
    zeroer._meta = (seg_addrs, seg_page_sizes, 4, 4096, 2)
    zeroer._id_cap = 4
    zeroer._ids_pinned = torch.empty(4, dtype=torch.int64)
    zeroer._ids_gpu = torch.empty(4, dtype=torch.int64)

    fake_kernel = _FakeKernel()
    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", fake_kernel)
    monkeypatch.setattr(worker_utils, "get_vectorcore_num", lambda: 8)

    zeroer.zero_block_ids([1, 3])

    assert fake_kernel.grid == (8,)
    assert fake_kernel.args is not None
    assert fake_kernel.args[0] is seg_addrs
    assert fake_kernel.args[1] is seg_page_sizes
    assert isinstance(fake_kernel.args[2], torch.Tensor)
    assert torch.equal(fake_kernel.args[2], torch.tensor([1, 3], dtype=torch.int64))
    assert fake_kernel.args[3] == 2
    assert fake_kernel.kwargs == {
        "N_SEGS": 2,
        "MAX_CHUNKS": 4,
        "BLOCK_SIZE": 4096,
        "GRID_SIZE": 8,
    }


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
    segment_addresses, page_sizes_el, max_chunks, block_size, num_segments = zeroer._meta
    assert segment_addresses.numel() == 2
    assert page_sizes_el.tolist() == [16, 16]
    assert max_chunks == 1
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
    segment_addresses, _, _, _, num_segments = zeroer._meta
    assert segment_addresses.numel() == 1
    assert num_segments == 1


def test_init_meta_supports_nonuniform_attention_page_sizes_across_groups():
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

    zeroer.init_meta(
        [first_group, second_group],
        kernel_block_sizes=[4, 4],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=context,
    )

    assert zeroer._meta is not None
    segment_addresses, page_sizes_el, max_chunks, block_size, num_segments = zeroer._meta
    assert segment_addresses.numel() == 4
    assert page_sizes_el.tolist() == [16, 16, 24, 24]
    assert block_size == 8
    assert max_chunks == 3
    assert num_segments == 4
