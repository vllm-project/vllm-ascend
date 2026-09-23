# SPDX-License-Identifier: Apache-2.0
"""CPU byte-layout checks for the real FlashMLA cache and zeroer helpers."""

import ast
import itertools
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
HELPERS = runpy.run_path(str(ROOT / "vllm_ascend/worker/flash_kv_cache.py"))


@dataclass
class Spec:
    block_size: int
    dtype: torch.dtype
    page_size_bytes: int
    cache_dtype_str: str | None = "auto"
    state_content_bytes: int = 1152


def methods(path, class_name, names, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    parent = next(n for n in tree.body if getattr(n, "name", None) == class_name) if class_name else tree
    selected = [n for n in parent.body if getattr(n, "name", None) in names]
    assert len(selected) == len(names)
    module = ast.Module(body=selected, type_ignores=[])
    exec("from __future__ import annotations\n" + ast.unparse(module), scope)
    return scope


@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("ratio", [1, 2, 6])
@pytest.mark.parametrize("padding", [0, 256])
def test_kernel_pages_preserve_offsets_planes_and_padding(c8, ratio, padding):
    dtype = torch.float8_e4m3fn if c8 else torch.bfloat16
    kernel, pages, offset = 128, 3, 64
    content_bytes = 640 if c8 else 1152
    kernel_stride = kernel * content_bytes + padding
    spec = Spec(kernel * ratio, dtype, kernel_stride * ratio)
    backing = torch.full((offset + pages * spec.page_size_bytes + 64,), -91, dtype=torch.int8)
    raw = backing[offset:-64]
    result = HELPERS["view_flash_mla_cache"](raw, spec, kernel)
    components = result if c8 else (result,)
    assert all(t.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr() for t in components)
    assert all(t.shape[0] == pages * ratio for t in components)
    if c8:
        latent, rope = components
        assert latent.shape == (pages * ratio, 128, 1, 512)
        assert rope.shape == (pages * ratio, 128, 1, 64)
        assert rope.dtype == torch.bfloat16
        assert rope.data_ptr() - latent.data_ptr() == 128 * 512
    else:
        assert result.shape == (pages * ratio, 128, 576)
    for t in components:
        assert t.stride(0) * t.element_size() == kernel_stride
        t.view(torch.uint8).fill_(0)
    expected = torch.full_like(backing, -91)
    for page in range(pages * ratio):
        start = offset + page * kernel_stride
        expected[start : start + kernel * content_bytes] = 0
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


def test_c8_spec_retains_manager_contract():
    spec = Spec(768, torch.bfloat16, 768 * 1152)
    attention = SimpleNamespace(impl=SimpleNamespace(dtype=torch.float8_e4m3fn), kv_lora_rank=512, qk_rope_head_dim=64)
    updated = HELPERS["customize_flash_mla_c8_spec"](spec, attention)
    assert updated.state_content_bytes == 640
    assert updated.dtype == torch.float8_e4m3fn and updated.cache_dtype_str is None
    assert updated.block_size == spec.block_size
    assert spec.dtype == torch.bfloat16 and spec.state_content_bytes == 1152


@pytest.mark.parametrize("c8", [False, True])
@pytest.mark.parametrize("ratio", [1, 2])
def test_zeroer_owns_only_selected_manager_payload(c8, ratio):
    dtype = torch.float8_e4m3fn if c8 else torch.bfloat16
    kernel, pages, padding, offset = 128, 3, 256, 64
    content_bytes = 640 if c8 else 1152
    kernel_stride = kernel * content_bytes + padding
    spec = Spec(kernel * ratio, dtype, kernel_stride * ratio)
    backing = torch.full((offset + pages * spec.page_size_bytes + 64,), -91, dtype=torch.int8)
    cache = HELPERS["view_flash_mla_cache"](backing[offset:-64], spec, kernel)
    scope = methods(
        "vllm_ascend/worker/utils.py",
        "AscendKVBlockZeroer",
        ["__init__", "init_meta"],
        dict(torch=torch, FullAttentionSpec=Spec, largest_power_of_2_divisor=lambda n: n & -n, iprod=itertools.product),
    )
    zeroer = type("Zeroer", (), {name: scope[name] for name in ("__init__", "init_meta")})(torch.device("cpu"), False)
    zeroer.init_meta(
        attn_groups_iter=[SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["mla"])],
        kernel_block_sizes=[[kernel]],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context={"mla": SimpleNamespace(kv_cache=cache, impl=SimpleNamespace(use_flash_mla=True))},
        page_strided=True,
    )
    addresses = zeroer._meta[0].tolist()
    for address, stride, size in zip(addresses, zeroer._seg_strides.tolist(), zeroer._seg_sizes.tolist()):
        start = address - backing.data_ptr() + stride * 4  # scheduler block 1
        assert 0 <= start < backing.numel() and start + size * 4 <= backing.numel()
        backing[start : start + size * 4] = 0
    expected = torch.full_like(backing, -91)
    for subpage in range(ratio):
        start = offset + spec.page_size_bytes + subpage * kernel_stride
        expected[start : start + kernel * content_bytes] = 0
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


@pytest.mark.parametrize("runner", [1, 2])
@pytest.mark.parametrize("c8", [False, True])
def test_runner_reshape_dispatches_real_flash_cache_helper(runner, c8):
    dtype = torch.float8_e4m3fn if c8 else torch.bfloat16
    spec = Spec(256, dtype, 256 * (640 if c8 else 1152))
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    layer_type = type("MLAAttention", (), {})
    layer = layer_type()
    layer.impl = SimpleNamespace(use_flash_mla=True)
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["mla"], backend=None)
    config = SimpleNamespace(kv_cache_groups=[group])
    common = dict(
        torch=torch,
        get_current_vllm_config=lambda: SimpleNamespace(),
        _is_dsv4_model=lambda _: False,
        get_layers_from_vllm_config=lambda *args: {"mla": layer},
        AttentionLayerBase=object,
        MLAAttention=layer_type,
        _get_layer_kv_cache_specs=lambda _: {"mla": spec},
        get_storage_block_size=lambda s: s.block_size,
        view_flash_mla_cache=HELPERS["view_flash_mla_cache"],
        requires_padded_page_layout=lambda _: False,
        is_deepseek_v41_cache=lambda _: False,
        is_glm5_next_cache_spec=lambda _: False,
    )
    if runner == 1:
        scope = methods(
            "vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner", ["_reshape_kv_cache_tensors"], common
        )
        model = SimpleNamespace(
            _get_layer_kv_cache_specs=lambda _: {"mla": spec},
            _kv_cache_spec_attn_group_iterator=lambda: iter([group]),
            runner_only_attn_layers=set(),
            kernel_block_sizes=[[128]],
            compilation_config=SimpleNamespace(static_forward_context={"mla": layer}),
        )
        actual = scope["_reshape_kv_cache_tensors"](model, config, {"mla": raw})["mla"]
    else:
        scope = methods("vllm_ascend/worker/v2/attn_utils.py", None, ["_reshape_kv_cache_v2"], common)
        actual = scope["_reshape_kv_cache_v2"]([group], {"mla": raw}, "auto", [128], {}, config)["mla"]
    components = actual if c8 else (actual,)
    assert all(t.shape[0] == 6 and t.shape[1] == 128 for t in components)
    assert all(t.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr() for t in components)
