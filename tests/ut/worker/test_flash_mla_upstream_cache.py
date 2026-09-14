# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run VA allocation and reshape methods with real CPU tensor storage."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _MambaSpec(SimpleNamespace):
    pass


class _MLAAttention:
    pass


def _runner_methods():
    # Extract production methods without importing unrelated NPU/vLLM modules.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    selected = {
        "_align_memory",
        "_allocate_int8_cache_tensor",
        "_get_layer_kv_cache_specs",
        "_allocate_kv_cache_tensors",
        "_reshape_kv_cache_tensors",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"))
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in selected]
    scope = {
        "torch": torch,
        "math": math,
        "ascend_envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
        "get_kv_cache_tensor_layers": lambda descriptor: descriptor.layers,
        "MambaSpec": _MambaSpec,
        "MLAAttention": _MLAAttention,
        "UniformTypeKVCacheSpecs": type("UniformTypeKVCacheSpecs", (), {}),
        "AttentionLayerBase": type("AttentionLayerBase", (), {}),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return type("RunnerMethods", (), {name: scope[name] for name in selected})()


@pytest.mark.parametrize("with_kv_transfer", [False, True])
def test_va_shared_backing_preserves_mixed_views_and_page_holes(with_kv_transfer):
    runner = _runner_methods()
    blocks, block_size, offset = 3, 16, 64
    mla_bytes = block_size * 576 * 2
    layer_stride = mla_bytes + 128
    pitch = 4 * layer_stride + 128
    size = offset + blocks * pitch + 64

    def attention_spec(heads, head_dim):
        return SimpleNamespace(
            block_size=block_size,
            num_heads=heads,
            num_states=block_size,
            state_content_size_bytes=head_dim * 2,
            dtype=torch.bfloat16,
            get_num_kernel_states=lambda kernel_size: kernel_size,
        )

    mla_spec = attention_spec(1, 576)
    gqa_spec = attention_spec(4, 64)
    shapes = ((6, 48), (2, 8, 8))
    dtypes = (torch.bfloat16, torch.float32)
    state_bytes = 6 * 48 * 2 + 2 * 8 * 8 * 4
    mamba_spec = _MambaSpec(
        shapes=shapes,
        dtypes=dtypes,
        num_heads=1,
        num_states=1,
        state_content_size_bytes=state_bytes,
    )
    groups = [
        SimpleNamespace(layer_names=["mla0", "mla1"], kv_cache_spec=mla_spec, backend=None, kv_cache_group_id=0),
        SimpleNamespace(layer_names=["gqa"], kv_cache_spec=gqa_spec, backend=None, kv_cache_group_id=1),
        SimpleNamespace(layer_names=["mamba"], kv_cache_spec=mamba_spec, backend=None, kv_cache_group_id=2),
    ]
    config = SimpleNamespace(
        num_blocks=blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            SimpleNamespace(
                size=size,
                layers=group.layer_names,
                offset=offset + slot * layer_stride,
                layer_stride=layer_stride,
                block_stride=pitch,
            )
            for group, slot in zip(groups, (0, 2, 3))
        ],
    )
    runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    runner.vllm_config = SimpleNamespace(kv_transfer_config=object() if with_kv_transfer else None)
    runner.device = torch.device("cpu")
    runner.compilation_config = SimpleNamespace(
        static_forward_context={"mla0": _MLAAttention(), "mla1": _MLAAttention(), "gqa": object(), "mamba": object()}
    )
    runner.runner_only_attn_layers = set()
    runner.kernel_block_sizes = [[block_size], [block_size], [32768]]
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(groups)

    raw_caches = runner._allocate_kv_cache_tensors(config)
    caches = runner._reshape_kv_cache_tensors(config, raw_caches)
    assert len({raw.untyped_storage().data_ptr() for raw in raw_caches.values()}) == 1
    first = raw_caches["mla0"]
    backing_offset = first.storage_offset() - offset
    if with_kv_transfer:
        assert (first.data_ptr() - offset) % (2 * 1024 * 1024) == 0
    raw = torch.empty(0, dtype=torch.int8).set_(first.untyped_storage())
    assert not torch.count_nonzero(raw)

    mla = caches["mla1"]
    gqa = caches["gqa"]
    conv, ssm = caches["mamba"]
    assert mla.shape == (blocks, block_size, 576)
    assert gqa.shape == (blocks, 4, block_size, 64)
    assert mla.stride() == (pitch // 2, 576, 1)
    assert gqa.stride() == (pitch // 2, block_size * 64, 64, 1)
    assert conv.shape == (blocks, *shapes[0]) and conv.dtype == torch.bfloat16
    assert ssm.shape == (blocks, *shapes[1]) and ssm.dtype == torch.float32
    assert conv.stride() == (pitch // 2, 48, 1)
    assert ssm.stride() == (pitch // 4, 64, 8, 1)
    expected = {
        "mla0": backing_offset + offset,
        "mla1": backing_offset + offset + layer_stride,
        "gqa": backing_offset + offset + 2 * layer_stride,
        "mamba": backing_offset + offset + 3 * layer_stride,
    }
    for name, tensor in raw_caches.items():
        assert tensor.storage_offset() == expected[name]
        assert tensor.stride() == (pitch, 1)
    assert mla.storage_offset() * 2 == expected["mla1"]
    assert gqa.storage_offset() * 2 == expected["gqa"]
    assert conv.storage_offset() * 2 == expected["mamba"]
    assert ssm.storage_offset() * 4 == expected["mamba"] + 6 * 48 * 2
    for tensor in (caches["mla0"], mla, gqa, conv, ssm):
        assert tensor.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()

    raw.fill_(-91)
    touched = torch.zeros_like(raw, dtype=torch.bool)
    for page in range(blocks):
        mla[page, :, :512].fill_(page + 1)
        mla[page, :, 512:].fill_(page + 2)
        gqa[page, :2].fill_(page + 3)
        gqa[page, 2:].fill_(page + 4)
        conv[page].fill_(page + 5)
        ssm[page].fill_(page + 6)
        for name, payload in (("mla1", mla_bytes), ("gqa", 4 * block_size * 64 * 2), ("mamba", state_bytes)):
            start = expected[name] + page * pitch
            touched[start : start + payload] = True
        assert torch.all(mla[page, :, :512] == page + 1)
        assert torch.all(mla[page, :, 512:] == page + 2)
        assert torch.all(gqa[page, :2] == page + 3)
        assert torch.all(gqa[page, 2:] == page + 4)
        assert torch.all(conv[page] == page + 5)
        assert torch.all(ssm[page] == page + 6)
    # Includes untouched layers, page padding, prefix/suffix and alignment slack.
    assert torch.all(raw[~touched] == -91)
