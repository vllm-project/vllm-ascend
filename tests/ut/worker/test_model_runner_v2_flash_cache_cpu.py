# SPDX-License-Identifier: Apache-2.0
"""Execute MRv2 physical cache methods on CPU without the NPU/vLLM stack."""

import ast
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@dataclass(frozen=True)
class FullAttentionSpec:
    block_size: int = 16
    num_heads: int = 1
    state_content_size_bytes: int = 1152
    dtype: torch.dtype = torch.bfloat16
    page_size_padded: int | None = None
    non_causal_multi_token_decode: bool = False

    @property
    def num_states(self):
        return self.block_size

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.num_heads * self.num_states * self.state_content_size_bytes

    def get_num_kernel_states(self, kernel_size):
        return kernel_size


class MLAAttentionSpec(FullAttentionSpec):
    pass


@dataclass(frozen=True)
class AscendDCPReplicatedDraftAttentionSpec(FullAttentionSpec):
    dcp_replication_size: int = 1

    @property
    def lane_page_size_bytes(self):
        return self.num_heads * self.num_states * self.state_content_size_bytes

    @property
    def page_size_bytes(self):
        return self.dcp_replication_size * self.lane_page_size_bytes

    @classmethod
    def from_full_attention_spec(cls, spec, replication_size):
        return cls(**(spec.__dict__ | {"page_size_padded": None}), dcp_replication_size=replication_size)


@dataclass(frozen=True)
class MambaSpec:
    block_size: int = 32768
    shapes: tuple = ((6, 48), (2, 8, 8))
    dtypes: tuple = (torch.bfloat16, torch.float32)
    num_heads: int = 1
    num_states: int = 1
    state_content_size_bytes: int = 1088
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self):
        return self.page_size_padded or self.state_content_size_bytes


@dataclass
class UniformTypeKVCacheSpecs:
    kv_cache_specs: dict


class MLAAttention:
    def __init__(self, spec):
        self.spec = spec

    def get_kv_cache_spec(self, config):
        return self.spec


def _execute_definitions(relative_path, names, namespace):
    source = Path(__file__).resolve().parents[3] / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in definitions} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *definitions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)


def _cache_methods(with_transfer=False):
    config = SimpleNamespace(kv_transfer_config=object() if with_transfer else None)
    namespace = {
        "torch": torch,
        "np": np,
        "replace": replace,
        "ascend_envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        "get_current_vllm_config": lambda: config,
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
        "get_kv_cache_tensor_layers": lambda descriptor: descriptor.layers,
        "KVPPConfig": SimpleNamespace(from_vllm_config=lambda config: SimpleNamespace(size=1)),
        "MambaSpec": MambaSpec,
        "FullAttentionSpec": FullAttentionSpec,
        "MLAAttentionSpec": MLAAttentionSpec,
        "AscendMLAAttentionSpec": MLAAttentionSpec,
        "AscendDCPReplicatedDraftAttentionSpec": AscendDCPReplicatedDraftAttentionSpec,
        "MLAAttention": MLAAttention,
        "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
        "AttentionLayerBase": object,
        "get_storage_block_size": lambda spec: spec.block_size,
        "enable_sfa_dcp_replicated_indexer": lambda config: False,
        "HardwareCapability": SimpleNamespace(FP8_ATTENTION=object()),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda capability: False),
        "vllm_version_is": lambda version: False,
        "is_kimi_k3_gqa_dspark": lambda config: False,
    }
    _execute_definitions(
        "vllm_ascend/worker/v2/attn_utils.py",
        {
            "get_kv_cache_spec",
            "_get_layer_kv_cache_specs",
            "_is_dsv4_model",
            "_align_memory",
            "_allocate_kv_cache",
            "_reshape_kv_cache_v2",
        },
        namespace,
    )
    return namespace


def test_flash_spec_preserves_draft_capability_and_pads_hybrid_pages(monkeypatch):
    namespace = _cache_methods()
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.deepseek_v2", SimpleNamespace(DeepseekV32IndexerCache=object)
    )
    mla = MLAAttentionSpec(non_causal_multi_token_decode=True)
    gqa = FullAttentionSpec(num_heads=2, state_content_size_bytes=128, non_causal_multi_token_decode=True)
    mamba = MambaSpec()
    namespace["get_layers_from_vllm_config"] = lambda config, kind: {
        "mamba": SimpleNamespace(get_kv_cache_spec=lambda config: mamba),
        "mla": MLAAttention(mla),
        "gqa": SimpleNamespace(
            get_kv_cache_spec=lambda config: gqa,
            get_attn_backend=lambda: SimpleNamespace(
                customize_spec=lambda spec: replace(spec, num_heads=spec.num_heads * 2)
            ),
        ),
    }
    specs = namespace["get_kv_cache_spec"](SimpleNamespace())
    assert list(specs) == ["mla", "gqa", "mamba"]
    assert all(spec.page_size_bytes == mla.page_size_bytes for spec in specs.values())
    assert isinstance(specs["mla"], MLAAttentionSpec)
    assert specs["mla"].non_causal_multi_token_decode and specs["gqa"].non_causal_multi_token_decode
    assert specs["gqa"].num_heads == 4
    assert specs["mamba"].shapes == mamba.shapes and specs["mamba"].dtypes == mamba.dtypes


def test_replicated_gqa_spec_does_not_convert_target_mla(monkeypatch):
    namespace = _cache_methods()
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.deepseek_v2", SimpleNamespace(DeepseekV32IndexerCache=object)
    )
    mla = MLAAttentionSpec()
    gqa = FullAttentionSpec(num_heads=4, state_content_size_bytes=128)
    namespace["is_kimi_k3_gqa_dspark"] = lambda config: True
    namespace["get_layers_from_vllm_config"] = lambda config, kind: {
        "target": MLAAttention(mla),
        "draft": SimpleNamespace(
            get_kv_cache_spec=lambda config: gqa,
            get_attn_backend=lambda: SimpleNamespace(customize_spec=lambda spec: spec),
        ),
    }
    config = SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=2))
    specs = namespace["get_kv_cache_spec"](config)
    assert specs["target"] is mla
    assert isinstance(specs["draft"], AscendDCPReplicatedDraftAttentionSpec)
    assert specs["draft"].dcp_replication_size == 2
    assert specs["draft"].page_size_bytes == 2 * gqa.page_size_bytes


class _ZeroKernel:
    """CPU interval oracle for the real zeroer's recorded launch metadata."""

    def __init__(self, backing):
        self.backing = backing

    def __getitem__(self, grid):
        assert grid[0] > 0

        def launch(addresses, block_ids, n_blocks, strides, sizes, **constants):
            assert constants["PAGE_STRIDED"]
            for address, pitch, payload in zip(addresses.tolist(), strides.tolist(), sizes.tolist()):
                for block in block_ids[:n_blocks].tolist():
                    offset = address + block * pitch * 4 - self.backing.data_ptr()
                    length = payload * 4
                    assert offset >= 0 and offset + length <= self.backing.numel()
                    self.backing[offset : offset + length].zero_()

        return launch


def _init_zeroer(caches, groups, kernel_sizes, backing):
    namespace = {
        "torch": torch,
        "KVBlockZeroer": object,
        "FullAttentionSpec": FullAttentionSpec,
        "largest_power_of_2_divisor": lambda value: value & -value,
        "get_vectorcore_num": lambda: 8,
        "_zero_kv_blocks_kernel": _ZeroKernel(backing),
        "ascend_envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        "is_pin_memory_available": lambda: False,
    }
    _execute_definitions("vllm_ascend/worker/utils.py", {"AscendKVBlockZeroer"}, namespace)
    _execute_definitions("vllm_ascend/worker/v2/model_runner.py", {"_init_kv_zero_meta"}, namespace)
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        attn_groups=[[groups[0], groups[1]], [groups[2]]],
        kernel_block_sizes=kernel_sizes,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(
            static_forward_context={name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()}
        ),
    )
    namespace["_init_kv_zero_meta"](runner)
    return runner.kv_block_zeroer


@pytest.mark.parametrize("with_transfer", [False, True])
@pytest.mark.parametrize("pad_pages", [False, True])
@pytest.mark.parametrize("block_size", [16, 128])
def test_shared_flash_views_and_zeroer_preserve_page_holes(with_transfer, pad_pages, block_size):
    namespace = _cache_methods(with_transfer)
    blocks, prefix = 3, 64
    mla_bytes = block_size * 576 * 2
    padded_size = mla_bytes + 64 if pad_pages else None
    mla_spec = MLAAttentionSpec(block_size=block_size, page_size_padded=padded_size)
    gqa_spec = FullAttentionSpec(
        block_size=block_size, num_heads=4, state_content_size_bytes=128, page_size_padded=padded_size
    )
    mamba_spec = MambaSpec(page_size_padded=padded_size)
    slot_stride, pitch = mla_bytes + 128, 4 * (mla_bytes + 128) + 128
    size = prefix + blocks * pitch + 64
    groups = [
        SimpleNamespace(layer_names=["mla0", "mla1"], kv_cache_spec=mla_spec, kv_cache_group_id=0),
        SimpleNamespace(layer_names=["gqa"], kv_cache_spec=gqa_spec, kv_cache_group_id=1),
        SimpleNamespace(layer_names=["mamba"], kv_cache_spec=mamba_spec, kv_cache_group_id=2),
    ]
    config = SimpleNamespace(
        num_blocks=blocks,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=group.layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs({name: group.kv_cache_spec for name in group.layer_names}),
            )
            for group in groups
        ],
        kv_cache_tensors=[
            SimpleNamespace(
                size=size,
                layers=group.layer_names,
                offset=prefix + slot * slot_stride,
                layer_stride=slot_stride,
                block_stride=pitch,
            )
            for group, slot in zip(groups, (0, 2, 3))
        ],
    )
    raw_caches = namespace["_allocate_kv_cache"](config, {}, torch.device("cpu"))
    kernel_sizes = [block_size, block_size, mamba_spec.block_size]
    caches = namespace["_reshape_kv_cache_v2"](groups, raw_caches, "auto", kernel_sizes, {"shared_mla": "mla0"}, config)
    assert caches["shared_mla"] is caches["mla0"]
    first = raw_caches["mla0"]
    base_offset = first.storage_offset() - prefix
    backing = torch.empty(0, dtype=torch.int8).set_(first.untyped_storage())
    if with_transfer:
        assert (first.data_ptr() - prefix) % (2 * 1024 * 1024) == 0
    assert not torch.count_nonzero(backing)
    expected_offsets = {
        name: base_offset + prefix + slot * slot_stride
        for name, slot in (("mla0", 0), ("mla1", 1), ("gqa", 2), ("mamba", 3))
    }
    payloads = {
        "mla0": mla_bytes,
        "mla1": mla_bytes,
        "gqa": block_size * 4 * 64 * 2,
        "mamba": mamba_spec.state_content_size_bytes,
    }
    for name, raw in raw_caches.items():
        assert raw.untyped_storage().data_ptr() == backing.data_ptr()
        assert raw.storage_offset() == expected_offsets[name]
        assert raw.shape == (blocks, payloads[name]) and raw.stride() == (pitch, 1)
    assert caches["mla1"].shape == (blocks, block_size, 576)
    assert caches["mla1"].stride() == (pitch // 2, 576, 1)
    assert caches["gqa"].shape == (blocks, 4, block_size, 64)
    assert caches["gqa"].stride() == (pitch // 2, block_size * 64, 64, 1)
    conv, ssm = caches["mamba"]
    assert conv.shape == (blocks, 6, 48) and conv.dtype == torch.bfloat16
    assert ssm.shape == (blocks, 2, 8, 8) and ssm.dtype == torch.float32
    assert conv.stride() == (pitch // 2, 48, 1)
    assert ssm.stride() == (pitch // 4, 64, 8, 1)
    assert conv.storage_offset() * 2 == expected_offsets["mamba"]
    assert ssm.storage_offset() * 4 == expected_offsets["mamba"] + 6 * 48 * 2
    for name in ("mla0", "mla1", "gqa"):
        assert caches[name].storage_offset() * 2 == expected_offsets[name]

    backing.fill_(-91)
    touched = torch.zeros_like(backing, dtype=torch.bool)
    for page in range(blocks):
        for index, name in enumerate(("mla0", "mla1", "gqa")):
            caches[name][page].fill_(page + index + 1)
            assert torch.all(caches[name][page] == page + index + 1)
        conv[page].fill_(page + 4)
        ssm[page].fill_(page + 5)
        assert torch.all(conv[page] == page + 4) and torch.all(ssm[page] == page + 5)
        for name, length in payloads.items():
            start = expected_offsets[name] + page * pitch
            touched[start : start + length] = True
    assert torch.all(backing[~touched] == -91)

    expected = backing.clone()
    for name in ("mla0", "mla1", "gqa"):
        start = expected_offsets[name] + pitch
        expected[start : start + payloads[name]].zero_()
    zeroer = _init_zeroer(caches, groups, kernel_sizes, backing)
    zeroer.zero_block_ids([1])
    zeroer.zero_block_ids([])
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


@pytest.mark.parametrize("requested_block_size", [None, 128, 768])
def test_flash_large_recurrent_state_aligns_manager_pages_to_kernel_multiple(requested_block_size):
    state_shapes = ((6, 4096), (16, 128, 128))
    state_dtypes = (torch.bfloat16, torch.float32)
    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda config: state_shapes,
        get_mamba_state_dtype_from_config=lambda config: state_dtypes,
    )
    namespace = {
        "math": math,
        "cdiv": lambda a, b: (a + b - 1) // b,
        "envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        "MambaModelConfig": SimpleNamespace(verify_and_update_config=lambda config: None),
        "ModelRegistry": SimpleNamespace(resolve_model_cls=lambda *args, **kwargs: (model_cls, None)),
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
        "logger": SimpleNamespace(debug=lambda *args: None, info=lambda *args: None),
        "_get_sparse_index_kpool": lambda config: None,
    }
    _execute_definitions("vllm_ascend/patch/platform/patch_mamba_config.py", {"verify_and_update_config"}, namespace)
    cache = SimpleNamespace(
        cache_dtype="auto",
        block_size=requested_block_size,
        mamba_page_size_padded=None,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        cache_config=cache,
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            architecture="KimiK3ForConditionalGeneration",
            use_mla=True,
            get_num_kv_heads=lambda config: 1,
            max_model_len=8192,
            hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64),
        ),
        parallel_config=SimpleNamespace(),
        speculative_config=None,
    )
    namespace["verify_and_update_config"].__func__(None, config)
    assert cache.block_size == cache.mamba_block_size
    assert cache.block_size % 128 == 0
    assert cache.block_size >= (requested_block_size or 0)
    expected_state_bytes = sum(math.prod(shape) * dtype.itemsize for shape, dtype in zip(state_shapes, state_dtypes))
    assert cache.mamba_page_size_padded == cache.block_size * 576 * 2
    assert cache.mamba_page_size_padded >= expected_state_bytes
