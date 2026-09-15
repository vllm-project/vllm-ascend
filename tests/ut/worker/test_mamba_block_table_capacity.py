# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""MRV2 must reserve every speculative Mamba state column before allocation."""

import ast
import math
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import vllm
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

ASCEND_ROOT = Path(__file__).resolve().parents[3]
VLLM_ROOT = Path(vllm.__file__).resolve().parent


def _load_function(path, name, namespace, class_name=None):
    """Execute production Python logic without importing the NPU model runner."""
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    if class_name is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _unwrap(config):
    function = _load_function(
        ASCEND_ROOT / "vllm_ascend/worker/v2/attn_utils.py",
        "unwrap_mamba_kv_cache_groups",
        {"replace": replace, "MambaSpec": MambaSpec, "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs},
    )
    return function(config)


def _mamba_spec(block_size=72320, num_speculative_blocks=3, mode="none"):
    return MambaSpec(
        block_size=block_size,
        shapes=((4, 8),),
        dtypes=(torch.float32,),
        mamba_cache_mode=mode,
        num_speculative_blocks=num_speculative_blocks,
    )


def _wrapped_group(spec, names):
    wrapped = UniformTypeKVCacheSpecs(block_size=spec.block_size, kv_cache_specs=dict.fromkeys(names, spec))
    return KVCacheGroupSpec(layer_names=names, kv_cache_spec=wrapped)


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
@pytest.mark.parametrize("num_speculative_blocks", [0, 1, 3])
def test_upstream_mrv2_allocates_all_wrapped_mamba_state_columns(enable_prefix_caching, num_speculative_blocks):
    max_model_len = 72320
    block_size = 1024 if enable_prefix_caching else max_model_len
    spec = _mamba_spec(block_size, num_speculative_blocks, "align" if enable_prefix_caching else "none")
    config = KVCacheConfig(16, [], [_wrapped_group(spec, ["mamba.0", "mamba.1"])])
    config = _unwrap(config)

    # Run the actual upstream initializer and stop at its first BlockTables
    # construction, before it allocates device tensors or initializes a model.
    width = _load_function(
        VLLM_ROOT / "v1/worker/block_table.py",
        "get_block_table_width",
        {"math": math, "cdiv": lambda x, y: (x + y - 1) // y},
    )
    width_spy = MagicMock(wraps=width)
    allocated = {}

    class BlockTablesReached(Exception):
        pass

    def record_block_tables(**kwargs):
        allocated.update(kwargs)
        raise BlockTablesReached

    # Main checks these types before constructing BlockTables. This fixture
    # has only Mamba groups and no speculator; neither branch is exercised.
    class CircularBufferSpec:
        pass

    class DraftModelSpeculator:
        pass

    initialize = _load_function(
        VLLM_ROOT / "v1/worker/gpu/model_runner.py",
        "initialize_kv_cache",
        {
            "deepcopy": deepcopy,
            "cdiv": lambda x, y: (x + y - 1) // y,
            "MambaSpec": MambaSpec,
            "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
            "CircularBufferSpec": CircularBufferSpec,
            "DraftModelSpeculator": DraftModelSpeculator,
            "get_block_table_width": width_spy,
            "init_attn_backend": MagicMock(return_value=([], MagicMock(), [block_size])),
            "maybe_create_adaptive_verification_manager": MagicMock(return_value=None),
            "get_max_chunk_logits": MagicMock(return_value=1),
            "BlockTables": record_block_tables,
        },
        class_name="GPUModelRunner",
    )
    cache_config = SimpleNamespace(
        enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode="align" if enable_prefix_caching else "none",
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        dcp_size=1,
        dcp_rank=0,
        cp_interleave=1,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        vllm_config=SimpleNamespace(cache_config=cache_config),
        device=torch.device("cpu"),
        model_state=MagicMock(),
        speculator=None,
        req_states=MagicMock(),
        input_buffers=SimpleNamespace(query_start_loc=None),
        vocab_size=1,
        max_num_reqs=2,
        max_num_tokens=8,
    )
    with pytest.raises(BlockTablesReached):
        initialize(runner, config)

    expected_width = (71 if enable_prefix_caching else 1) + num_speculative_blocks
    assert allocated["max_num_blocks_per_group"] == [expected_width]
    width_spy.assert_called_once_with(expected_width, block_size, token_alignment=None)


def test_unwrap_preserves_allocation_descriptors_group_order_and_other_specs():
    attention = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=4, dtype=torch.float32)
    attention_group = _wrapped_group(attention, ["attention.0", "attention.1"])
    mamba = _mamba_spec()
    mamba_group = _wrapped_group(mamba, ["mamba.0", "mamba.1"])
    mamba_group.is_eagle_group = True
    plain_group = KVCacheGroupSpec(layer_names=["mamba.2"], kv_cache_spec=mamba)
    num_blocks = 16
    layer_stride = num_blocks * mamba.page_size_bytes
    descriptor_kwargs = {"size": 32 + layer_stride, "offset": 32, "block_stride": mamba.page_size_bytes}
    descriptor_fields = {field.name for field in fields(KVCacheTensor)}
    layer_field = "layers" if "layers" in descriptor_fields else "shared_by"
    descriptor_kwargs[layer_field] = ["mamba.0"]
    if "layer_stride" in descriptor_fields:
        descriptor_kwargs["layer_stride"] = layer_stride
    descriptors = [KVCacheTensor(**descriptor_kwargs)]
    original = KVCacheConfig(num_blocks, descriptors, [attention_group, mamba_group, plain_group])
    original_snapshot = deepcopy(original)

    normalized = _unwrap(original)

    assert normalized is not original
    assert original == original_snapshot
    assert normalized.num_blocks == original.num_blocks
    assert normalized.kv_cache_tensors is descriptors
    assert normalized.kv_cache_groups[0] is attention_group
    assert normalized.kv_cache_groups[2] is plain_group
    assert normalized.kv_cache_groups[1].kv_cache_spec is mamba
    assert normalized.kv_cache_groups[1].is_eagle_group
    assert [group.layer_names for group in normalized.kv_cache_groups] == [
        group.layer_names for group in original.kv_cache_groups
    ]


@pytest.mark.parametrize("is_vllm_028", [True, False])
def test_npu_model_runner_only_normalizes_mamba_groups_for_vllm_028(is_vllm_028):
    mamba = _mamba_spec()
    original = KVCacheConfig(16, [], [_wrapped_group(mamba, ["mamba.0", "mamba.1"])])
    original_snapshot = deepcopy(original)

    class ParentInitializationReached(Exception):
        pass

    parent_initialize = MagicMock(side_effect=ParentInitializationReached)
    version_is = MagicMock(return_value=is_vllm_028)
    unwrap = MagicMock(wraps=_unwrap)
    initialize = _load_function(
        ASCEND_ROOT / "vllm_ascend/worker/v2/model_runner.py",
        "initialize_kv_cache",
        {
            "vllm_version_is": version_is,
            "unwrap_mamba_kv_cache_groups": unwrap,
            "graph_manager_wrapper": lambda runner: nullcontext(),
            "super": lambda: SimpleNamespace(initialize_kv_cache=parent_initialize),
        },
        class_name="NPUModelRunner",
    )

    with pytest.raises(ParentInitializationReached):
        initialize(SimpleNamespace(), original)

    parent_initialize.assert_called_once()
    version_is.assert_called_once_with("0.28.0")
    (parent_config,) = parent_initialize.call_args.args
    if is_vllm_028:
        unwrap.assert_called_once_with(original)
        assert parent_config is not original
        assert parent_config.kv_cache_groups[0].kv_cache_spec is mamba
    else:
        unwrap.assert_not_called()
        assert parent_config is original
    assert original == original_snapshot
    assert isinstance(original.kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
